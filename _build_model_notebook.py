"""Generator untuk text_mining_model.ipynb.

Jalankan:  python _build_model_notebook.py
Setiap sel ditulis sebagai string di bawah, lalu dirakit menjadi JSON notebook.
Edit sel di file ini (bukan di notebook) supaya notebook selalu bisa digenerate ulang.
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src.strip("\n")))


def code(src):
    CELLS.append(("code", src.strip("\n")))


# =========================================================================== #
md(r'''
# Text Mining Model -- Klasifikasi, Deteksi Duplikat Semantik & Ringkasan Berita Pengaduan

**Judul:** *Classification and Semantic Duplicate Detection of News Articles for Public Complaint
Categorization Using IndoBERT and Sentence-BERT: A Case Study of Public Complaints in Indonesia*

Notebook ini adalah **Tahap 2** setelah scraping. Input: `data/raw/<sumber>_raw.csv` (output 8 notebook scraping).

| Tahap | Isi | Output |
|---|---|---|
| 1 | Load & gabung CSV hasil scraping | `df_raw` |
| 2 | Preprocessing (teks minimal untuk transformer, teks Sastrawi untuk EDA/baseline) | `data/processed/news_clean.csv` |
| 3 | EDA | grafik & kata teratas |
| 4 | Weak labeling (keyword + skor isi) + review manual (human-in-the-loop) | `data/labeled/label_review.csv` |
| 5 | Klasifikasi: baseline TF-IDF vs **IndoBERT** fine-tuning | `models/indobert_classifier/` |
| 6 | Deteksi duplikat semantik: **Sentence-BERT** + kalibrasi threshold + clustering | `data/labeled/dup_pairs_labeled.csv` |
| 7 | Ringkasan multi-dokumen per cluster (extractive MMR; abstractive opsional) | |
| 8 | Fungsi pipeline end-to-end untuk artikel baru | `run_pipeline()` |
| 9 | Simpan hasil | `data/processed/final_dataset.csv`, `data/processed/clustered_summaries.csv` |

## Cara menjalankan
1. Buka folder project di VS Code, pilih kernel environment Anda.
2. `DEBUG_MODE = True` (sel Konfigurasi) -> **Restart & Run All** -> smoke test cepat (subset kecil, 1 epoch).
3. Kalau lolos, `DEBUG_MODE = False` -> **Restart & Run All** untuk run penuh.
4. Isi file review manual (lihat Tahap 4 & 6) bila ingin label/threshold yang tervalidasi manusia, lalu Run All lagi.

## Library (install sendiri, tidak ada sel `pip install`)
**GPU wajib.** PyTorch harus build CUDA. Untuk RTX seri 30/40 dengan driver terbaru:
```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers sentence-transformers Sastrawi scikit-learn pandas numpy matplotlib tqdm
```
Opsional: `wordcloud` (grafik word cloud), `sentencepiece protobuf` (hanya untuk ringkasan abstractive).
''')

# =========================================================================== #
code(r'''
# [CELL] imports + cek GPU (WAJIB CUDA, tidak ada fallback CPU)
# Library: pandas numpy scikit-learn matplotlib tqdm torch(CUDA) transformers sentence-transformers Sastrawi
import os, re, json, math, random, time, warnings, unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_colwidth", 120)

if not torch.cuda.is_available():
    raise RuntimeError(
        f"CUDA tidak terdeteksi (torch {torch.__version__}). Notebook ini WAJIB GPU.\n"
        "Kalau versi torch berakhiran '+cpu', install ulang build CUDA:\n"
        "  pip uninstall -y torch\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "Lalu restart kernel."
    )

DEVICE = "cuda"
_props = torch.cuda.get_device_properties(0)
VRAM_GB = _props.total_memory / 1024**3
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {_props.name} | VRAM {VRAM_GB:.1f} GB")
''')

# =========================================================================== #
code(r'''
# [CELL] konfigurasi -- semua knob ada di sini
DEBUG_MODE = True          # True: smoke test (subset kecil, 1 epoch). False: run penuh.
SEED = 42

# ---- Path ----
ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
LABELED_DIR = ROOT / "data" / "labeled"
MODEL_DIR = ROOT / "models" / "indobert_classifier"
for _d in (PROC_DIR, LABELED_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)
EXPECTED_SOURCES = ["detikcom", "kompas", "okezone", "liputan6", "cnnindonesia", "kumparan", "merdeka", "tribunnews"]

# ---- Kategori (9 kelas) ----
COMPLAINT_CATEGORIES = ["Sampah", "Jalan", "Saluran Air/Drainase", "Fasos/Fasum", "Ketertiban Umum",
                        "Lampu/PJU", "Banjir", "Transportasi/Lalu Lintas"]
NON_COMPLAINT = "Bukan Pengaduan"
CATEGORIES = COMPLAINT_CATEGORIES + [NON_COMPLAINT]

# ---- Preprocessing ----
MIN_CONTENT_CHARS = 200

# ---- Weak labeling ----
TITLE_WEIGHT = 3           # kemunculan istilah di judul bernilai 3x
REL_MIN_SCORE = 3          # skor minimal agar artikel dianggap membahas kategori tsb
MIN_CLASS_COUNT = 10       # kelas dengan data < ini dibuang dari klasifikasi
REVIEW_PER_CLASS = 15      # sampel acak per kelas untuk review manual
REVIEW_MAX_RELABELED = 150 # maksimal artikel hasil relabel yang ikut direview

# ---- Klasifikasi IndoBERT ----
INDOBERT_NAME = "indobenchmark/indobert-base-p1"
if VRAM_GB < 5:
    TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS, SBERT_BATCH_SIZE, MAX_LENGTH = 4, 4, 32, 256
elif VRAM_GB < 7:
    TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS, SBERT_BATCH_SIZE, MAX_LENGTH = 8, 2, 32, 256
elif VRAM_GB < 10:
    TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS, SBERT_BATCH_SIZE, MAX_LENGTH = 16, 1, 64, 256
else:
    TRAIN_BATCH_SIZE, GRAD_ACCUM_STEPS, SBERT_BATCH_SIZE, MAX_LENGTH = 32, 1, 128, 384
EVAL_BATCH_SIZE = TRAIN_BATCH_SIZE * 2
EPOCHS = 1 if DEBUG_MODE else 6
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
USE_CLASS_WEIGHTS = True   # bobot loss untuk kelas yang sedikit
DEBUG_SAMPLE_PER_CLASS = 30

# ---- Deteksi duplikat semantik (SBERT) ----
SBERT_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SBERT_DOC_CHARS = 600       # judul + 600 karakter awal isi untuk embedding dokumen
DUP_TOP_K = 10              # tetangga terdekat per artikel (util.semantic_search)
CANDIDATE_MIN_SIM = 0.50    # pasangan kandidat minimal
DUP_THRESHOLD_DEFAULT = 0.80  # dipakai bila belum ada pasangan berlabel manual
MIN_LABELED_PAIRS = 30
DUP_PAIRS_PER_BIN = 15
TIME_WINDOW_DAYS = 3        # duplikat = kejadian sama -> terbit berdekatan
NAT_MARGIN = 0.05           # tanggal kosong: butuh similarity threshold + margin
DUP_REQUIRE_SAME_CATEGORY = True
CLUSTER_METHOD = "complete"  # "complete" (tiap pasangan dalam cluster >= threshold) atau "components"

# ---- Ringkasan ----
SUMMARY_MIN_SENTENCES = 3
SUMMARY_MAX_SENTENCES = 5
MMR_LAMBDA = 0.7
POSITION_BONUS = 0.05       # kalimat awal berita (lead) sedikit diutamakan
SENT_MIN_CHARS, SENT_MAX_CHARS = 40, 400
MAX_SENTS_PER_DOC = 25
SENT_DUP_SIM = 0.90         # kalimat nyaris identik antar sumber tidak dipilih dua kali
USE_ABSTRACTIVE = False     # opsional; butuh sentencepiece + VRAM cukup
ABSTRACTIVE_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
ABSTRACTIVE_MIN_VRAM = 4.0
PIPELINE_DEMO_N = 40

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"DEBUG_MODE={DEBUG_MODE} | batch {TRAIN_BATCH_SIZE} x accum {GRAD_ACCUM_STEPS} | max_len {MAX_LENGTH} | epochs {EPOCHS}")
''')

# =========================================================================== #
md(r'''
## Tahap 1 -- Load & gabung data hasil scraping
Notebook berhenti kalau belum ada CSV hasil scraping (data harus hasil scraping sendiri).
''')

code(r'''
# [CELL] load data
raw_files = sorted(RAW_DIR.glob("*_raw.csv"))
if not raw_files:
    raise FileNotFoundError(f"Tidak ada *_raw.csv di {RAW_DIR}. Jalankan notebook scraping dulu.")

frames, counts = [], {}
for f in raw_files:
    d = pd.read_csv(f, encoding="utf-8-sig")
    src = f.stem.replace("_raw", "")
    if "source" not in d.columns:
        d["source"] = src
    d["source"] = d["source"].fillna(src)
    frames.append(d)
    counts[src] = len(d)

df_raw = pd.concat(frames, ignore_index=True)
before = len(df_raw)
df_raw = df_raw.drop_duplicates(subset="url").reset_index(drop=True)
print(pd.Series(counts, name="artikel").to_string())
print(f"\nTotal {before} baris -> {len(df_raw)} setelah buang URL duplikat")
missing = [s for s in EXPECTED_SOURCES if s not in counts]
if missing:
    print(f"PERINGATAN: belum ada CSV untuk {missing} (dilewati)")
''')

# =========================================================================== #
md(r'''
## Tahap 2 -- Preprocessing
Dua versi teks:
- **`content_clean`** (minimal): hanya buang boilerplate ("Baca juga", URL, tag embed, inisial redaksi). Huruf besar,
  tanda baca, dan kata hubung tetap ada karena IndoBERT & SBERT butuh konteks kalimat utuh.
- **`text_eda`**: lowercase, tanpa angka/tanda baca, stopword Sastrawi dibuang, di-*stemming*. Untuk EDA & baseline TF-IDF.
''')

code(r'''
# [CELL] fungsi preprocessing
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ImportError as e:
    raise ImportError("Sastrawi belum terpasang: pip install Sastrawi") from e

BOILERPLATE_PATTERNS = [re.compile(p) for p in [
    r"\[Gambas:[^\]]*\]",
    r"(?i)\b(baca juga|baca selengkapnya|simak juga|lihat juga|simak video|tonton juga)\s*:[^.!?\n]{0,150}",
    r"(?i)\bADVERTISEMENT\b",
    r"(?i)SCROLL TO CONTINUE WITH CONTENT",
    r"https?://\S+",
    r"\bwww\.\S+",
    r"\S+@\S+\.\S+",
    r"\(\s*[a-z]{2,6}\s*/\s*[a-z]{2,6}\s*\)",   # inisial redaksi, mis. (rdp/imk)
    r"^kumparan[A-Z]+\s+\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}\s+WIB\s*\u00b7\s*waktu baca \d+ menit\s*",  # header kumparan
]]
TITLE_SUFFIX_RE = re.compile(
    r"\s*[:|\-\u2013]\s*(Okezone|Kompas|detik|Liputan6|CNN Indonesia|merdeka|kumparan|Tribunnews)[^:|]*$", re.I)
# Dateline di awal berita, mis. "JAKARTA, KOMPAS.com - ", "Liputan6.com, Jakarta - ", "SURABAYA - "
DATELINE_RES = [
    re.compile(r"^[^.!?]{0,60}?\.(?:com|co\.id|id)\b[^\-\u2013\u2014]{0,30}[\-\u2013\u2014]\s*"),
    re.compile(r"^[A-Z]{3,}(?: [A-Z]{3,}){0,2}\s*[\-\u2013\u2014]\s*"),
]


def clean_text_minimal(text):
    """Bersihkan boilerplate tapi pertahankan kalimat asli (untuk transformer)."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for pat in DATELINE_RES:
        text = pat.sub("", text, count=1)
    return text


def clean_title(title):
    title = clean_text_minimal(title)
    return TITLE_SUFFIX_RE.sub("", title).strip()


_STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | {
    "baca", "juga", "kata", "ujar", "jelas", "tutur", "ucap", "wib", "tersebut", "yakni", "pun", "nya"}
_STEMMER = StemmerFactory().create_stemmer()
_STEM_CACHE = {}


def _stem(word):
    if word not in _STEM_CACHE:
        _STEM_CACHE[word] = _STEMMER.stem(word)
    return _STEM_CACHE[word]


def clean_text_eda(text):
    """Lowercase, buang non-huruf, stopword, lalu stemming (cache per kata -> cepat)."""
    text = re.sub(r"[^a-z\s]", " ", str(text).lower())
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOPWORDS]
    tokens = [_stem(t) for t in tokens]
    return " ".join(t for t in tokens if len(t) > 2 and t not in _STOPWORDS)
''')

code(r'''
# [CELL] terapkan preprocessing
df = df_raw.copy()
df["title"] = df["title"].fillna("").map(clean_title)
df["content_clean"] = df["content"].fillna("").map(clean_text_minimal)
df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta")

n0 = len(df)
df = df[(df["title"].str.len() > 0) & (df["content_clean"].str.len() >= MIN_CONTENT_CHARS)]
n1 = len(df)
df = df.drop_duplicates(subset="content_clean")
n2 = len(df)
print(f"{n0} -> {n1} (buang judul kosong / isi < {MIN_CONTENT_CHARS} char) -> {n2} (buang isi identik)")

# (tqdm.pandas()/progress_map tidak kompatibel dengan pandas 3 -> pakai list comprehension)
df["text_eda"] = [clean_text_eda(t) for t in tqdm(df["title"] + " " + df["content_clean"], desc="Stemming")]
df = df.reset_index(drop=True)
df["doc_id"] = np.arange(len(df))

df.drop(columns=["content"]).to_csv(PROC_DIR / "news_clean.csv", index=False, encoding="utf-8-sig")
print("Tersimpan:", PROC_DIR / "news_clean.csv")
df[["source", "title", "published_date"]].head()
''')

# =========================================================================== #
md("## Tahap 3 -- Exploratory Data Analysis")

code(r'''
# [CELL] EDA grafik
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
df["source"].value_counts().plot.bar(ax=ax[0, 0], title="Artikel per sumber")
df["keyword_category"].fillna("(kosong)").value_counts().plot.barh(ax=ax[0, 1], title="Kategori keyword saat scraping")
df["content_clean"].str.len().clip(upper=12000).plot.hist(bins=50, ax=ax[1, 0], title="Panjang isi (karakter, dipotong 12k)")
monthly = df["published_date"].dt.tz_localize(None).dt.to_period("M").value_counts().sort_index()
monthly.plot(ax=ax[1, 1], marker="o", title="Artikel per bulan terbit")
plt.tight_layout(); plt.show()

print("Rentang tanggal:", df["published_date"].min(), "->", df["published_date"].max(),
      f"| tanggal kosong: {df['published_date'].isna().sum()}")
print("Median panjang isi:", int(df["content_clean"].str.len().median()), "karakter")
''')

code(r'''
# [CELL] kata teratas per kategori keyword (+ word cloud opsional)
for cat, g in df.groupby(df["keyword_category"].fillna("(kosong)")):
    top = Counter(" ".join(g["text_eda"]).split()).most_common(12)
    print(f"{cat:28s} ({len(g):4d}): " + ", ".join(w for w, _ in top))

try:
    from wordcloud import WordCloud
    wc = WordCloud(width=1200, height=500, background_color="white", random_state=SEED)
    plt.figure(figsize=(14, 6)); plt.imshow(wc.generate(" ".join(df["text_eda"])))
    plt.axis("off"); plt.title("Word cloud seluruh korpus"); plt.show()
except ImportError:
    print("(wordcloud tidak terpasang -- grafik word cloud dilewati)")
''')

# =========================================================================== #
md(r'''
## Tahap 4 -- Weak labeling + review manual
`keyword_category` dari scraper hanya **petunjuk** (keyword pencarian), bukan label final. Aturan label lemah:

1. `Neutral` (dicari dengan keyword netral) -> **Bukan Pengaduan**.
2. Kategori keyword dikonfirmasi bila skor istilah kategori itu di judul (x3) + isi >= `REL_MIN_SCORE`.
3. Kalau tidak terkonfirmasi (termasuk `General`), pakai kategori dengan skor tertinggi bila >= `REL_MIN_SCORE`.
4. Selain itu -> **Bukan Pengaduan** (artikel tidak cukup membahas masalah fasilitas/layanan publik).

**Human-in-the-loop:** run pertama mengekspor `data/labeled/label_review.csv`. Buka di Excel, isi kolom
`label_manual` (salah satu dari 9 kategori) untuk baris yang labelnya salah, simpan sebagai CSV UTF-8, lalu Run All lagi.
Label manual menimpa label lemah. File ini tidak pernah ditimpa notebook.
''')

code(r'''
# [CELL] fungsi weak labeling
CATEGORY_TERMS = {
    "Sampah": ["sampah", "tps", "tpa", "tempat pembuangan", "pengangkutan sampah"],
    "Jalan": ["jalan rusak", "jalan berlubang", "jalan amblas", "aspal rusak", "jembatan rusak",
              "trotoar rusak", "jalan bergelombang", "perbaikan jalan", "lubang jalan", "amblas", "berlubang"],
    "Saluran Air/Drainase": ["drainase", "gorong-gorong", "selokan", "saluran air", "got", "tersumbat"],
    "Fasos/Fasum": ["fasilitas umum", "fasum", "fasos", "halte", "taman kota", "toilet umum", "jpo",
                    "jembatan penyeberangan", "terbengkalai"],
    "Ketertiban Umum": ["pkl", "pedagang kaki lima", "parkir liar", "bangunan liar", "pungli", "premanisme",
                        "preman", "balap liar", "juru parkir", "jukir", "kebisingan", "tawuran"],
    "Lampu/PJU": ["lampu jalan", "penerangan jalan", "pju", "lampu lalu lintas", "jalan gelap", "lampu mati"],
    "Banjir": ["banjir", "rob", "genangan", "luapan sungai", "kebanjiran", "tergenang"],
    "Transportasi/Lalu Lintas": ["macet", "kemacetan", "angkutan umum", "lalu lintas", "transjakarta", "krl",
                                 "transportasi umum", "ojek online", "ojol", "angkot", "rekayasa lalu lintas"],
}
_TERM_RE = {cat: [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms] for cat, terms in CATEGORY_TERMS.items()}


def category_scores(title, content):
    t, c = str(title).lower(), str(content).lower()
    return {cat: sum(TITLE_WEIGHT * len(p.findall(t)) + len(p.findall(c)) for p in pats)
            for cat, pats in _TERM_RE.items()}


def weak_label(title, content, keyword_category):
    """Return (label, label_source, score)."""
    scores = category_scores(title, content)
    best_cat = max(scores, key=scores.get)
    if keyword_category == "Neutral":
        return NON_COMPLAINT, "neutral_keyword", 0
    if keyword_category in COMPLAINT_CATEGORIES and scores[keyword_category] >= REL_MIN_SCORE:
        return keyword_category, "keyword_confirmed", scores[keyword_category]
    if scores[best_cat] >= REL_MIN_SCORE:
        return best_cat, "content_relabel", scores[best_cat]
    return NON_COMPLAINT, "low_relevance", scores[best_cat]
''')

code(r'''
# [CELL] terapkan label lemah + review manual
_lab = [weak_label(t, c, k) for t, c, k in zip(df["title"], df["content_clean"], df["keyword_category"])]
df["label"] = [x[0] for x in _lab]
df["label_source"] = [x[1] for x in _lab]
df["label_score"] = [x[2] for x in _lab]

print(df["label_source"].value_counts().to_string(), "\n")
print(pd.crosstab(df["keyword_category"].fillna("(kosong)"), df["label"]).to_string())

REVIEW_PATH = LABELED_DIR / "label_review.csv"
if REVIEW_PATH.exists():
    rev = pd.read_csv(REVIEW_PATH, encoding="utf-8-sig", dtype=str)
    rev["label_manual"] = rev["label_manual"].fillna("").str.strip()
    filled = rev[rev["label_manual"] != ""]
    bad = sorted(set(filled["label_manual"]) - set(CATEGORIES))
    if bad:
        raise ValueError(f"label_manual tidak valid di {REVIEW_PATH.name}: {bad}. Pilihan: {CATEGORIES}")
    manual = dict(zip(filled["url"], filled["label_manual"]))
    mask = df["url"].isin(manual)
    changed = (df.loc[mask, "label"] != df.loc[mask, "url"].map(manual)).sum()
    df.loc[mask, "label"] = df.loc[mask, "url"].map(manual)
    df.loc[mask, "label_source"] = "manual"
    print(f"\nLabel manual diterapkan: {mask.sum()} artikel ({changed} berubah dari label lemah)")
else:
    relab = df[df["label_source"].isin(["content_relabel", "low_relevance"])]
    relab = relab.sample(min(len(relab), REVIEW_MAX_RELABELED), random_state=SEED)
    strat = df.groupby("label", group_keys=False)[df.columns.tolist()].apply(
        lambda g: g.sample(min(len(g), REVIEW_PER_CLASS), random_state=SEED))
    rev = pd.concat([relab, strat]).drop_duplicates("url")
    rev = rev.assign(content_snippet=rev["content_clean"].str[:400], label_manual="")
    rev[["url", "source", "title", "content_snippet", "keyword_category", "label_source", "label", "label_manual"]] \
        .to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    print(f"\nFile review dibuat: {REVIEW_PATH} ({len(rev)} baris). Isi kolom label_manual bila label salah, lalu Run All lagi.")
''')

code(r'''
# [CELL] distribusi kelas final + subset DEBUG
dist = df["label"].value_counts()
print(dist.to_string())
rare = dist[dist < MIN_CLASS_COUNT].index.tolist()
if rare:
    print(f"PERINGATAN: kelas {rare} < {MIN_CLASS_COUNT} artikel -> dibuang dari klasifikasi")
    df = df[~df["label"].isin(rare)].reset_index(drop=True)

if DEBUG_MODE:
    idx = []
    for _, g in df.groupby("label"):
        idx += g.sample(min(len(g), DEBUG_SAMPLE_PER_CLASS), random_state=SEED).index.tolist()
    df = df.loc[sorted(idx)].reset_index(drop=True)
    print(f"DEBUG_MODE: subset {len(df)} artikel")

LABELS = [c for c in CATEGORIES if c in set(df["label"])]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

dist = df["label"].value_counts().reindex(LABELS)
plt.figure(figsize=(8, 4)); plt.barh(list(dist.index), dist.values); plt.gca().invert_yaxis()
plt.title("Distribusi label final"); plt.show()
''')

# =========================================================================== #
md(r'''
## Tahap 5 -- Klasifikasi
Split stratified 70/15/15. Baseline TF-IDF (LogReg & LinearSVC) dibandingkan dengan fine-tuning **IndoBERT**
(fp16, gradient accumulation, early stopping berdasarkan macro-F1 validasi). Metrik utama: **macro-F1**
(adil untuk kelas yang jumlahnya tidak seimbang).
''')

code(r'''
# [CELL] split data
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)
df["split"] = "train"
df.loc[df["doc_id"].isin(val_df["doc_id"]), "split"] = "val"
df.loc[df["doc_id"].isin(test_df["doc_id"]), "split"] = "test"
print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")
''')

code(r'''
# [CELL] baseline TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=50000)
X_tr = tfidf.fit_transform(train_df["text_eda"])
X_te = tfidf.transform(test_df["text_eda"])

results = []
for name, clf in [("TF-IDF + LogReg", LogisticRegression(max_iter=2000, class_weight="balanced")),
                  ("TF-IDF + LinearSVC", LinearSVC(class_weight="balanced"))]:
    clf.fit(X_tr, train_df["label_id"])
    p = clf.predict(X_te)
    results.append({"model": name, "accuracy": accuracy_score(test_df["label_id"], p),
                    "precision_macro": precision_score(test_df["label_id"], p, average="macro", zero_division=0),
                    "recall_macro": recall_score(test_df["label_id"], p, average="macro", zero_division=0),
                    "f1_macro": f1_score(test_df["label_id"], p, average="macro")})
pd.DataFrame(results).round(4)
''')

code(r'''
# [CELL] dataset IndoBERT
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)


def model_text(frame):
    return (frame["title"] + ". " + frame["content_clean"]).tolist()


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


tokenizer = AutoTokenizer.from_pretrained(INDOBERT_NAME)
collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(NewsDataset(model_text(train_df), train_df["label_id"], tokenizer),
                          batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collator, num_workers=0)
val_loader = DataLoader(NewsDataset(model_text(val_df), val_df["label_id"], tokenizer),
                        batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)
test_loader = DataLoader(NewsDataset(model_text(test_df), test_df["label_id"], tokenizer),
                         batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator, num_workers=0)
print(f"batch/epoch: train {len(train_loader)} | val {len(val_loader)} | test {len(test_loader)}")
''')

code(r'''
# [CELL] fine-tuning IndoBERT (fp16 + gradient accumulation + early stopping)
from sklearn.utils.class_weight import compute_class_weight

model = AutoModelForSequenceClassification.from_pretrained(
    INDOBERT_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id).to(DEVICE)

if USE_CLASS_WEIGHTS:
    cw = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_df["label_id"].values)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE))
else:
    loss_fn = torch.nn.CrossEntropyLoss()

no_decay = ("bias", "LayerNorm.weight")
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
], lr=LEARNING_RATE)
steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
total_steps = steps_per_epoch * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda")


def evaluate(mdl, loader):
    mdl.eval()
    preds, labs, probs, loss_sum, n = [], [], [], 0.0, 0
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for b in loader:
            b = {k: v.to(DEVICE) for k, v in b.items()}
            y = b.pop("labels")
            logits = mdl(**b).logits.float()
            loss_sum += loss_fn(logits, y).item() * len(y); n += len(y)
            pr = torch.softmax(logits, dim=-1)
            probs.append(pr.cpu().numpy()); preds += pr.argmax(-1).tolist(); labs += y.tolist()
    m = {"loss": loss_sum / max(n, 1), "accuracy": accuracy_score(labs, preds),
         "f1_macro": f1_score(labs, preds, average="macro")}
    return m, np.array(preds), np.array(labs), np.concatenate(probs)


torch.cuda.reset_peak_memory_stats()
best_f1, patience_left, history = -1.0, EARLY_STOPPING_PATIENCE, []
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train(); optimizer.zero_grad(set_to_none=True); run_loss = 0.0
    for step, b in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
        b = {k: v.to(DEVICE) for k, v in b.items()}
        y = b.pop("labels")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(**b).logits
        loss = loss_fn(logits.float(), y) / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        run_loss += loss.item() * GRAD_ACCUM_STEPS
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step()

    val_m, *_ = evaluate(model, val_loader)
    history.append({"epoch": epoch, "train_loss": run_loss / len(train_loader), **{f"val_{k}": v for k, v in val_m.items()}})
    print(f"epoch {epoch}: train_loss {history[-1]['train_loss']:.4f} | val_loss {val_m['loss']:.4f} "
          f"| val_acc {val_m['accuracy']:.4f} | val_f1 {val_m['f1_macro']:.4f}")
    if val_m["f1_macro"] > best_f1:
        best_f1, patience_left = val_m["f1_macro"], EARLY_STOPPING_PATIENCE
        model.save_pretrained(MODEL_DIR); tokenizer.save_pretrained(MODEL_DIR)
        print(f"  -> model terbaik disimpan ke {MODEL_DIR}")
    else:
        patience_left -= 1
        if patience_left == 0:
            print("  -> early stopping"); break

print(f"Selesai {time.time() - t0:.0f} s | val macro-F1 terbaik {best_f1:.4f} | "
      f"puncak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
hist = pd.DataFrame(history)
hist.plot(x="epoch", y=["train_loss", "val_loss"], marker="o", title="Loss"); plt.show()
''')

code(r'''
# [CELL] reload model terbaik + evaluasi test
del model, optimizer, scheduler
torch.cuda.empty_cache()
clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

test_m, test_pred, test_lab, _ = evaluate(clf_model, test_loader)
print(classification_report(test_lab, test_pred, labels=range(len(LABELS)), target_names=LABELS, digits=4, zero_division=0))

cm = confusion_matrix(test_lab, test_pred, labels=range(len(LABELS)))
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
for i in range(len(LABELS)):
    for j in range(len(LABELS)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Prediksi"); ax.set_ylabel("Label asli"); ax.set_title("Confusion matrix IndoBERT (test)")
plt.colorbar(im); plt.tight_layout(); plt.show()

results.append({"model": "IndoBERT (fine-tuned)", "accuracy": test_m["accuracy"],
                "precision_macro": precision_score(test_lab, test_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(test_lab, test_pred, average="macro", zero_division=0),
                "f1_macro": test_m["f1_macro"]})
comparison = pd.DataFrame(results).round(4)
(MODEL_DIR / "label_map.json").write_text(json.dumps({"labels": LABELS}, ensure_ascii=False, indent=2), encoding="utf-8")
(MODEL_DIR / "metrics.json").write_text(json.dumps({"comparison": comparison.to_dict("records"), "history": history},
                                                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
comparison
''')

code(r'''
# [CELL] prediksi kategori untuk semua artikel
def predict_categories(texts, mdl=None, tok=None, batch_size=None):
    mdl, tok, batch_size = mdl or clf_model, tok or clf_tokenizer, batch_size or EVAL_BATCH_SIZE
    labels, confs = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LENGTH, padding=True,
                      return_tensors="pt").to(DEVICE)
            pr = torch.softmax(mdl(**enc).logits.float(), dim=-1)
            c, k = pr.max(-1)
            labels += [mdl.config.id2label[int(x)] for x in k]; confs += c.tolist()
    return labels, confs


df["pred_category"], df["pred_confidence"] = predict_categories(model_text(df))
print("Akurasi pada seluruh data (termasuk train, hanya sanity check):",
      round((df["pred_category"] == df["label"]).mean(), 4))
df["pred_category"].value_counts()
''')

# =========================================================================== #
md(r'''
## Tahap 6 -- Deteksi duplikat semantik (Sentence-BERT)
Duplikat = berita tentang **kejadian/keluhan yang sama**, biasanya dari sumber berbeda.

1. Embedding dokumen (judul + awal isi) dengan SBERT multibahasa, cosine similarity di GPU.
2. Pasangan kandidat diambil dengan `util.semantic_search` (top-k tetangga).
3. **Kalibrasi threshold empiris:** run pertama mengekspor `data/labeled/dup_pairs_labeled.csv` berisi pasangan dari
   berbagai rentang similarity. Isi kolom `is_duplicate` (1 = kejadian sama, 0 = beda), simpan, Run All lagi ->
   threshold dipilih otomatis berdasarkan F1 terbaik. Sebelum diisi, dipakai `DUP_THRESHOLD_DEFAULT`.
4. Clustering *complete-linkage*: setiap pasangan dalam satu cluster harus >= threshold, terbit dalam
   `TIME_WINDOW_DAYS` hari, dan (opsional) berkategori sama. Ini mencegah efek rantai A~B~C yang menyatukan berita tak terkait.
''')

code(r'''
# [CELL] embedding dokumen SBERT
from sentence_transformers import SentenceTransformer, util

sbert = SentenceTransformer(SBERT_NAME, device=DEVICE)


def doc_repr(frame):
    return (frame["title"] + ". " + frame["content_clean"].str[:SBERT_DOC_CHARS]).tolist()


doc_emb = sbert.encode(doc_repr(df), batch_size=SBERT_BATCH_SIZE, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)
sim = (doc_emb @ doc_emb.T).float().cpu().numpy()
iu = np.triu_indices(len(df), k=1)
plt.hist(sim[iu], bins=100, log=True); plt.title("Distribusi cosine similarity semua pasangan (log)")
plt.xlabel("cosine similarity"); plt.show()
''')

code(r'''
# [CELL] kandidat pasangan + kalibrasi threshold
hits = util.semantic_search(doc_emb, doc_emb, top_k=min(DUP_TOP_K + 1, len(df)))
pair_set = {}
for i, hh in enumerate(hits):
    for h in hh:
        j = h["corpus_id"]
        if j != i and h["score"] >= CANDIDATE_MIN_SIM:
            pair_set[(min(i, j), max(i, j))] = float(h["score"])
cand = pd.DataFrame([(a, b, s) for (a, b), s in pair_set.items()], columns=["i", "j", "sim"])
for side, col in (("a", "i"), ("b", "j")):
    sub = df.loc[cand[col], ["url", "source", "title", "published_date"]].reset_index(drop=True)
    for c in sub.columns:
        cand[f"{c}_{side}"] = sub[c].values
cand["days_apart"] = (cand["published_date_a"] - cand["published_date_b"]).abs().dt.total_seconds() / 86400
print(f"{len(cand)} pasangan kandidat (sim >= {CANDIDATE_MIN_SIM})")


def sweep_threshold(sims, y):
    rows = []
    for t in np.arange(0.50, 0.99, 0.01):
        p = (sims >= t).astype(int)
        rows.append({"threshold": round(t, 2), "precision": precision_score(y, p, zero_division=0),
                     "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0)})
    return pd.DataFrame(rows)


PAIRS_PATH = LABELED_DIR / "dup_pairs_labeled.csv"
DUP_THRESHOLD, labeled_pairs = DUP_THRESHOLD_DEFAULT, None
if PAIRS_PATH.exists():
    lp = pd.read_csv(PAIRS_PATH, encoding="utf-8-sig")
    lp = lp[pd.to_numeric(lp["is_duplicate"], errors="coerce").isin([0, 1])].copy()
    lp["is_duplicate"] = lp["is_duplicate"].astype(int)
    if len(lp) >= MIN_LABELED_PAIRS and lp["is_duplicate"].nunique() == 2:
        labeled_pairs = lp
        sw = sweep_threshold(lp["sim"].values, lp["is_duplicate"].values)
        best = sw.loc[sw["f1"].idxmax()]
        DUP_THRESHOLD = float(best["threshold"])
        sw.plot(x="threshold", y=["precision", "recall", "f1"], title="Kalibrasi threshold duplikat"); plt.show()
        print(f"Threshold terkalibrasi dari {len(lp)} pasangan berlabel: {DUP_THRESHOLD:.2f} "
              f"(P {best['precision']:.3f} | R {best['recall']:.3f} | F1 {best['f1']:.3f})")
    else:
        print(f"{PAIRS_PATH.name}: baru {len(lp)} pasangan berlabel (butuh >= {MIN_LABELED_PAIRS}, dua kelas). "
              f"Pakai default {DUP_THRESHOLD_DEFAULT}.")
else:
    bins = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.01]
    cand["bin"] = pd.cut(cand["sim"], bins, right=False)
    sample = cand.groupby("bin", observed=True, group_keys=False)[cand.columns.tolist()].apply(
        lambda g: g.sample(min(len(g), DUP_PAIRS_PER_BIN), random_state=SEED))
    cols = ["sim", "days_apart", "source_a", "title_a", "source_b", "title_b", "url_a", "url_b"]
    sample[cols].assign(is_duplicate="").round({"sim": 4, "days_apart": 1}) \
        .to_csv(PAIRS_PATH, index=False, encoding="utf-8-sig")
    print(f"File kalibrasi dibuat: {PAIRS_PATH} ({len(sample)} pasangan). Isi is_duplicate (1/0), lalu Run All lagi.")
    print(f"Sementara pakai threshold default {DUP_THRESHOLD_DEFAULT} (BELUM terkalibrasi).")
''')

code(r'''
# [CELL] fungsi clustering duplikat
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def build_duplicate_clusters(sim, dates, categories, threshold, window_days=TIME_WINDOW_DAYS,
                             same_category=DUP_REQUIRE_SAME_CATEGORY, method=CLUSTER_METHOD):
    """Return array cluster id (0..k-1) per dokumen; cluster 0 = paling besar."""
    n = len(sim)
    if n < 2:
        return np.zeros(n, dtype=int)
    days = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce", utc=True).dt.tz_localize(None)
    d = (days - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy(dtype=float) / 86400
    diff = np.abs(d[:, None] - d[None, :])
    nat = np.isnan(diff)
    allowed = np.where(nat, sim >= threshold + NAT_MARGIN, diff <= window_days)
    if same_category:
        cat = np.asarray(categories)
        allowed &= cat[:, None] == cat[None, :]
    if method == "complete":
        dist = (1.0 - sim).astype(np.float64)
        dist[~allowed] = 2.0
        np.fill_diagonal(dist, 0.0)
        labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="complete",
                                         distance_threshold=1.0 - threshold).fit_predict(dist)
    else:
        adj = (sim >= threshold) & allowed
        np.fill_diagonal(adj, False)
        _, labels = connected_components(csr_matrix(adj), directed=False)
    order = pd.Series(labels).value_counts().index          # urutkan dari cluster terbesar
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[l] for l in labels])
''')

code(r'''
# [CELL] jalankan clustering + evaluasi
df["cluster_id"] = build_duplicate_clusters(sim, df["published_date"], df["pred_category"], DUP_THRESHOLD)
df["cluster_size"] = df.groupby("cluster_id")["doc_id"].transform("count")

multi = df[df["cluster_size"] > 1]
cross = multi.groupby("cluster_id")["source"].nunique()
print(f"threshold {DUP_THRESHOLD:.2f} | {df['cluster_id'].nunique()} cluster dari {len(df)} artikel")
print(f"cluster berisi >1 artikel: {multi['cluster_id'].nunique()} ({len(multi)} artikel) | "
      f"lintas sumber: {(cross > 1).sum()}")

if labeled_pairs is not None:
    cid = dict(zip(df["url"], df["cluster_id"]))
    lp = labeled_pairs[labeled_pairs["url_a"].isin(cid) & labeled_pairs["url_b"].isin(cid)]
    if len(lp):
        pred_dup = (lp["url_a"].map(cid) == lp["url_b"].map(cid)).astype(int)
        print(f"Evaluasi cluster vs {len(lp)} pasangan berlabel: P {precision_score(lp['is_duplicate'], pred_dup, zero_division=0):.3f}"
              f" | R {recall_score(lp['is_duplicate'], pred_dup, zero_division=0):.3f}"
              f" | F1 {f1_score(lp['is_duplicate'], pred_dup, zero_division=0):.3f}")

for c in multi["cluster_id"].drop_duplicates().head(5):
    g = df[df["cluster_id"] == c]
    print(f"\n[cluster {c}] {g['pred_category'].iloc[0]} | {len(g)} artikel")
    for _, r in g.iterrows():
        print(f"   - {r['source']:12s} {str(r['published_date'])[:10]}  {r['title'][:90]}")
''')

# =========================================================================== #
md(r'''
## Tahap 7 -- Ringkasan multi-dokumen per cluster
Setiap cluster (= satu kejadian) mendapat **satu ringkasan**. Cluster berisi satu artikel tetap diringkas (single-document).

**Extractive (utama), MMR:** kalimat dari semua artikel dalam cluster di-embed dengan SBERT. Kalimat dipilih berdasarkan
kemiripan dengan pusat cluster, dikurangi kemiripan dengan kalimat yang sudah terpilih (supaya informasi sama
dari beberapa sumber tidak berulang). Jumlah kalimat: 3 untuk satu artikel, bertambah per artikel sampai maksimal 5.

**Abstractive (opsional):** `USE_ABSTRACTIVE = True` untuk menulis ulang ringkasan dengan mT5 (cluster >1 artikel saja).
''')

code(r'''
# [CELL] fungsi ringkasan extractive
_ABBREV = r"(?:dr|ir|h|hj|prof|drs|st|no|jl|kec|kab|kel|dkk|sdr|tbk|pt|bpk|ny|tn|dll|dsb|km|rp|sp|mr|mrs)"
_ABBREV_RE = re.compile(r"\b(" + _ABBREV + r")\.", re.I)
_SPLIT_RE = re.compile(r"(?<=[.!?][\"\u201d'\u2019])\s+(?=[A-Z0-9\"\u201c'(])|(?<=[.!?])\s+(?=[A-Z0-9\"\u201c'(])")
_SENT_JUNK_RE = re.compile(r"(?i)baca juga|simak video|lihat juga|whatsapp channel|google news|ikuti kami|download aplikasi")


def split_sentences(text):
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", str(text))
    text = _ABBREV_RE.sub(lambda m: m.group(1) + "<DOT>", text)
    out = []
    for s in _SPLIT_RE.split(text):
        s = s.replace("<DOT>", ".").strip()
        if SENT_MIN_CHARS <= len(s) <= SENT_MAX_CHARS and not _SENT_JUNK_RE.search(s):
            out.append(s)
    return out


def mmr_select(emb, centroid, k, lambda_=MMR_LAMBDA, bonus=None):
    """emb: (n,d) ternormalisasi. Return indeks kalimat terpilih (urutan pemilihan)."""
    relevance = emb @ centroid + (bonus if bonus is not None else 0.0)
    selected, cand = [], list(range(len(emb)))
    while cand and len(selected) < k:
        if selected:
            red = (emb[cand] @ emb[selected].T).max(axis=1)
            keep = red < SENT_DUP_SIM
            if not keep.any():
                break
            cand = [c for c, ok in zip(cand, keep) if ok]
            red = red[keep]
            scores = lambda_ * relevance[cand] - (1 - lambda_) * red
        else:
            scores = relevance[cand]
        best = cand[int(np.argmax(scores))]
        selected.append(best); cand.remove(best)
    return selected


def summarize_clusters(frame, encoder, batch_size=SBERT_BATCH_SIZE):
    """frame wajib punya cluster_id, title, content_clean, published_date, source, url, pred_category."""
    recs = []
    for di, (cid, txt) in enumerate(zip(frame["cluster_id"], frame["content_clean"])):
        for pos, s in enumerate(split_sentences(txt)[:MAX_SENTS_PER_DOC]):
            recs.append((cid, di, pos, s))
    sents = pd.DataFrame(recs, columns=["cluster_id", "doc_pos", "pos", "text"])
    s_emb = encoder.encode(sents["text"].tolist(), batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True) if len(sents) else np.zeros((0, 1))
    frame = frame.reset_index(drop=True)
    d_emb = encoder.encode(doc_repr(frame), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)
    date_rank = frame["published_date"].rank(method="first", na_option="bottom").to_numpy()

    rows = []
    for cid, g in tqdm(frame.groupby("cluster_id"), desc="Ringkas cluster"):
        idx = sents.index[sents["cluster_id"] == cid].to_numpy()
        n_docs = len(g)
        k = min(SUMMARY_MAX_SENTENCES, SUMMARY_MIN_SENTENCES + n_docs - 1)
        dc = d_emb[g.index].mean(0); dc /= np.linalg.norm(dc) + 1e-12
        rep = g.index[int(np.argmax(d_emb[g.index] @ dc))]
        if len(idx):
            e = s_emb[idx]
            cen = e.mean(0); cen /= np.linalg.norm(cen) + 1e-12
            bonus = np.where(sents.loc[idx, "pos"].to_numpy() < 2, POSITION_BONUS, 0.0)
            chosen = idx[mmr_select(e, cen, k, bonus=bonus)]
            chosen = sorted(chosen, key=lambda r: (date_rank[sents.at[r, "doc_pos"]], sents.at[r, "pos"]))
            summary = " ".join(sents.loc[chosen, "text"])
            ce = s_emb[chosen]
            redundancy = float((ce @ ce.T)[np.triu_indices(len(ce), 1)].mean()) if len(ce) > 1 else 0.0
        else:
            summary, redundancy, chosen = g["content_clean"].iloc[0][:300], 0.0, []
        rows.append({
            "cluster_id": cid, "category": g["pred_category"].mode().iloc[0], "n_articles": n_docs,
            "n_sources": g["source"].nunique(), "sources": ", ".join(sorted(g["source"].unique())),
            "date_start": g["published_date"].min(), "date_end": g["published_date"].max(),
            "representative_title": frame.at[rep, "title"], "titles": " || ".join(g["title"]),
            "urls": " ".join(g["url"]), "summary_extractive": summary, "n_summary_sentences": len(chosen),
            "compression_ratio": round(len(summary) / max(g["content_clean"].str.len().sum(), 1), 4),
            "redundancy": round(redundancy, 4)})
    return pd.DataFrame(rows).sort_values(["n_articles", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
''')

code(r'''
# [CELL] buat ringkasan semua cluster
summaries = summarize_clusters(df, sbert)
multi_s = summaries[summaries["n_articles"] > 1]
print(f"{len(summaries)} ringkasan | {len(multi_s)} dari cluster multi-artikel")
print(f"rata-rata kompresi {summaries['compression_ratio'].mean():.3f} | "
      f"redundansi antar kalimat ringkasan {multi_s['redundancy'].mean():.3f} (makin kecil makin beragam)")
for _, r in multi_s.head(3).iterrows():
    print(f"\n=== [{r['category']}] {r['representative_title']} ({r['n_articles']} artikel: {r['sources']})")
    print(r["summary_extractive"])
''')

code(r'''
# [CELL] ringkasan abstractive (opsional)
summaries["summary_abstractive"] = ""
if not USE_ABSTRACTIVE:
    print("USE_ABSTRACTIVE=False -> dilewati (ringkasan utama = extractive)")
elif VRAM_GB < ABSTRACTIVE_MIN_VRAM:
    print(f"VRAM {VRAM_GB:.1f} GB < {ABSTRACTIVE_MIN_VRAM} GB -> abstractive dilewati")
else:
    from transformers import AutoModelForSeq2SeqLM
    abs_tok = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL)
    abs_model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL).to(DEVICE).eval()  # fp32: mT5 tidak stabil di fp16
    targets = summaries.index[summaries["n_articles"] > 1]
    if DEBUG_MODE:
        targets = targets[:5]
    for i in tqdm(targets, desc="Abstractive"):
        enc = abs_tok(summaries.at[i, "summary_extractive"], truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = abs_model.generate(**enc, max_new_tokens=96, num_beams=4, no_repeat_ngram_size=3)
        summaries.at[i, "summary_abstractive"] = abs_tok.decode(out[0], skip_special_tokens=True)
    del abs_model; torch.cuda.empty_cache()
    print(summaries.loc[targets[:3], ["representative_title", "summary_abstractive"]].to_string())
''')

# =========================================================================== #
md(r'''
## Tahap 8 -- Pipeline end-to-end
`run_pipeline(articles)` menerima artikel mentah (kolom `title, content, published_date, source, url`) dan
mengembalikan `(artikel + kategori + cluster, ringkasan per cluster)` memakai model yang sudah dilatih.
Contoh di bawah memakai artikel terbaru sebagai simulasi "berita masuk".
''')

code(r'''
# [CELL] pipeline end-to-end
def run_pipeline(articles, threshold=None):
    a = articles.copy()
    a["title"] = a["title"].fillna("").map(clean_title)
    a["content_clean"] = a["content"].fillna("").map(clean_text_minimal)
    a["published_date"] = pd.to_datetime(a["published_date"], errors="coerce", utc=True).dt.tz_convert("Asia/Jakarta")
    a = a[(a["title"].str.len() > 0) & (a["content_clean"].str.len() >= MIN_CONTENT_CHARS)].reset_index(drop=True)
    if a.empty:
        return a, pd.DataFrame()
    a["pred_category"], a["pred_confidence"] = predict_categories(model_text(a))
    emb = sbert.encode(doc_repr(a), batch_size=SBERT_BATCH_SIZE, convert_to_tensor=True, normalize_embeddings=True)
    s = (emb @ emb.T).float().cpu().numpy()
    a["cluster_id"] = build_duplicate_clusters(s, a["published_date"], a["pred_category"], threshold or DUP_THRESHOLD)
    a["cluster_size"] = a.groupby("cluster_id")["url"].transform("count")
    return a, summarize_clusters(a, sbert)


demo_in = df_raw.assign(_d=pd.to_datetime(df_raw["published_date"], errors="coerce", utc=True)) \
    .sort_values("_d", ascending=False).head(PIPELINE_DEMO_N).drop(columns="_d")
demo_articles, demo_summaries = run_pipeline(demo_in)
print(demo_articles["pred_category"].value_counts().to_string())
demo_summaries[["category", "n_articles", "sources", "representative_title", "summary_extractive"]].head(5)
''')

# =========================================================================== #
md("## Tahap 9 -- Simpan hasil")

code(r'''
# [CELL] simpan output
final_cols = ["doc_id", "source", "url", "title", "published_date", "author", "category_section", "keyword_used",
              "keyword_category", "content_clean", "label", "label_source", "label_score", "split",
              "pred_category", "pred_confidence", "cluster_id", "cluster_size"]
final_cols = [c for c in final_cols if c in df.columns]
suffix = "_debug" if DEBUG_MODE else ""
df[final_cols].to_csv(PROC_DIR / f"final_dataset{suffix}.csv", index=False, encoding="utf-8-sig")
summaries.to_csv(PROC_DIR / f"clustered_summaries{suffix}.csv", index=False, encoding="utf-8-sig")
run_info = {"debug_mode": DEBUG_MODE, "n_articles": int(len(df)), "labels": LABELS,
            "dup_threshold": DUP_THRESHOLD, "dup_threshold_calibrated": labeled_pairs is not None,
            "n_clusters": int(df["cluster_id"].nunique()), "n_multi_clusters": int(len(multi_s)),
            "classification": comparison.to_dict("records")}
(PROC_DIR / f"run_summary{suffix}.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
for f in (f"final_dataset{suffix}.csv", f"clustered_summaries{suffix}.csv", f"run_summary{suffix}.json"):
    print("Tersimpan:", PROC_DIR / f)
print("Model IndoBERT:", MODEL_DIR)
''')

md(r'''
## Catatan untuk laporan
- **Label adalah weak label** (keyword pencarian + skor istilah di isi). Akurasi model mengukur kemampuan meniru label
  itu; validasi manual lewat `label_review.csv` membuat evaluasi lebih kuat. Sebutkan jumlah label manual di laporan.
- **Threshold duplikat** dipilih dari pasangan berlabel manual yang sama dengan yang dipakai evaluasi. Untuk laporan
  yang lebih ketat, pisahkan sebagian pasangan sebagai data uji.
- **Ringkasan extractive** dievaluasi secara intrinsik (rasio kompresi, redundansi). ROUGE butuh ringkasan referensi
  buatan manusia, yang belum ada di dataset ini.
''')


# =========================================================================== #
def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {}, "source": lines}
        if kind == "code":
            compile(src, f"<cell {i}>", "exec")   # gagal keras kalau ada syntax error
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Ditulis {path}: {len(CELLS)} sel ({sum(k == 'code' for k, _ in CELLS)} code)")


if __name__ == "__main__":
    build(Path(__file__).parent / "text_mining_model.ipynb")
