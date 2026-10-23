# Text Mining News Complaint

**Judul:** *Classification and Semantic Duplicate Detection of News Articles for Public
Complaint Categorization Using IndoBERT and Sentence-BERT: A Case Study of Public
Complaints in Indonesia*

Project ini mengumpulkan artikel berita dari 8 portal berita Indonesia, lalu:

1. **Klasifikasi** artikel ke kategori pengaduan masyarakat dengan **IndoBERT**.
2. **Deteksi duplikat semantik** (berita dengan kejadian/keluhan sama dari sumber
   berbeda) dengan **Sentence-BERT**.
3. **Penggabungan** artikel yang terdeteksi sama menjadi **satu ringkasan**
   (multi-document summarization: extractive sebagai metode utama, abstractive opsional).

---

## Struktur Folder

```
text-mining-news-complaint/
├── scraping/
│   ├── web_scraping_detikcom.ipynb
│   ├── web_scraping_kompas.ipynb
│   ├── web_scraping_liputan6.ipynb
│   ├── web_scraping_cnnindonesia.ipynb
│   ├── web_scraping_tempo.ipynb
│   ├── web_scraping_kumparan.ipynb
│   ├── web_scraping_merdeka.ipynb
│   └── web_scraping_tribunnews.ipynb
├── data/
│   ├── raw/          # output CSV tiap scraper: <sumber>_raw.csv
│   ├── processed/    # data gabungan, bersih, berlabel, hasil cluster & ringkasan
│   └── labeled/      # sampel verifikasi manual + set pasangan duplikat berlabel
├── models/           # model IndoBERT hasil fine-tuning
├── text_mining_model.ipynb
└── README.md
```

---

## Urutan Menjalankan

**Jalankan semua notebook manual di VS Code dengan environment Python Anda sendiri.**

### Tahap 1 — Scraping (folder `scraping/`)

Jalankan kedelapan notebook scraping (urutan bebas, masing-masing menulis ke
`data/raw/<sumber>_raw.csv`):

1. Buka notebook scraper.
2. **Set `TEST_MODE = True`** di sel konfigurasi → Restart & Run All → cek
   `data/raw/<sumber>_raw.csv` terbentuk, kolom benar, konten tidak kosong.
3. Kalau lolos, set **`TEST_MODE = False`** → Restart & Run All untuk run penuh
   sesuai `TARGET_ARTICLES` dan `MAX_PAGES`.

### Tahap 2 — Text Mining (`text_mining_model.ipynb`)

Butuh **GPU CUDA** (wajib; notebook berhenti dengan pesan error kalau CUDA tidak
terdeteksi — tidak ada fallback diam-diam ke CPU).

1. **Set `DEBUG_MODE = True`** → Restart & Run All → smoke test seluruh pipeline
   dengan subset kecil & 1 epoch.
2. Kalau lolos, set **`DEBUG_MODE = False`** → Restart & Run All untuk run penuh.

---

## Library yang Dibutuhkan

Tidak ada `requirements.txt` dan tidak ada sel `pip install` di notebook — Anda
install sendiri di environment Anda. Daftar library (komentar juga ada di sel import
paling atas tiap notebook):

**Scraping:** `requests`, `beautifulsoup4`, `lxml`, `pandas`, `tqdm`

**Text mining:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,
`wordcloud`, `tqdm`, `torch` (**build CUDA**), `transformers`, `sentence-transformers`,
`faiss-cpu`, `Sastrawi`, `rouge-score`, `nltk`

> **PyTorch harus versi CUDA**, bukan CPU-only. Install dari
> <https://pytorch.org/get-started/locally/> sesuai versi CUDA GPU Anda. Kalau
> `torch.cuda.is_available()` mengembalikan `False`, notebook model sengaja berhenti.

Library opsional untuk Kumparan (konten dinamis) — hanya jika endpoint JSON tidak
cukup: `selenium` + driver, atau `playwright`. Lihat markdown di
`web_scraping_kumparan.ipynb`.

---

## Konfigurasi Penting (per keputusan desain)

- **Kategori pengaduan (9 kelas):** `Sampah`, `Jalan`, `Saluran Air/Drainase`,
  `Fasos/Fasum`, `Ketertiban Umum`, `Lampu/PJU`, `Banjir`, `Transportasi/Lalu Lintas`,
  `Bukan Pengaduan`. Daftar kategori + keyword ada di **satu sel konfigurasi** di
  `text_mining_model.ipynb` agar mudah diubah.
- **Target data:** ±1500–2500 artikel total (±200–300 per situs), rentang **6 bulan
  terakhir**. Diatur lewat `TARGET_ARTICLES`, `MAX_PAGES`, dan `DATE_RANGE_MONTHS` di
  tiap notebook scraping.
- **Ringkasan extractive:** 3–5 kalimat (`SUMMARY_MIN_SENTENCES` /
  `SUMMARY_MAX_SENTENCES`).

### Catatan GPU / VRAM

Default dikalibrasi untuk **GPU laptop/PC 6 GB VRAM** (kasus paling sempit). Semua
knob ada di sel konfigurasi `text_mining_model.ipynb`:

| VRAM     | `TRAIN_BATCH_SIZE` | `GRAD_ACCUM_STEPS` | `MAX_LENGTH` | `SBERT_BATCH_SIZE` | Abstractive |
|----------|--------------------|--------------------|--------------|--------------------|-------------|
| 6 GB     | 8                  | 2                  | 256          | 32                 | opsional (mT5-small) |
| 8 GB     | 16                 | 1                  | 256          | 64                 | mT5-small / IndoBART |
| 12 GB+   | 32                 | 1                  | 384          | 128                | mT5-base    |

Metode summarization utama adalah **extractive** (tidak butuh model besar).
Abstractive **opsional** dan otomatis dilewati kalau VRAM tidak cukup.

---

## Etika Scraping

- Pakai `User-Agent` yang wajar dan `time.sleep` dengan jitter antar request.
- Cek `robots.txt` tiap situs sebelum scraping (ada sel pengecekan di tiap notebook).
- Retry dengan backoff untuk error sementara; jangan membebani server.
- Simpan checkpoint berkala agar bisa dilanjutkan kalau terputus.
- Data dipakai untuk riset akademik (klasifikasi & ringkasan pengaduan publik), bukan
  untuk reproduksi/redistribusi konten.

---

## Catatan Penting soal Selector HTML

Selector HTML tiap situs **bisa berubah sewaktu-waktu**. Kode scraper ditandai dengan
komentar `# [CEK SAAT RUNNING]` di bagian selector yang paling mungkin perlu
disesuaikan. Kalau sebuah situs mengembalikan 0 artikel atau konten kosong, buka satu
URL artikel di browser → Inspect Element → sesuaikan selector di sel yang ditandai.
