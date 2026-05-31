# Arsitektur & Requirements — Sistem Prediksi Market Hybrid

**Tema:** Prediksi XAU/USD, EUR/USD, BTC/USD menggabungkan *quant trading*, fitur teknikal SMC/ICT *swing*, BOS, CHoCH, *liquidity*, OB, IDM, FVG, dan penalaran LLM.
**Proyek:** Individu — Tugas Besar Kecerdasan Mesin.
**Bahasa:** Python.

> Catatan: dokumen ini disusun tanpa akses langsung ke isi keempat repo (koneksi internet sedang dimatikan saat penyusunan). Bagian yang bertanda **[verifikasi]** perlu kamu cek sendiri di repo aslinya, karena API/struktur bisa berubah.

---

## 0. Prinsip utama (baca ini dulu)

1. **Jangan lebur empat framework jadi satu.** Itu sumber "alur absurd" dan *dependency hell*. Tiap repo diberi **satu peran**. Sebagian cukup jadi **referensi** (dibaca kodenya), bukan *dependency*.
2. **Kamu membangun "lem" tipis sendiri.** Inti proyek adalah pipeline milikmu (feature extractor SMC + labeling + backtest). Repo eksternal dipanggil sebagai komponen, bukan dilebur.
3. **Subjektif → objektif lewat kode, bukan lewat model membaca PDF.** Aturan ICT/SMC kamu tulis jadi definisi pasti (angka). Model menguji apakah definisi itu prediktif.
4. **LLM tidak membaca chart.** Kode mendeteksi pola → angka. LLM menalar status yang sudah dihitung + konteks fundamental.
5. **MVP dulu.** Mulai dari satu aset, satu timeframe, satu baseline. Saran aset awal: **BTC/USD** (data paling lengkap & gratis, termasuk order book via exchange API).

---

## 1. Peran tiap repo

| Repo | Peran dalam sistem | Cara integrasi | Catatan |
|---|---|---|---|
| **amazon-science/chronos-forecasting** | Lapis *forecasting numerik probabilistik* (prediksi harga/return dengan ketidakpastian) | **Library** (`pip install chronos-forecasting`) — integrasi paling bersih | Model time-series pretrained berbasis arsitektur T5. **Base-nya univariate** — meramal deret harga itu sendiri, belum tentu bisa dikondisikan oleh fitur SMC-mu. Jadi perannya: jadi **satu fitur/sinyal** (forecast + interval), bukan otak utama. **[verifikasi]** dukungan covariate/exogenous |
| **TauricResearch/TradingAgents** | Lapis *penalaran LLM* (analisis fundamental/berita + sintesis keputusan + rationale) | **Adopsi arsitektur**, atau jalankan **terpisah** dan panggil via API | Framework multi-agent LLM (analis fundamental/sentimen/teknikal, peneliti bull/bear, trader, manajemen risiko). Dependensi berat (LLM/langchain). Saran: pakai sebagai **lapis penalaran (L5)** atau cukup tiru pola agennya pakai SDK LLM langsung agar ringan |
| **UFund-Me/Qbot** | **Sebaiknya dilewati** untuk kasus ini | — | TERVERIFIKASI: berorientasi pasar saham/reksadana **China**, GUI-heavy & **freemium** (banyak indikator & model AI terbaik di versi Pro berbayar). Lisensi **kontradiktif**: nav menyebut MIT, LICENSE menampilkan **CC BY-NC-SA 4.0 (non-komersial)** — **[verifikasi]**. Strategi AI-nya dibangun di atas **`qlib` (Microsoft)** → kalau butuh model zoo ML, ambil **`qlib` langsung**, lebih bersih |
| **0xemmkty/QuantMuse** | **Referensi arsitektur** + contoh pola integrasi LLM | **Referensi / cherry-pick** (lisensi MIT, aman disalin) | TERVERIFIKASI: sudah menggabungkan faktor quant + LLM (OpenAI/LangChain) + sentimen NLP + backtest + inti C++. Tapi **sangat muda** (9 commit, 1 dev); klaim "production ready" berlebihan. Bagus sebagai template, jangan diandalkan sebagai fondasi teruji |

**Ringkasnya:** hanya **Chronos** yang masuk sebagai library di environment utama. **TradingAgents** diisolasi/dipanggil terpisah. **QuantMuse** di-clone untuk dibaca polanya (MIT, aman). **Qbot dilewati** — bila butuh model zoo ML, pakai **`qlib`** langsung.

---

## 2. Kenapa jangan digabung utuh

- **Konflik dependensi.** Chronos butuh `torch`/`transformers` versi tertentu; TradingAgents butuh ekosistem LLM/langchain; Qbot punya stack-nya sendiri. Menginstal semua di satu env hampir pasti memunculkan konflik versi.
- **Lisensi.** Cek lisensi tiap repo (MIT/Apache/GPL/lainnya) sebelum menyalin kode ke proyek tugas. Cantumkan atribusi.
- **Solusi:** environment utama hanya berisi kode-mu + Chronos. Lapis LLM dijalankan di environment terpisah (atau via API). Repo referensi disimpan di folder `external/` sebagai bahan bacaan, bukan paket terinstal.

---

## 3. Arsitektur sistem (alur)

```
             ┌─────────────────────────────────────────────┐
    L0  DATA │ Ingestion multi-timeframe: 5m / 15m / 1–4H / 1D │
             └───────────────────────┬─────────────────────┘
                                     │ OHLCV (+ order book utk BTC)
             ┌───────────────────────▼─────────────────────┐
    L1  FITUR│ Feature extractor SMC/ICT  ← KAMU BANGUN      │
             │ swing • BOS • CHoCH • liquidity • OB • IDM • FVG │
             └──────────┬─────────────────────────┬──────────┘
                        │ fitur teknikal (angka)  │ deret harga
             ┌──────────▼───────────┐   ┌──────────▼──────────┐
    L2  PREDI│ Baseline numerik     │   │ Chronos (forecast    │
             │ XGBoost / LSTM       │   │ probabilistik)       │
             └──────────┬───────────┘   └──────────┬──────────┘
                        └───────────┬──────────────┘
             ┌──────────────────────▼──────────────────────┐
    L3 LABEL │ Triple-barrier labeling (TP / SL / timeout)  │
             └──────────────────────┬──────────────────────┘
             ┌──────────────────────▼──────────────────────┐
    L4 SINYAL│ Model keputusan / META-LABELING              │
             │ gabung fitur teknikal + forecast → sinyal primer │
             └──────────────────────┬──────────────────────┘
             ┌──────────────────────▼──────────────────────┐
    L5  LLM  │ Penalaran (gaya TradingAgents)               │
             │ berita/fundamental + sintesis + rationale     │
             │ "menggerbangi" sinyal (ambil / tunda / tolak) │
             └──────────────────────┬──────────────────────┘
             ┌──────────────────────▼──────────────────────┐
    L6 RISIKO│ Position sizing • stop berbasis ATR • batas DD │
             └──────────────────────┬──────────────────────┘
             ┌──────────────────────▼──────────────────────┐
    L7 UJI   │ Backtest jujur: walk-forward + biaya transaksi │
             └──────────────────────────────────────────────┘
```

### Pemetaan multi-timeframe (sesuai workflow-mu) → fitur

| Timeframe | Perannya | Jadi fitur apa |
|---|---|---|
| **1D** | Struktur market | `struktur_1d` (bullish/bearish), swing besar, BOS/CHoCH harian |
| **1–4H** | Arah market | tren H1/H4, posisi relatif terhadap zona besar |
| **15m** | Pola chart | order block, FVG, liquidity zone, equal highs/lows |
| **5m** | Eksekusi | trigger entry: CHoCH 5m **di dalam** OB 15m, **searah** struktur 1D, **setelah** liquidity sweep |

Inti gagasan: timeframe rendah "digerbangi" oleh konteks timeframe tinggi. Tiap kondisi adalah `True/False` atau angka — confluence yang tadinya diskresioner menjadi fitur terukur.

---

## 4. Requirements

### 4.1 Pengetahuan
- Python menengah (pandas, numpy, OOP dasar).
- Statistika & probabilitas, aljabar linear dasar.
- ML fundamental: train/val/test split, overfitting, cross-validation.
- Konsep finansial: log-return, volatilitas, stationarity.
- **Tiga bias wajib paham:** look-ahead bias, data leakage, survivorship bias.
- Dasar prompt engineering / RAG untuk lapis LLM (tak perlu melatih LLM dari nol).

### 4.2 Hardware
- XGBoost + feature engineering: laptop biasa cukup.
- Deep learning (LSTM/Transformer) & Chronos: GPU membantu. Tanpa GPU → **Google Colab** atau **Kaggle Notebooks** (gratis).
- LLM: **pakai API**, jangan self-host model besar.

### 4.3 Software & library

**Environment utama (`venv`/`conda`, Python 3.10+):**
- Data & komputasi: `pandas`, `numpy`
- ML: `scikit-learn`, `xgboost`, `lightgbm`
- Deep learning: `pytorch`
- Forecasting: `chronos-forecasting`
- Indikator teknikal: `pandas-ta`
- Ambil data: `ccxt` (crypto), `MetaTrader5` / `alpha_vantage` (forex)
- Backtest: `vectorbt` or `backtrader`
- Visualisasi: `matplotlib`, `plotly`

**Environment terpisah (lapis LLM):**
- TradingAgents + dependensinya, **atau** pendekatan ringan: SDK LLM + `transformers` (mis. FinBERT untuk sentimen).

> **Penting:** jangan instal TradingAgents/Qbot di environment utama karena risiko konflik versi `torch`/`transformers`/`langchain`. Pisahkan env, atau panggil lapis LLM lewat API/subprocess.

### 4.4 Data
- **BTC/USD:** `ccxt` dari exchange (mis. Binance) — OHLCV **dan** order book L2 gratis & real-time. Aset terbaik untuk MVP.
- **EUR/USD & XAU/USD:** OHLCV dari MetaTrader 5 / Dukascopy / Alpha Vantage. **Catatan struktural:** forex itu OTC, **tidak ada order book terpusat** — fitur microstructure terbatas; fokus ke OHLCV + teknikal + fundamental.
- **Fundamental/berita:** FRED (data makro AS, gratis), kalender ekonomi (ForexFactory), berita (NewsAPI tier gratis).

### 4.5 Akses LLM
- API provider (berbayar per-token) **atau** model open-source lokal jika ada GPU memadai.
- Untuk tugas kuliah, sentimen bisa pakai **FinBERT** (gratis, ringan) sebagai alternatif hemat sebelum LLM penuh.

### 4.6 Lisensi & legal
- Cek lisensi tiap repo sebelum menyalin kode; cantumkan atribusi di README.
- Ini proyek akademik/riset — **bukan nasihat finansial**. Tidak ada model yang menjamin profit.

---

## 5. Struktur folder proyek

```
market-hybrid/
├── data/
│   ├── raw/                 # OHLCV mentah per timeframe
│   └── processed/           # fitur siap pakai
├── src/
│   ├── ingestion/           # fetcher: ccxt, MT5
│   ├── features/            # ← INTI: extractor SMC (swing, bos, choch,
│   │                        #   liquidity, order_block, idm, fvg)
│   ├── forecasting/         # wrapper Chronos + baseline (xgb, lstm)
│   ├── labeling/            # triple-barrier
│   ├── model/               # meta-labeling, training, evaluasi
│   ├── reasoning/           # lapis LLM (TradingAgents-style / SDK)
│   ├── risk/                # position sizing, ATR stop
│   └── backtest/            # walk-forward harness + biaya transaksi
├── notebooks/               # eksplorasi & analisis
├── configs/                 # parameter (k swing, toleransi, dst.)
├── external/                # repo referensi (clone, read-only)
│   ├── TradingAgents/       # pola lapis penalaran LLM
│   ├── QuantMuse/           # template arsitektur quant+LLM (MIT)
│   └── qlib/                # model zoo ML (opsional, pengganti Qbot)
├── requirements.txt
│── requirements_chronos.txt
└── README.md
```

Folder `features/` adalah jantung proyek dan **harus kamu tulis sendiri** — di sinilah aturan SMC subjektif berubah jadi kode objektif.

---

## 6. Roadmap bertahap

1. **MVP** — BTC/USD, satu timeframe. Ingestion → extractor swing + BOS + CHoCH → triple-barrier → baseline XGBoost → backtest jujur (dengan biaya transaksi). Pastikan **tidak ada kebocoran data** sebelum lanjut.
2. **Tambah Chronos** sebagai fitur forecast probabilistik; bandingkan dengan/ tanpa Chronos.
3. **Lengkapi fitur SMC** — liquidity sweep, order block, FVG, lalu IDM (paling noisy). Tambah **confluence multi-timeframe** (1D→15m→5m).
4. **Lapis LLM (L5)** — mulai sederhana: sentimen berita + rationale teks. Baru kemudian arsitektur multi-agent gaya TradingAgents.
5. **Meta-labeling** — gabungkan fitur teknikal + forecast Chronos jadi sinyal primer, lalu model sekunder memutuskan ambil/ukuran.
6. **Perluas aset** — XAU/USD lalu EUR/USD (dengan catatan keterbatasan order book).
7. **Evaluasi & laporan** — Sharpe ratio, max drawdown, profit factor, expectancy, feature importance (SHAP). Bukan sekadar akurasi/winrate.

---

## 7. Risiko, batasan, & ekspektasi jujur

- **Pasar mendekati efisien.** Akurasi arah konsisten di atas 52–55% pada pair likuid sudah bagus dan bermakna — bukan 90%.
- **Winrate bukan tujuan.** Yang menentukan profitabilitas adalah *expectancy* (winrate × reward) dan *risk-adjusted return*.
- **IDM & pemilihan order block paling subjektif.** Implementasikan, parameterkan, lalu biarkan data yang memvonis apakah prediktif.
- **Chronos univariate.** Jangan berharap ia "membaca" fitur SMC-mu; ia satu sinyal di antara banyak. **[verifikasi]** kemampuan covariate versi terbaru.
- **Repo bisa berubah.** Verifikasi API, lisensi, dan dependensi sebelum mengandalkannya.
- **Penilaian tugas** ada pada **rigor metodologi** (labeling benar, backtest jujur, anti-kebocoran), bukan klaim cuan.

---

## 8. Keputusan yang perlu kamu tentukan

Beberapa percabangan yang akan mengubah detail dokumen ini — jawabanmu akan saya pakai untuk merevisi:

1. **Aset fokus MVP:** BTC dulu (saran saya) atau langsung ke forex (XAU/EUR)?
2. **Durasi & deadline proyek:** berapa minggu/bulan tersedia? (menentukan seberapa jauh roadmap dikejar)
3. **Ases LLM:** ada budget API, atau harus gratis/lokal? (menentukan apakah TradingAgents penuh realistis atau cukup FinBERT + SDK ringan)
4. **Cakupan:** cukup backtest historis untuk laporan, atau perlu simulasi *paper trading* real-time?
