# Alur & Mekanisme Kerja Program — Sistem Prediksi Market Hybrid (SMC + Chronos + LLM)

Dokumen ini merinci secara teknis **Alur Program (Workflow)** dan **Mekanisme Modular (Component Mechanism)** dari program trading kuantitatif hibrida yang telah diimplementasikan dalam folder `src/`.

---

## 1. Alur Kerja Program (Program Workflow)

Alur pemrosesan data, kalkulasi fitur, training model, hingga pengambilan keputusan transaksi akhir berjalan secara sekuensial (bar-demi-bar) sebagai berikut:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                           │
│    - Menarik data OHLCV 15m (eksekusi) & 1D (tren utama)    │
│    - CCXT API (Binance) -> Jaringan Gagal -> Fallback Sintetis│
└──────────────────────────────┬──────────────────────────────┘
                               │ pandas.DataFrame
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SMC/ICT FEATURE EXTRACTION                               │
│    - Cari swing high/low terkonfirmasi (pointer j = i - k)   │
│    - Deteksi BOS/CHoCH breakout dan FVG imbalance 3-candle  │
│    - Petakan Zona Premium (Jual) & Discount (Beli)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ SMC Features
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PROBABILISTIC CHRONOS FORECASTING                        │
│    - Masukkan context window Close historis                 │
│    - Jalankan model pretrained amazon/chronos-t5-tiny        │
│    - Ekstrak Kuantil 10% (Bawah), 50% (Median), 90% (Atas)  │
│    - Bentuk fitur trend, volatility, skewness, & OB breach  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Exogenous Chronos Features
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. SAFE MULTI-TIMEFRAME ALIGNMENT                           │
│    - Geser data Daily harian kemarin (Shift 1 bar daily)    │
│    - Satukan data daily kemarin ke data 15m hari ini        │
│    - Gunakan pd.merge_asof(direction="backward")            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Unified Leak-Free Dataset
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. TRIPLE-BARRIER LABELING                                  │
│    - Hitung volatilitas dinamis (ATR 14 periode)            │
│    - Tentukan batasan TP (2 * ATR) & SL (1.5 * ATR)         │
│    - Evaluasi masa depan sampai 40 bar (Timeout)            │
│    - Beri label: 1 (BUY), -1 (SELL), 0 (HOLD)               │
└──────────────────────────────┬──────────────────────────────┘
                               │ Target Labels [1, 0, -1]
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. DUAL-STAGE MACHINE LEARNING MODEL                        │
│    - Model ML Primer: Prediksi arah harga (BUY/SELL/HOLD)    │
│    - Model ML Sekunder (Meta-ML): Prediksi peluang sukses   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Sinyal + Probabilitas Sukses > 40%
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. LLM REASONING GATEKEEPER                                 │
│    - Kirim data pasar terkuantisasi & berita makro          │
│    - Evaluasi kognitif Gemini 1.5 Flash -> Keputusan Akhir   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Mekanisme Komponen Program (Module Mechanism)

Sistem ini modular dan dibagi menjadi 5 komponen utama yang bekerja secara terintegrasi:

### A. Ingestion & Preprocessing (`smc_features.py` & `src/pipeline.py`)
*   **OHLCV Ingestion:** Menggunakan CCXT API untuk mengambil data real-time. Jika jaringan internet offline, sistem secara otomatis mengaktifkan **Fallback Mode** dengan membangkitkan data sintetis random walk yang realistis (`_synthetic_ohlcv`) sehingga program tetap dapat diuji.
*   **Anti Look-Ahead Bias Swing:** Swing High dan Swing Low lokal sejauh $k$ bar ke kiri-kanan dikonfirmasi secara objektif pada bar `i` hanya dengan membaca indeks `i-k`.
*   **Premium / Discount Zone:** 
    *   *Equilibrium (50%):* $\text{Equilibrium} = (\text{Swing High terkonfirmasi terakhir} + \text{Swing Low terkonfirmasi terakhir}) / 2$.
    *   Jika harga saat ini (`close`) di bawah equilibrium, harga berada di **Discount Zone** (`in_discount = 1`). Sangat ideal untuk entry **BUY** karena harga dinilai "murah" dalam rentang swing terakhir.
    *   Jika harga saat ini (`close`) di atas equilibrium, harga berada di **Premium Zone** (`in_premium = 1`). Sangat ideal untuk entry **SELL** karena harga dinilai "mahal".

### B. Amazon Chronos Wrapper (`src/forecasting/chronos_wrapper.py`)
*   **Probabilistic Forecasting:** Memanfaatkan `amazon/chronos-t5-tiny` untuk menghasilkan ramalan probabilistik ke depan sebanyak 12 bar.
*   **5 Exogenous Features:**
    1.  `chronos_trend`: Log-return dari harga Close sekarang ke median prediksi ($t+12$).
    2.  `chronos_volatility`: Standar deviasi dari seluruh prediksi median.
    3.  `chronos_skewness`: Mengukur asimetri antara kuantil 90% (atas) dan 10% (bawah).
    4.  `chronos_breach_sh`: `1` jika proyeksi atas Chronos menembus level Swing High SMC terakhir.
    5.  `chronos_breach_sl`: `1` jika proyeksi bawah Chronos menembus level Swing Low SMC terakhir.

### C. Multi-Timeframe Alignment (`src/features/alignment.py`)
*   Mencegah kebocoran data (*data leakage*) harian. Data daily (1D) mengalami penggeseran (`shift(1)`) sehingga data daily hari ini baru dapat digabungkan dengan data 15m hari esok setelah candle harian daily resmi ditutup. Proses merger menggunakan `pd.merge_asof(direction="backward")`.

### D. Volatility-Adjusted Triple-Barrier Labeling (`src/labeling/triple_barrier.py`)
*   Menggunakan Average True Range (ATR) 14 periode untuk menentukan lebar TP dan SL secara dinamis sesuai volatilitas pasar saat ini. Batas waktu maksimal memegang posisi ditetapkan `timeout_bars = 40`.

### E. Dual XGBoost & LLM reasoning (`src/pipeline.py` & `src/reasoning/llm_gate.py`)
*   **Base Model (XGBoost):** Memprediksi arah transaksi (BUY/SELL/HOLD).
*   **Meta-Labeling Model (XGBoost Sekunder):** Dilatih khusus memprediksi probabilitas keberhasilan transaksi aktif (`1` jika TP tersentuh, `0` jika SL atau Timeout).
*   **LLM Gatekeeper:** Jika probabilitas sukses Meta-ML $> 40\%$, detail metrik teknis dikirim ke LLM Gemini (`gemini-1.5-flash`) dengan prompt terstruktur bersama berita makro (FRED). LLM melakukan penalaran kognitif tingkat tinggi untuk menghasilkan keputusan final: `CONFIRM`, `DELAY`, atau `REJECT`.
