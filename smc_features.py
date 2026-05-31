"""
smc_features.py
Extractor fitur SMC/ICT yang OBJEKTIF dan LEAK-FREE dari data OHLCV.

Prinsip penting (anti look-ahead bias):
  Sebuah swing di index i baru BISA dipastikan setelah k bar berikutnya (i+k).
  Jadi level swing hanya boleh dipakai mulai bar (i+k), bukan di bar i.
  Seluruh fitur di sini menghormati aturan itu.

Kolom input wajib: open, high, low, close  (volume opsional)
"""

import numpy as np
import pandas as pd


def detect_swings(df: pd.DataFrame, k: int = 2):
    """Swing high/low via fractal: ekstrem lokal ketat sejauh k bar kiri-kanan."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        if high[i] > high[i - k:i].max() and high[i] > high[i + 1:i + k + 1].max():
            sh[i] = True
        if low[i] < low[i - k:i].min() and low[i] < low[i + 1:i + k + 1].min():
            sl[i] = True
    return sh, sl


def build_features(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Hitung fitur SMC/ICT dari OHLCV. Semua level swing dipakai mulai
    bar konfirmasi (swing_index + k) → tidak ada kebocoran masa depan."""
    df = df.copy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    sh, sl = detect_swings(df, k)

    structure = np.zeros(n, dtype=int)          # 1 bullish, -1 bearish, 0 netral
    bos_up = np.zeros(n, dtype=bool)
    bos_down = np.zeros(n, dtype=bool)
    choch_up = np.zeros(n, dtype=bool)
    choch_down = np.zeros(n, dtype=bool)
    sweep_up = np.zeros(n, dtype=bool)          # sell-side liquidity grab (bullish)
    sweep_down = np.zeros(n, dtype=bool)        # buy-side liquidity grab (bearish)
    run_sh = np.full(n, np.nan)                 # level swing high terkonfirmasi terakhir
    run_sl = np.full(n, np.nan)

    trend = 0
    brk_sh = np.nan   # level untuk deteksi break (di-null setelah ditembus)
    brk_sl = np.nan
    last_sh = np.nan  # level persisten untuk jarak & sweep
    last_sl = np.nan

    for i in range(n):
        j = i - k  # swing di j baru terkonfirmasi saat bar i
        if j >= 0:
            if sh[j]:
                brk_sh = high[j]
                last_sh = high[j]
            if sl[j]:
                brk_sl = low[j]
                last_sl = low[j]

        # Liquidity sweep: wick menembus level, tapi CLOSE balik ke dalam (rejection)
        if not np.isnan(last_sl) and low[i] < last_sl and close[i] > last_sl:
            sweep_up[i] = True
        if not np.isnan(last_sh) and high[i] > last_sh and close[i] < last_sh:
            sweep_down[i] = True

        # Structure break: CLOSE menembus level terkonfirmasi
        if not np.isnan(brk_sh) and close[i] > brk_sh:
            if trend <= 0:
                choch_up[i] = True       # pergantian karakter (reversal)
            else:
                bos_up[i] = True         # kelanjutan tren
            trend = 1
            brk_sh = np.nan
        if not np.isnan(brk_sl) and close[i] < brk_sl:
            if trend >= 0:
                choch_down[i] = True
            else:
                bos_down[i] = True
            trend = -1
            brk_sl = np.nan

        structure[i] = trend
        run_sh[i] = last_sh
        run_sl[i] = last_sl

    # Fair Value Gap (imbalance 3-candle) — vektor, leak-free (pakai i dan i-2)
    bull_fvg = (df["low"] > df["high"].shift(2)).fillna(False).to_numpy()
    bear_fvg = (df["high"] < df["low"].shift(2)).fillna(False).to_numpy()

    # Jarak ke level swing terkonfirmasi terakhir (dinormalisasi harga)
    dist_to_swing_high = (run_sh - close) / close   # >0: resistance di atas
    dist_to_swing_low = (close - run_sl) / close     # >0: support di bawah

    # Rasio wick (shadow) — bukti bahwa info sumbu candle ada di data
    o = df["open"].to_numpy()
    body_top = np.maximum(o, close)
    body_bottom = np.minimum(o, close)
    rng = np.maximum(high - low, 1e-12)
    upper_wick_ratio = (high - body_top) / rng
    lower_wick_ratio = (body_bottom - low) / rng

    out = pd.DataFrame(index=df.index)
    out["structure"] = structure
    out["bos_up"] = bos_up
    out["bos_down"] = bos_down
    out["choch_up"] = choch_up
    out["choch_down"] = choch_down
    out["sweep_up"] = sweep_up
    out["sweep_down"] = sweep_down
    out["bull_fvg"] = bull_fvg
    out["bear_fvg"] = bear_fvg
    out["dist_to_swing_high"] = dist_to_swing_high
    out["dist_to_swing_low"] = dist_to_swing_low
    out["upper_wick_ratio"] = upper_wick_ratio
    out["lower_wick_ratio"] = lower_wick_ratio
    return out


def calculate_premium_discount(df_smc: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung Premium/Discount Zone berdasarkan konsep SMC/ICT yang objektif.
    - Equilibrium (50%): Titik tengah dari Swing High dan Swing Low terkonfirmasi terakhir.
    - Discount Zone (< 0.5): Harga saat ini berada di separuh bawah rentang swing. Cocok untuk BUY.
    - Premium Zone (> 0.5): Harga saat ini berada di separuh atas rentang swing. Cocok untuk SELL.
    """
    df = df_smc.copy()
    close = df_raw['close'].to_numpy()
    
    dist_sh = df['dist_to_swing_high'].to_numpy()
    dist_sl = df['dist_to_swing_low'].to_numpy()
    
    run_sh = close * (1.0 + dist_sh)
    run_sl = close * (1.0 - dist_sl)
    
    swing_range = run_sh - run_sl
    swing_range = np.where(swing_range <= 0, 1e-12, swing_range)
    
    premium_discount_pct = (close - run_sl) / swing_range
    premium_discount_pct = np.clip(premium_discount_pct, 0.0, 1.0)
    
    df['premium_discount_pct'] = premium_discount_pct
    df['in_discount'] = (premium_discount_pct < 0.5).astype(int)
    df['in_premium'] = (premium_discount_pct > 0.5).astype(int)
    df['in_equilibrium'] = (premium_discount_pct == 0.5).astype(int)
    
    return df


def load_ohlcv_ccxt(symbol="BTC/USDT", timeframe="15m", limit=1500, exchange_id="binance"):
    """Ambil OHLCV dari exchange via ccxt (jalankan LOKAL, butuh internet).
    Return DataFrame berindeks waktu dengan kolom open/high/low/close/volume."""
    import ccxt  # pip install ccxt
    ex = getattr(ccxt, exchange_id)()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def _synthetic_ohlcv(n=600, seed=0):
    """Buat OHLCV sintetis (random walk) untuk menguji logika tanpa internet."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.003, n)
    close = 30000 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, 0.0025, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=idx)


if __name__ == "__main__":
    df = _synthetic_ohlcv()
    feats = build_features(df, k=2)
    print("Jumlah bar          :", len(df))
    print("Swing high/low       :",
          int(detect_swings(df, 2)[0].sum()), "/", int(detect_swings(df, 2)[1].sum()))
    print("BOS up / down        :", int(feats.bos_up.sum()), "/", int(feats.bos_down.sum()))
    print("CHoCH up / down      :", int(feats.choch_up.sum()), "/", int(feats.choch_down.sum()))
    print("Sweep up / down      :", int(feats.sweep_up.sum()), "/", int(feats.sweep_down.sum()))
    print("FVG bull / bear      :", int(feats.bull_fvg.sum()), "/", int(feats.bear_fvg.sum()))
    print("\nContoh 8 baris fitur:")
    print(feats.tail(8).to_string())
