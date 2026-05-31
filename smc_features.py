"""
smc_features.py
Extractor fitur SMC/ICT & Harmonic Patterns yang OBJEKTIF dan LEAK-FREE dari data OHLCV.

Prinsip penting (anti look-ahead bias):
  Sebuah swing di index i baru BISA dipastikan setelah k bar berikutnya (i+k).
  Jadi level swing hanya boleh dipakai mulai bar (i+k), bukan di bar i.
  Seluruh fitur di sini menghormati aturan itu.
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


def detect_harmonics_vectorized(df: pd.DataFrame, k: int = 2, tol: float = 0.08) -> pd.DataFrame:
    """
    Mendeteksi 5 Pola Harmonik Utama (Gartley, Butterfly, Bat, Crab, Cypher)
    secara leak-free berdasarkan 4 titik swing historis terkonfirmasi (X, A, B, C)
    dan harga close saat ini (D).
    """
    sh, sl = detect_swings(df, k)
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    close = df['close'].to_numpy()
    n = len(df)
    
    bull_gartley = np.zeros(n, dtype=int)
    bear_gartley = np.zeros(n, dtype=int)
    bull_butterfly = np.zeros(n, dtype=int)
    bear_butterfly = np.zeros(n, dtype=int)
    bull_bat = np.zeros(n, dtype=int)
    bear_bat = np.zeros(n, dtype=int)
    bull_crab = np.zeros(n, dtype=int)
    bear_crab = np.zeros(n, dtype=int)
    bull_cypher = np.zeros(n, dtype=int)
    bear_cypher = np.zeros(n, dtype=int)
    
    active_swings = []
    
    for i in range(n):
        # Konfirmasi swing terjadi pada j = i - k
        j = i - k
        if j >= 0:
            if sh[j]:
                active_swings.append((j, highs[j], 1))
            if sl[j]:
                active_swings.append((j, lows[j], -1))
        
        if len(active_swings) < 4:
            continue
            
        # Dapatkan 4 swing terkonfirmasi terakhir
        s0, s1, s2, s3 = active_swings[-4:]
        
        # Harus berselang-seling (High, Low, High, Low / Low, High, Low, High)
        if s0[2] == s1[2] or s1[2] == s2[2] or s2[2] == s3[2]:
            continue
            
        X, A, B, C = s0[1], s1[1], s2[1], s3[1]
        D = close[i]
        
        XA = abs(A - X)
        AB = abs(B - A)
        BC = abs(C - B)
        XC = abs(C - X)
        
        if XA == 0 or AB == 0 or BC == 0 or XC == 0:
            continue
            
        ratio_B = AB / XA
        ratio_C = BC / AB
        ratio_D_XA = abs(D - X) / XA
        ratio_D_BC = abs(D - C) / BC
        
        is_bullish_sequence = (s0[2] == -1 and s1[2] == 1 and s2[2] == -1 and s3[2] == 1)
        is_bearish_sequence = (s0[2] == 1 and s1[2] == -1 and s2[2] == 1 and s3[2] == -1)
        
        if is_bullish_sequence:
            # 1. Gartley
            if (abs(ratio_B - 0.618) <= tol and 
                0.382 - tol <= ratio_C <= 0.786 + tol and
                abs(ratio_D_XA - 0.786) <= tol and
                1.27 - tol <= ratio_D_BC <= 1.618 + tol and
                D < C):
                bull_gartley[i] = 1
                
            # 2. Butterfly
            if (abs(ratio_B - 0.786) <= tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                1.27 - tol <= ratio_D_XA <= 1.618 + tol and
                1.618 - tol <= ratio_D_BC <= 2.618 + tol and
                D < X):
                bull_butterfly[i] = 1
                
            # 3. Bat
            if (0.382 - tol <= ratio_B <= 0.50 + tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                abs(ratio_D_XA - 0.886) <= tol and
                1.618 - tol <= ratio_D_BC <= 2.618 + tol and
                D < C):
                bull_bat[i] = 1
                
            # 4. Crab
            if (0.382 - tol <= ratio_B <= 0.618 + tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                abs(ratio_D_XA - 1.618) <= tol and
                2.24 - tol <= ratio_D_BC <= 3.618 + tol and
                D < X):
                bull_crab[i] = 1
                
            # 5. Cypher
            ratio_C_XA = abs(C - X) / XA
            ratio_D_XC = abs(D - X) / XC
            if (0.382 - tol <= ratio_B <= 0.618 + tol and 
                1.272 - tol <= ratio_C_XA <= 1.414 + tol and
                abs(ratio_D_XC - 0.786) <= tol and
                D > X):
                bull_cypher[i] = 1
                
        elif is_bearish_sequence:
            # 1. Gartley
            if (abs(ratio_B - 0.618) <= tol and 
                0.382 - tol <= ratio_C <= 0.786 + tol and
                abs(ratio_D_XA - 0.786) <= tol and
                1.27 - tol <= ratio_D_BC <= 1.618 + tol and
                D > C):
                bear_gartley[i] = 1
                
            # 2. Butterfly
            if (abs(ratio_B - 0.786) <= tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                1.27 - tol <= ratio_D_XA <= 1.618 + tol and
                1.618 - tol <= ratio_D_BC <= 2.618 + tol and
                D > X):
                bear_butterfly[i] = 1
                
            # 3. Bat
            if (0.382 - tol <= ratio_B <= 0.50 + tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                abs(ratio_D_XA - 0.886) <= tol and
                1.618 - tol <= ratio_D_BC <= 2.618 + tol and
                D > C):
                bear_bat[i] = 1
                
            # 4. Crab
            if (0.382 - tol <= ratio_B <= 0.618 + tol and 
                0.382 - tol <= ratio_C <= 0.886 + tol and
                abs(ratio_D_XA - 1.618) <= tol and
                2.24 - tol <= ratio_D_BC <= 3.618 + tol and
                D > X):
                bear_crab[i] = 1
                
            # 5. Cypher
            ratio_C_XA = abs(C - X) / XA
            ratio_D_XC = abs(D - X) / XC
            if (0.382 - tol <= ratio_B <= 0.618 + tol and 
                1.272 - tol <= ratio_C_XA <= 1.414 + tol and
                abs(ratio_D_XC - 0.786) <= tol and
                D < X):
                bear_cypher[i] = 1
                
    out = pd.DataFrame(index=df.index)
    out['bull_gartley'] = bull_gartley
    out['bear_gartley'] = bear_gartley
    out['bull_butterfly'] = bull_butterfly
    out['bear_butterfly'] = bear_butterfly
    out['bull_bat'] = bull_bat
    out['bear_bat'] = bear_bat
    out['bull_crab'] = bull_crab
    out['bear_crab'] = bear_crab
    out['bull_cypher'] = bull_cypher
    out['bear_cypher'] = bear_cypher
    out['harmonic_bullish_signal'] = (bull_gartley | bull_butterfly | bull_bat | bull_crab | bull_cypher)
    out['harmonic_bearish_signal'] = (bear_gartley | bear_butterfly | bear_bat | bear_crab | bear_cypher)
    return out


def build_features(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """
    Hitung fitur SMC/ICT & Harmonic Patterns dari OHLCV. Semua level swing dipakai mulai
    bar konfirmasi (swing_index + k) → tidak ada kebocoran masa depan.
    """
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
    
    # Simpan indeks swing terakhir
    last_sh_idx = -1
    last_sl_idx = -1

    # Inisialisasi pelacakan Order Block (OB) unmitigated
    # Format OB: {'top': float, 'bottom': float, 'type': 'bullish'/'bearish', 'has_fvg': bool, 'mitigated': bool}
    active_obs = []

    # Fair Value Gap (imbalance 3-candle) — pre-calculate agar leak-free
    bull_fvg_series = (df["low"] > df["high"].shift(2)).fillna(False).to_numpy()
    bear_fvg_series = (df["high"] < df["low"].shift(2)).fillna(False).to_numpy()

    dist_to_unmitigated_bullish_ob = np.full(n, np.nan)
    dist_to_unmitigated_bearish_ob = np.full(n, np.nan)
    ob_fvg_confluence = np.zeros(n, dtype=int)

    for i in range(n):
        j = i - k  # swing di j baru terkonfirmasi saat bar i
        if j >= 0:
            if sh[j]:
                brk_sh = high[j]
                last_sh = high[j]
                last_sh_idx = j
            if sl[j]:
                brk_sl = low[j]
                last_sl = low[j]
                last_sl_idx = j

        # Liquidity sweep: wick menembus level, tapi CLOSE balik ke dalam (rejection)
        if not np.isnan(last_sl) and low[i] < last_sl and close[i] > last_sl:
            sweep_up[i] = True
        if not np.isnan(last_sh) and high[i] > last_sh and close[i] < last_sh:
            sweep_down[i] = True

        # Structure break: CLOSE menembus level terkonfirmasi
        created_ob = None
        if not np.isnan(brk_sh) and close[i] > brk_sh:
            if trend <= 0:
                choch_up[i] = True       # pergantian karakter (reversal)
            else:
                bos_up[i] = True         # kelanjutan tren
            trend = 1
            if not np.isnan(brk_sl): last_sl = brk_sl
            brk_sh = np.nan
            
            # Form Valid Bullish OB di swing low terakhir sebelum breakout
            if last_sl_idx != -1:
                # Cek jika ada Bullish FVG antara j dan i (mitigation area)
                has_fvg = bool(np.any(bull_fvg_series[last_sl_idx:i+1]))
                created_ob = {
                    'top': high[last_sl_idx],
                    'bottom': low[last_sl_idx],
                    'type': 'bullish',
                    'has_fvg': has_fvg,
                    'mitigated': False
                }
                
        if not np.isnan(brk_sl) and close[i] < brk_sl:
            if trend >= 0:
                choch_down[i] = True
            else:
                bos_down[i] = True
            trend = -1
            if not np.isnan(brk_sh): last_sh = brk_sh
            brk_sl = np.nan
            
            # Form Valid Bearish OB di swing high terakhir sebelum breakout
            if last_sh_idx != -1:
                # Cek jika ada Bearish FVG antara j dan i
                has_fvg = bool(np.any(bear_fvg_series[last_sh_idx:i+1]))
                created_ob = {
                    'top': high[last_sh_idx],
                    'bottom': low[last_sh_idx],
                    'type': 'bearish',
                    'has_fvg': has_fvg,
                    'mitigated': False
                }

        if created_ob is not None:
            active_obs.append(created_ob)

        # Update status mitigasi & invalidasi OB aktif secara realtime (leak-free)
        remaining_obs = []
        for zone in active_obs:
            if zone['type'] == 'bullish':
                # Mitigated jika terpantul (low di bawah/sama dengan batas atas OB)
                if low[i] <= zone['top']:
                    zone['mitigated'] = True
                # Invalidated/Broken jika close menembus batas bawah OB
                if close[i] < zone['bottom']:
                    continue # Discard broken OB
            else: # bearish
                # Mitigated jika terpantul (high di atas/sama dengan batas bawah OB)
                if high[i] >= zone['bottom']:
                    zone['mitigated'] = True
                # Invalidated/Broken jika close menembus batas atas OB
                if close[i] > zone['top']:
                    continue # Discard broken OB
            remaining_obs.append(zone)
        active_obs = remaining_obs

        # Hitung jarak ke OB unmitigated (atau termitigasi tapi belum broken) terdekat
        closest_bull_ob = None
        closest_bear_ob = None
        
        for zone in active_obs:
            if zone['type'] == 'bullish' and close[i] >= zone['top']:
                if closest_bull_ob is None or zone['top'] > closest_bull_ob['top']:
                    closest_bull_ob = zone
            elif zone['type'] == 'bearish' and close[i] <= zone['bottom']:
                if closest_bear_ob is None or zone['bottom'] < closest_bear_ob['bottom']:
                    closest_bear_ob = zone

        if closest_bull_ob is not None:
            dist_to_unmitigated_bullish_ob[i] = (close[i] - closest_bull_ob['top']) / close[i]
            # Konfluensi FVG: dekat dengan OB berkualitas tinggi
            if closest_bull_ob['has_fvg'] and dist_to_unmitigated_bullish_ob[i] <= 0.01:
                ob_fvg_confluence[i] = 1
                
        if closest_bear_ob is not None:
            dist_to_unmitigated_bearish_ob[i] = (closest_bear_ob['bottom'] - close[i]) / close[i]
            # Konfluensi FVG: dekat dengan OB berkualitas tinggi
            if closest_bear_ob['has_fvg'] and dist_to_unmitigated_bearish_ob[i] <= 0.01:
                ob_fvg_confluence[i] = -1

        structure[i] = trend
        run_sh[i] = last_sh
        run_sl[i] = last_sl

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
    out["bull_fvg"] = bull_fvg_series
    out["bear_fvg"] = bear_fvg_series
    out["dist_to_swing_high"] = dist_to_swing_high
    out["dist_to_swing_low"] = dist_to_swing_low
    out["upper_wick_ratio"] = upper_wick_ratio
    out["lower_wick_ratio"] = lower_wick_ratio
    
    # Fitur Baru: OB Memory & Confluence
    out["dist_to_unmitigated_bullish_ob"] = dist_to_unmitigated_bullish_ob
    out["dist_to_unmitigated_bearish_ob"] = dist_to_unmitigated_bearish_ob
    out["ob_fvg_confluence"] = ob_fvg_confluence
    
    # Isi NaN pada jarak OB dengan nilai default aman
    out["dist_to_unmitigated_bullish_ob"] = out["dist_to_unmitigated_bullish_ob"].fillna(1.0)
    out["dist_to_unmitigated_bearish_ob"] = out["dist_to_unmitigated_bearish_ob"].fillna(1.0)

    # Deteksi & Gabungkan Fitur Harmonik
    harmonics = detect_harmonics_vectorized(df, k)
    out = out.join(harmonics)

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
    """Ambil OHLCV dari exchange via ccxt (jalankan LOKAL, butuh internet)."""
    import ccxt
    ex = getattr(ccxt, exchange_id)()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.set_index("ts")


def _synthetic_ohlcv(n=600, seed=0, symbol="BTC/USDT"):
    """Buat OHLCV sintetis (random walk) dengan parameter realistis per instrumen."""
    rng = np.random.default_rng(seed)
    
    # Atur volatilitas, harga awal, dan wick khas tiap instrumen
    if "EUR" in symbol:
        vol = 0.0005
        init_price = 1.09
        wick_scale = 0.0004
    elif "XAU" in symbol or "GOLD" in symbol:
        vol = 0.0015
        init_price = 2000.0
        wick_scale = 0.0012
    else: # Default Kripto (BTC)
        vol = 0.003
        init_price = 30000.0
        wick_scale = 0.0025
        
    ret = rng.normal(0, vol, n)
    close = init_price * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, wick_scale, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=idx)



if __name__ == "__main__":
    df = _synthetic_ohlcv(800)
    feats = build_features(df, k=2)
    print("Jumlah bar          :", len(df))
    print("Swing Highs / Lows  :", int(detect_swings(df, 2)[0].sum()), "/", int(detect_swings(df, 2)[1].sum()))
    print("BOS up / down        :", int(feats.bos_up.sum()), "/", int(feats.bos_down.sum()))
    print("OB unmitigated bull / bear average dist:")
    print(f"- Bullish OB: {feats['dist_to_unmitigated_bullish_ob'].mean():.4f}")
    print(f"- Bearish OB: {feats['dist_to_unmitigated_bearish_ob'].mean():.4f}")
    print("OB FVG Confluences  :", int((feats.ob_fvg_confluence != 0).sum()))
    print("Cypher patterns bull / bear:", int(feats.bull_cypher.sum()), "/", int(feats.bear_cypher.sum()))
