"""
alignment.py
Modul untuk menyelaraskan data multi-timeframe (15m & 1D) secara LEAK-FREE
dan memperkaya dataset dengan fitur SMC objektif (Premium/Discount Zone) serta 
fitur probabilistik dari Amazon Chronos.
"""

import numpy as np
import pandas as pd
from smc_features import build_features
from forecasting.chronos_wrapper import ChronosForecaster

def calculate_premium_discount(df_smc: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung Premium/Discount Zone berdasarkan konsep SMC/ICT yang objektif.
    - Equilibrium (50%): Titik tengah dari Swing High dan Swing Low terkonfirmasi terakhir.
    - Discount Zone (< 0.5): Harga saat ini berada di separuh bawah rentang swing. Cocok untuk BUY.
    - Premium Zone (> 0.5): Harga saat ini berada di separuh atas rentang swing. Cocok untuk SELL.
    """
    df = df_smc.copy()
    close = df_raw['close'].to_numpy()
    
    # Ambil level swing terkonfirmasi yang sedang berjalan (tidak bocor)
    # dist_to_swing_high = (run_sh - close) / close  =>  run_sh = close * (1 + dist_to_swing_high)
    dist_sh = df['dist_to_swing_high'].to_numpy()
    dist_sl = df['dist_to_swing_low'].to_numpy()
    
    run_sh = close * (1.0 + dist_sh)
    run_sl = close * (1.0 - dist_sl)
    
    # Hitung nilai persentase posisi harga dalam range swing (Premium / Discount)
    # 0.0 = Tepat di Swing Low, 1.0 = Tepat di Swing High, 0.5 = Equilibrium
    swing_range = run_sh - run_sl
    # Cegah pembagian dengan nol
    swing_range = np.where(swing_range <= 0, 1e-12, swing_range)
    
    premium_discount_pct = (close - run_sl) / swing_range
    # Klip nilai agar tetap di rentang [0, 1] jika ada ekstrusi sementara
    premium_discount_pct = np.clip(premium_discount_pct, 0.0, 1.0)
    
    df['premium_discount_pct'] = premium_discount_pct
    df['in_discount'] = (premium_discount_pct < 0.5).astype(int)
    df['in_premium'] = (premium_discount_pct > 0.5).astype(int)
    df['in_equilibrium'] = (premium_discount_pct == 0.5).astype(int)
    
    return df

def build_hybrid_dataset(
    df_low_tf: pd.DataFrame, 
    df_high_tf: pd.DataFrame, 
    k_low: int = 2, 
    k_high: int = 2,
    run_chronos: bool = True,
    chronos_model: str = "amazon/chronos-t5-tiny"
) -> pd.DataFrame:
    """
    Menyusun dataset hibrida (SMC + Chronos + Multi-Timeframe Alignment) secara 100% leak-free.
    """
    print("\n--- PROSES PENYUSUNAN DATASET HIBRIDA ---")
    
    # 1. Ekstrak fitur SMC untuk Low Timeframe (15m)
    print("[1] Mengekstrak fitur SMC untuk Low Timeframe (15m)...")
    feats_low = build_features(df_low_tf, k=k_low)
    feats_low = df_low_tf[['open', 'high', 'low', 'close']].join(feats_low)
    
    # Hitung Premium/Discount untuk Low Timeframe
    feats_low = calculate_premium_discount(feats_low, df_low_tf)
    
    # 2. Ekstrak fitur SMC untuk High Timeframe (1D)
    print("[2] Mengekstrak fitur SMC untuk High Timeframe (1D)...")
    feats_high = build_features(df_high_tf, k=k_high)
    feats_high = df_high_tf[['close']].join(feats_high)
    
    # Hitung Premium/Discount untuk High Timeframe
    feats_high = calculate_premium_discount(feats_high, df_high_tf)
    
    # Tambahkan prefix 'htf_' agar unik
    feats_high = feats_high.add_prefix("htf_")
    
    # --- PREVENT LOOK-AHEAD BIAS: Shift High Timeframe 1 bar ---
    print("[3] Melakukan pergeseran (shift 1 bar) pada High Timeframe...")
    feats_high_shifted = feats_high.shift(1)
    
    # 3. Integrasikan Amazon Chronos pada Low Timeframe (15m)
    if run_chronos:
        print("[4] Mengaktifkan Amazon Chronos Pipeline...")
        forecaster = ChronosForecaster(model_name=chronos_model)
        chronos_feats = forecaster.extract_features(df_low_tf, prediction_length=12, context_length=100)
        feats_low = feats_low.join(chronos_feats)
        
        # Integrasikan ramalan Chronos dengan level OB/Swing SMC (Kombinasi Maksimal)
        # Cek apakah kuantil 90% Chronos menembus Swing High terkonfirmasi terakhir
        close_low = df_low_tf['close'].to_numpy()
        run_sh_low = close_low * (1.0 + feats_low['dist_to_swing_high'].to_numpy())
        run_sl_low = close_low * (1.0 - feats_low['dist_to_swing_low'].to_numpy())
        
        # Ambil median forecast dan std deviasi
        trend_est = feats_low['chronos_trend'].to_numpy()
        vol_est = feats_low['chronos_volatility'].to_numpy()
        
        # Proyeksikan batas kuantil 90% (atas) dan 10% (bawah) ke depan secara aproksimasi
        approx_q90 = close_low * np.exp(trend_est + 1.28 * vol_est)
        approx_q10 = close_low * np.exp(trend_est - 1.28 * vol_est)
        
        feats_low['chronos_breach_sh'] = (approx_q90 > run_sh_low).astype(int)
        feats_low['chronos_breach_sl'] = (approx_q10 < run_sl_low).astype(int)
        print("[4] Kombinasi Fitur Kuantitatif Chronos & SMC Sukses!")
    else:
        print("[4] Amazon Chronos dilewati.")

    # 4. Gabungkan Multi-Timeframe secara aman
    print("[5] Menggabungkan data multi-timeframe menggunakan pd.merge_asof...")
    feats_low = feats_low.sort_index()
    feats_high_shifted = feats_high_shifted.sort_index()
    
    final_dataset = pd.merge_asof(
        feats_low,
        feats_high_shifted,
        left_index=True,
        right_index=True,
        direction="backward"
    )
    
    print("--- PROSES PENYUSUNAN DATASET SELESAI ---")
    return final_dataset


if __name__ == "__main__":
    print("=== PENGUJIAN MODUL ALIGNMENT ===")
    from smc_features import _synthetic_ohlcv
    
    # Simulasi data
    idx_15m = pd.date_range("2026-01-01", periods=500, freq="15min")
    df_15m = _synthetic_ohlcv(n=500, seed=42)
    df_15m.index = idx_15m
    
    idx_1d = pd.date_range("2025-12-25", periods=20, freq="1D")
    df_1d = _synthetic_ohlcv(n=20, seed=24)
    df_1d.index = idx_1d
    
    # Jalankan prapemrosesan hibrida
    dataset = build_hybrid_dataset(df_15m, df_1d, run_chronos=True)
    
    print("\nRingkasan kolom hasil penggabungan hibrida:")
    print(list(dataset.columns))
    
    print("\nTampilan 5 baris terakhir dari dataset hibrida:")
    cols_to_print = ['close', 'premium_discount_pct', 'in_discount', 'chronos_trend', 'chronos_breach_sh', 'htf_premium_discount_pct']
    print(dataset[cols_to_print].tail(5).to_string())
