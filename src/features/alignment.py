"""
alignment.py
Modul untuk menyelaraskan data multi-timeframe (15m & 1D) secara LEAK-FREE
dan memperkaya dataset dengan fitur SMC objektif (Premium/Discount Zone) serta 
fitur probabilistik dari Amazon Chronos.
"""

import numpy as np
import pandas as pd
from smc_features import build_features, calculate_premium_discount
from forecasting.chronos_wrapper import ChronosForecaster


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
        
        # Gunakan kuantil ASLI dari model Chronos (bukan aproksimasi Z-score)
        # chronos_q90_end dan chronos_q10_end adalah rasio harga (q/close)
        close_low = df_low_tf['close'].to_numpy()
        run_sh_low = close_low * (1.0 + feats_low['dist_to_swing_high'].to_numpy())
        run_sl_low = close_low * (1.0 - feats_low['dist_to_swing_low'].to_numpy())
        
        # Proyeksi harga absolut dari kuantil asli Chronos
        actual_q90 = close_low * feats_low['chronos_q90_end'].to_numpy()
        actual_q10 = close_low * feats_low['chronos_q10_end'].to_numpy()
        
        feats_low['chronos_breach_sh'] = (actual_q90 > run_sh_low).astype(int)
        feats_low['chronos_breach_sl'] = (actual_q10 < run_sl_low).astype(int)
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
