"""
triple_barrier.py
Modul pelabelan dinamis menggunakan metode Triple-Barrier yang disesuaikan 
dengan volatilitas pasar aktual (Average True Range).
"""

import numpy as np
import pandas as pd


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Menghitung Average True Range (ATR) secara aman untuk mengukur 
    volatilitas pasar saat ini tanpa kebocoran data.
    """
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    close = df['close'].to_numpy()
    n = len(df)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
        
    return pd.Series(tr, index=df.index).rolling(period).mean()


def apply_triple_barrier_labeling(df: pd.DataFrame, tp_mult: float = 2.0, sl_mult: float = 1.5, timeout_bars: int = 40) -> pd.DataFrame:
    """
    Menerapkan pelabelan Triple-Barrier dinamis pada dataset trading.
    
    Tiga Batasan (Barriers):
    1. Upper Barrier (TP): close[i] + (tp_mult * ATR[i])
    2. Lower Barrier (SL): close[i] - (sl_mult * ATR[i])
    3. Vertical Barrier (Time-out): i + timeout_bars
    
    Output:
    - 1: Menyentuh TP terlebih dahulu (Sinyal Sukses/Bullish)
    - -1: Menyentuh SL terlebih dahulu (Sinyal Gagal/Bearish)
    - 0: Tidak menyentuh TP maupun SL hingga waktu habis (Netral/Timeout)
    """
    df = df.copy()
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    n = len(df)
    
    # Hitung volatilitas dinamis (ATR)
    atr_series = calculate_atr(df, period=14)
    # Gunakan bfill untuk menangani bar awal yang bernilai NaN
    atr = atr_series.fillna(method='bfill').to_numpy()
    
    labels = np.zeros(n, dtype=int)
    barrier_hits = []
    
    for i in range(n - timeout_bars):
        volatility = atr[i]
        
        # Level harga absolut untuk TP & SL
        tp_level = close[i] + (tp_mult * volatility)
        sl_level = close[i] - (sl_mult * volatility)
        
        hit = 0
        hit_type = "timeout"
        
        # Jelajahi bar-bar berikutnya ke depan
        for step in range(1, timeout_bars + 1):
            idx = i + step
            
            # 1. Cek apakah menyentuh SL terlebih dahulu
            if low[idx] <= sl_level:
                hit = -1
                hit_type = "sl"
                break
            # 2. Cek apakah menyentuh TP terlebih dahulu
            elif high[idx] >= tp_level:
                hit = 1
                hit_type = "tp"
                break
                
        labels[i] = hit
        barrier_hits.append(hit_type)
        
    # Untuk sisa bar di ujung dataset
    for _ in range(n - timeout_bars, n):
        barrier_hits.append("insufficient_data")
        
    df['atr'] = atr_series
    df['label'] = labels
    df['barrier_hit'] = barrier_hits
    
    return df


if __name__ == "__main__":
    print("=== PENGUJIAN MODUL TRIPLE BARRIER LABELING ===")
    from smc_features import _synthetic_ohlcv
    
    df = _synthetic_ohlcv(n=200, seed=42)
    labeled = apply_triple_barrier_labeling(df, tp_mult=2.0, sl_mult=1.5, timeout_bars=20)
    
    print("\nStatistik Hasil Pelabelan:")
    print("TP Hits  (1) :", (labeled['label'] == 1).sum())
    print("SL Hits (-1) :", (labeled['label'] == -1).sum())
    print("Timeout  (0) :", (labeled['label'] == 0).sum())
    
    print("\nTampilan 5 data berlabel teratas:")
    print(labeled[['close', 'atr', 'label', 'barrier_hit']].head(5).to_string())
