"""
chronos_wrapper.py
Mengintegrasikan model Amazon Chronos Forecasting secara maksimal untuk 
mengekstrak fitur ramalan probabilistik tanpa look-ahead bias.
"""

import numpy as np
import pandas as pd
import warnings

# Coba import PyTorch dan Chronos Pipeline
try:
    import torch
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


class ChronosForecaster:
    """
    Pembungkus pipeline Chronos untuk menghasilkan ramalan probabilistik 
    dan mengekstrak fitur exogenous kuantitatif untuk model ML & LLM.
    """
    def __init__(self, model_name: str = "amazon/chronos-t5-tiny", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        
        if CHRONOS_AVAILABLE:
            try:
                print(f"[CHRONOS] Memuat model {model_name} pada device {device}...")
                # Tiny: 8M params, Mini: 20M params
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    model_name,
                    device_map=device,
                    torch_dtype=torch.float32 if device == "cpu" else torch.float16
                )
                print(f"[CHRONOS] Model {model_name} berhasil dimuat.")
            except Exception as e:
                warnings.warn(f"Gagal memuat model Chronos: {e}. Menggunakan mode Fallback.")
                self.pipeline = None
        else:
            print("[CHRONOS] Library 'chronos-forecasting' tidak terdeteksi. Menggunakan mode Fallback (Simulasi).")

    def predict_probabilistic(self, context_data: np.ndarray, prediction_length: int = 12, num_samples: int = 20) -> dict:
        """
        Menghasilkan ramalan probabilistik (Kuantil 10%, 50%, 90%).
        Input context_data: 1D array harga Close historis.
        """
        if self.pipeline is not None and CHRONOS_AVAILABLE:
            try:
                # Chronos membutuhkan input tensor bertipe Float32 dengan shape (num_series, context_len)
                context_tensor = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    # Prediksi menghasilkan tensor shape (1, num_samples, prediction_length)
                    forecast = self.pipeline.predict(
                        context_tensor,
                        prediction_length=prediction_length,
                        num_samples=num_samples
                    )
                
                # Konversi ke numpy array
                forecast_samples = forecast[0].cpu().numpy() # shape: (num_samples, prediction_length)
                
                # Hitung kuantil
                q10 = np.percentile(forecast_samples, 10, axis=0)
                q50 = np.percentile(forecast_samples, 50, axis=0)
                q90 = np.percentile(forecast_samples, 90, axis=0)
                
                return {
                    "q10": q10,
                    "q50": q50,
                    "q90": q90,
                    "samples": forecast_samples
                }
            except Exception as e:
                warnings.warn(f"Error saat inferensi Chronos: {e}. Menggunakan mode Fallback.")
        
        # --- FALLBACK MODE (Simulasi Statistik Menggunakan Brownian Motion / Drift) ---
        last_val = context_data[-1]
        # Hitung drift historis sederhana dan volatilitas untuk membuat simulasi realistis
        returns = np.diff(context_data) / context_data[:-1]
        mean_ret = np.mean(returns) if len(returns) > 0 else 0.0
        std_ret = np.std(returns) if len(returns) > 1 else 0.002
        
        # Buat simulasi jalur harga ke depan
        paths = []
        for _ in range(num_samples):
            sim_path = []
            curr = last_val
            for _ in range(prediction_length):
                curr = curr * np.exp(np.random.normal(mean_ret, std_ret))
                sim_path.append(curr)
            paths.append(sim_path)
            
        paths = np.array(paths)
        return {
            "q10": np.percentile(paths, 10, axis=0),
            "q50": np.percentile(paths, 50, axis=0),
            "q90": np.percentile(paths, 90, axis=0),
            "samples": paths
        }

    def extract_features(self, df: pd.DataFrame, prediction_length: int = 12, context_length: int = 100, step: int = 4) -> pd.DataFrame:
        """
        Mengekstrak Fitur Kuantitatif dari ramalan Chronos tanpa look-ahead bias.
        Menggunakan step inference (setiap `step` bar) untuk efisiensi CPU.
        Fitur dihitung dari kuantil ASLI model Chronos, bukan aproksimasi Z-score.
        """
        n = len(df)
        close = df['close'].to_numpy()
        
        chronos_trend = np.zeros(n)
        chronos_volatility = np.zeros(n)
        chronos_skewness = np.zeros(n)
        chronos_q90_end = np.zeros(n)
        chronos_q10_end = np.zeros(n)
        
        print(f"[CHRONOS] Mengekstrak fitur probabilistik untuk {n} baris (step={step})...")
        
        last_fc = None
        for i in range(context_length, n):
            # Hanya jalankan inferensi model setiap `step` bar untuk efisiensi
            anchor_idx = i - (i - context_length) % step
            if (i - context_length) % step == 0 or last_fc is None:
                context = close[i - context_length + 1 : i + 1]
                last_fc = self.predict_probabilistic(context, prediction_length=prediction_length)
            
            fc = last_fc
            q10 = fc["q10"]
            q50 = fc["q50"]
            q90 = fc["q90"]
            
            # Ambil harga Close pada saat ramalan dibuat (anchor bar)
            close_anchor = close[anchor_idx]
            
            # 1. Trend: Log-return dari harga anchor ke median prediksi di akhir horizon
            chronos_trend[i] = np.log(q50[-1] / close_anchor)
            
            # 2. Volatility: Lebar interval prediksi ASLI (q90 - q10) dinormalisasi
            chronos_volatility[i] = (q90[-1] - q10[-1]) / close_anchor
            
            # 3. Skewness: Bias arah distribusi probabilistik
            range_up = q90[-1] - q50[-1]
            range_down = q50[-1] - q10[-1]
            denom = range_up + range_down
            chronos_skewness[i] = (range_up - range_down) / (denom + 1e-12)
            
            # 4 & 5. Endpoint kuantil asli (dinormalisasi) untuk breach detection di alignment.py
            chronos_q90_end[i] = q90[-1] / close_anchor
            chronos_q10_end[i] = q10[-1] / close_anchor
            
        out = pd.DataFrame(index=df.index)
        out["chronos_trend"] = chronos_trend
        out["chronos_volatility"] = chronos_volatility
        out["chronos_skewness"] = chronos_skewness
        out["chronos_q90_end"] = chronos_q90_end
        out["chronos_q10_end"] = chronos_q10_end
        
        out = out.fillna(0.0)
        return out


if __name__ == "__main__":
    # Test internal
    print("=== PENGUJIAN MODUL CHRONOS WRAPPER ===")
    from smc_features import _synthetic_ohlcv
    
    df = _synthetic_ohlcv(n=120)
    forecaster = ChronosForecaster()
    
    # Ambil data historis singkat
    context = df['close'].values[:100]
    forecast = forecaster.predict_probabilistic(context, prediction_length=12)
    
    print("\nHasil Uji Prediksi Probabilistik (12 Bar ke Depan):")
    print(f"Harga Close Terakhir : {context[-1]:.2f}")
    print(f"Prediksi Kuantil 10% : {forecast['q10'][-1]:.2f}")
    print(f"Prediksi Kuantil 50% : {forecast['q50'][-1]:.2f}")
    print(f"Prediksi Kuantil 90% : {forecast['q90'][-1]:.2f}")
    
    print("\nMengekstrak Fitur Rolling...")
    features = forecaster.extract_features(df, prediction_length=12, context_length=100)
    print("\nContoh 5 baris terakhir fitur Chronos:")
    print(features.tail(5).to_string())
