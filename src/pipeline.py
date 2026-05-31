"""
pipeline.py
Pipeline Pembelajaran Mesin Hibrida (SMC + Chronos + Meta-Labeling + LLM Reasoning)
secara End-to-End dan Bebas Kebocoran Data (Leak-Free).
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Import modul internal kita
from smc_features import load_ohlcv_ccxt, _synthetic_ohlcv
from features.alignment import build_hybrid_dataset
from labeling.triple_barrier import apply_triple_barrier_labeling
from reasoning.llm_gate import MarketLLMReasoningGate


class HybridMLPipeline:
    """
    Orkestrator utama untuk melatih Model Kuantitatif ML (Base + Meta-Labeling)
    dan mengintegrasikannya dengan LLM Reasoning Gatekeeper.
    """
    def __init__(self, symbol: str = "BTC/USDT", chronos_model: str = "amazon/chronos-t5-tiny"):
        self.symbol = symbol
        self.chronos_model = chronos_model
        self.base_model = None
        self.meta_model = None
        self.features_list = []
        
    def load_data(self) -> tuple:
        """
        Menarik data historis multi-timeframe secara aman.
        Jika CCXT gagal (misal tidak ada koneksi internet), otomatis fallback ke data sintetis.
        """
        print(f"\n[PIPELINE] Memulai pemuatan data untuk {self.symbol}...")
        
        # 1. Low Timeframe (15m)
        try:
            print("[PIPELINE] Menarik data 15m dari exchange via CCXT...")
            df_15m = load_ohlcv_ccxt(symbol=self.symbol, timeframe="15m", limit=1000)
            print(f"[PIPELINE] Sukses menarik {len(df_15m)} bar data 15m.")
        except Exception as e:
            print(f"[WARN] Gagal memuat data via CCXT: {e}. Mengaktifkan Fallback Data Sintetis 15m...")
            df_15m = _synthetic_ohlcv(n=1000, seed=42)
            # Buat indeks datetime simulasi
            df_15m.index = pd.date_range("2026-01-01", periods=1000, freq="15min")
            
        # 2. High Timeframe (1D)
        try:
            print("[PIPELINE] Menarik data 1D dari exchange via CCXT...")
            df_1d = load_ohlcv_ccxt(symbol=self.symbol, timeframe="1d", limit=100)
            print(f"[PIPELINE] Sukses menarik {len(df_1d)} bar data 1D.")
        except Exception as e:
            print(f"[WARN] Gagal memuat data via CCXT: {e}. Mengaktifkan Fallback Data Sintetis 1D...")
            df_1d = _synthetic_ohlcv(n=100, seed=24)
            df_1d.index = pd.date_range("2025-11-01", periods=100, freq="1D")
            
        return df_15m, df_1d

    def run_end_to_end(self):
        # 1. Pemuatan Data
        df_15m, df_1d = self.load_data()
        
        # 2. Preprocessing & Alignment (SMC + Chronos)
        # Kita set run_chronos=True untuk mengaktifkan peramalan probabilistik Chronos
        dataset = build_hybrid_dataset(
            df_low_tf=df_15m,
            df_high_tf=df_1d,
            k_low=2,
            k_high=2,
            run_chronos=True,
            chronos_model=self.chronos_model
        )
        
        # 3. Triple-Barrier Labeling (Target ML Primer)
        print("\n[PIPELINE] Menerapkan Triple-Barrier Labeling...")
        dataset = apply_triple_barrier_labeling(dataset, tp_mult=2.0, sl_mult=1.5, timeout_bars=40)
        
        # Menghapus bar di ujung yang datanya tidak cukup untuk labeling
        dataset = dataset[dataset['barrier_hit'] != 'insufficient_data']
        
        # 4. Definisikan fitur ML
        # Kita pilih fitur-fitur teknikal SMC dan ramalan Chronos secara selektif
        self.features_list = [
            'structure', 'dist_to_swing_high', 'dist_to_swing_low', 
            'upper_wick_ratio', 'lower_wick_ratio', 'bull_fvg', 'bear_fvg',
            'premium_discount_pct', 'in_discount', 'in_premium',
            'chronos_trend', 'chronos_volatility', 'chronos_skewness',
            'chronos_breach_sh', 'chronos_breach_sl',
            'htf_structure', 'htf_premium_discount_pct', 'htf_in_discount', 'htf_in_premium'
        ]
        
        X = dataset[self.features_list]
        y = dataset['label'] # Target: 1 (BUY), -1 (SELL), 0 (HOLD/NO_TRADE)
        
        # Kita ubah label menjadi kelas 0, 1, 2 untuk XGBoost multi-class
        # -1 -> 0 (SELL), 0 -> 1 (HOLD), 1 -> 2 (BUY)
        y_encoded = y + 1
        
        # Split Data secara berurutan (Time-Series Split) untuk mencegah kebocoran temporal
        print("\n[PIPELINE] Membagi dataset menggunakan Time-Series Split...")
        split_idx = int(len(dataset) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]
        
        print(f"Ukuran Data Latihan: {X_train.shape[0]}")
        print(f"Ukuran Data Pengujian: {X_test.shape[0]}")
        
        # 5. Latih Base Classifier (Model ML Primer)
        print("\n[PIPELINE] Melatih Model ML Primer (XGBoost Classifier)...")
        self.base_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            random_state=42
        )
        self.base_model.fit(X_train, y_train)
        
        # Evaluasi Model Primer
        y_pred = self.base_model.predict(X_test)
        print("\nLaporan Performa Model ML Primer:")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
        
        # 6. Pipeline Meta-Labeling (Model ML Sekunder)
        # Model sekunder memprediksi apakah sinyal BUY/SELL dari Model Primer BENAR-BENAR sukses.
        print("\n[PIPELINE] Memulai penyusunan Meta-Labeling Dataset...")
        
        # Ambil probabilitas prediksi model primer pada data latihan
        base_train_preds = self.base_model.predict(X_train)
        base_test_preds = self.base_model.predict(X_test)
        
        # Meta-label = 1 jika prediksi model primer cocok dengan label asli dan merupakan aksi aktif (BUY/SELL)
        # Meta-label = 0 jika prediksi salah atau HOLD
        meta_y_train = np.where((base_train_preds == y_train) & (base_train_preds != 1), 1, 0)
        meta_y_test = np.where((base_test_preds == y_test) & (base_test_preds != 1), 1, 0)
        
        print("Jumlah Sinyal Sukses di Data Latihan  (Meta-Label=1):", np.sum(meta_y_train))
        print("Jumlah Sinyal Gagal/Hold di Data Latihan (Meta-Label=0):", len(meta_y_train) - np.sum(meta_y_train))
        
        print("\n[PIPELINE] Melatih Model ML Sekunder (Meta-Labeling XGBoost)...")
        self.meta_model = xgb.XGBClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42
        )
        self.meta_model.fit(X_train, meta_y_train)
        
        # Evaluasi Model Sekunder
        meta_preds = self.meta_model.predict(X_test)
        print("\nLaporan Performa Model Meta-Labeling:")
        print(classification_report(meta_y_test, meta_preds, target_names=['FAILED/HOLD', 'SUCCESS_CONFIRMED']))
        
        # 7. Integrasi LLM Reasoning Gatekeeper
        print("\n[PIPELINE] Mendemonstrasikan integrasi LLM Gatekeeper pada Sinyal Aktif terakhir...")
        
        # Cari indeks baris terakhir yang menghasilkan sinyal aktif (BUY atau SELL)
        prob_success = self.meta_model.predict_proba(X_test)[:, 1]
        
        # Dapatkan baris terakhir di data test
        latest_idx = X_test.index[-1]
        latest_features = X_test.iloc[-1]
        latest_pred_class = base_test_preds[-1] # 0 = SELL, 1 = HOLD, 2 = BUY
        latest_prob_success = prob_success[-1]
        
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map[latest_pred_class]
        
        print(f"\nSinyal Terakhir Terdeteksi pada {latest_idx}:")
        print(f"- Sinyal Arah ML Primer : {action}")
        print(f"- Probabilitas Sukses Meta-ML: {latest_prob_success * 100:.2f}%")
        
        # Jika probabilitas sukses melampaui ambang batas (> 40% untuk kebutuhan simulasi tugas)
        if action != "HOLD" and latest_prob_success > 0.40:
            print("\n[!] Sinyal Kuantitatif Lolos Penyaring Meta-Labeling. Mengirim data ke LLM Reasoning Gate...")
            
            gatekeeper = MarketLLMReasoningGate()
            
            # Ekstrak data kuantitatif pasar terkompresi
            tech_data = {
                "action": action,
                "symbol": self.symbol,
                "timeframe": "15m",
                "htf_structure": "Bullish Trend" if latest_features['htf_structure'] > 0 else "Bearish Trend",
                "dist_to_ob": f"{latest_features['dist_to_swing_high']*100:.2f}" if action == "SELL" else f"{latest_features['dist_to_swing_low']*100:.2f}",
                "chronos_direction": "Bullish Expansion" if latest_features['chronos_trend'] > 0 else "Bearish Contraction"
            }
            
            # Berita/Sentimen fundamental makro FRED + kalender
            fundamental_news = [
                {"source": "FRED", "headline": "US Inflation Rate cools down to 3.1% YoY (supporting assets)"},
                {"source": "Exchange Book", "headline": "Heavy limit buy order blocks stacked right below current price"}
            ]
            
            prompt = gatekeeper.generate_prompt(tech_data, fundamental_news)
            decision_json = gatekeeper.query_reasoning(prompt)
            
            print("\n=== RESPON PENALARAN LLM GATEKEEPER ===")
            print(f"Keputusan Akhir : {decision_json.get('decision')}")
            print(f"Skor Keyakinan  : {decision_json.get('confidence_score') * 100:.1f}%")
            print(f"Analisis Logis  : {decision_json.get('rationale')}")
            print("=======================================")
        else:
            print("\n[-] Sinyal dibatalkan karena model menyarankan HOLD atau tingkat keyakinan Meta-Labeling di bawah ambang batas.")


if __name__ == "__main__":
    # Jalankan pipeline
    pipeline = HybridMLPipeline(symbol="BTC/USDT")
    pipeline.run_end_to_end()
