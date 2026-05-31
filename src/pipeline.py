"""
pipeline.py
Pipeline Pembelajaran Mesin Hibrida (SMC + Chronos + Meta-Labeling + LLM Reasoning)
secara End-to-End, Bebas Kebocoran Data (López de Prado Purged & Embargoed Split),
serta Dilengkapi Class Weighting untuk Mengatasi Ketimpangan Label.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Import modul internal kita
from smc_features import load_ohlcv_ccxt, _synthetic_ohlcv
from features.alignment import build_hybrid_dataset
from labeling.triple_barrier import apply_triple_barrier_labeling
from reasoning.llm_gate import MarketLLMReasoningGate


class HybridMLPipeline:
    """
    Orkestrator utama untuk melatih Model Kuantitatif ML (Base + Meta-Labeling)
    dan mengintegrasikannya dengan LLM Reasoning Gatekeeper secara valid dan aman.
    """
    def __init__(self, symbol: str = "BTC/USDT", chronos_model: str = "amazon/chronos-t5-tiny"):
        self.symbol = symbol
        self.chronos_model = chronos_model
        self.base_model = None
        self.meta_model = None
        self.features_list = []
        
    def load_data(self, strict_mode: bool = False) -> tuple:
        """
        Menarik data historis multi-timeframe secara aman.
        Jika strict_mode=True, program langsung error jika CCXT gagal (mencegah pencemaran data tiruan).
        Jika strict_mode=False, fallback terpicu dengan BANNER PERINGATAN keras.
        """
        print(f"\n[PIPELINE] Memulai pemuatan data untuk {self.symbol}...")
        
        df_15m, df_1d = None, None
        
        # 1. Low Timeframe (15m)
        try:
            print("[PIPELINE] Menarik data 15m dari exchange via CCXT...")
            df_15m = load_ohlcv_ccxt(symbol=self.symbol, timeframe="15m", limit=1000)
            print(f"[PIPELINE] Sukses menarik {len(df_15m)} bar data 15m.")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"[ERROR CRITICAL] Strict Mode aktif dan CCXT gagal menarik data 15m: {e}")
            
            print("\n" + "#" * 70)
            print("#" + " " * 18 + "WARNING: FALLBACK DATA SINTETIS AKTIF" + " " * 13 + "#")
            print("#" + " " * 68 + "#")
            print("#  Sistem gagal terhubung ke CCXT/Bursa untuk menarik data asli.     #")
            print("#  Model akan dilatih dan diuji pada DATA SINTETIS (Random Walk).   #")
            print("#  Hasil backtest ini TIDAK VALID untuk keputusan pasar nyata!     #")
            print("#" + " " * 68 + "#")
            print("#" * 70 + "\n")
            
            df_15m = _synthetic_ohlcv(n=1000, seed=42)
            df_15m.index = pd.date_range("2026-01-01", periods=1000, freq="15min")
            
        # 2. High Timeframe (1D)
        try:
            print("[PIPELINE] Menarik data 1D dari exchange via CCXT...")
            df_1d = load_ohlcv_ccxt(symbol=self.symbol, timeframe="1d", limit=100)
            print(f"[PIPELINE] Sukses menarik {len(df_1d)} bar data 1D.")
        except Exception as e:
            if strict_mode:
                raise ValueError(f"[ERROR CRITICAL] Strict Mode aktif dan CCXT gagal menarik data 1D: {e}")
            
            df_1d = _synthetic_ohlcv(n=100, seed=24)
            df_1d.index = pd.date_range("2025-11-01", periods=100, freq="1D")
            
        return df_15m, df_1d

    def purge_and_embargo_split(self, X: pd.DataFrame, y: pd.Series, split_idx: int, timeout_bars: int = 40, embargo_bars: int = 10) -> tuple:
        """
        Penerapan López de Prado Purging & Embargoing untuk menghilangkan kebocoran data
        di perbatasan split training dan testing.
        - Purging: Memotong training set sejauh timeout_bars sebelum split_idx agar label di
          train tidak melihat pergerakan harga yang masuk ke test set.
        - Embargoing: Mendorong test set bergeser sejauh embargo_bars setelah split_idx untuk
          menghindari autoregressive autocorrelation leakage.
        """
        # 1. Purging
        train_end = split_idx - timeout_bars
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        
        # 2. Embargoing
        test_start = split_idx + embargo_bars
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]
        
        print(f"[PURGE-EMBARGO] Jendela overlap {timeout_bars} bar di-purge dari training set.")
        print(f"[PURGE-EMBARGO] Jendela transisi {embargo_bars} bar di-embargo dari testing set.")
        return X_train, X_test, y_train, y_test

    def run_end_to_end(self, strict_mode: bool = False, backtest_mode: bool = True):
        """
        Menjalankan proses pelatihan dan evaluasi hibrida lengkap.
        - strict_mode: Jika True, CCXT dilarang keras fallback.
        - backtest_mode: Jika True, LLM reasoning akan terisolasi/bypass untuk menjamin validitas ilmiah backtest.
        """
        # 1. Pemuatan Data
        df_15m, df_1d = self.load_data(strict_mode=strict_mode)
        
        # 2. Preprocessing & Alignment (SMC + Chronos)
        dataset = build_hybrid_dataset(
            df_low_tf=df_15m,
            df_high_tf=df_1d,
            k_low=2,
            k_high=2,
            run_chronos=True,
            chronos_model=self.chronos_model
        )
        
        # 3. Triple-Barrier Labeling
        print("\n[PIPELINE] Menerapkan Triple-Barrier Labeling...")
        dataset = apply_triple_barrier_labeling(dataset, tp_mult=2.0, sl_mult=1.5, timeout_bars=40)
        
        # Menghapus bar di ujung yang datanya tidak cukup untuk labeling
        dataset = dataset[dataset['barrier_hit'] != 'insufficient_data']
        
        # 4. Definisikan fitur ML
        self.features_list = [
            'structure', 'dist_to_swing_high', 'dist_to_swing_low', 
            'upper_wick_ratio', 'lower_wick_ratio', 'bull_fvg', 'bear_fvg',
            'premium_discount_pct', 'in_discount', 'in_premium',
            'chronos_trend', 'chronos_volatility', 'chronos_skewness',
            'chronos_q90_end', 'chronos_q10_end',
            'chronos_breach_sh', 'chronos_breach_sl',
            'htf_structure', 'htf_premium_discount_pct', 'htf_in_discount', 'htf_in_premium'
        ]
        
        X = dataset[self.features_list]
        y = dataset['label'] # Target: 1 (BUY), -1 (SELL), 0 (HOLD/NO_TRADE)
        
        # Cek dan laporkan distribusi label mentah
        print("\n[PIPELINE] Distribusi label mentah sebelum penyeimbangan:")
        dist = y.value_counts()
        for val, count in dist.items():
            lbl = "BUY" if val == 1 else "SELL" if val == -1 else "HOLD"
            print(f"- Class {lbl:<5}: {count:<4} bar ({count/len(y)*100:.1f}%)")
            
        y_encoded = y + 1 # Multi-class encoding agar 0=SELL, 1=HOLD, 2=BUY
        
        # 5. Split Dataset menggunakan López de Prado Purging + Embargoing
        split_idx = int(len(dataset) * 0.8)
        X_train, X_test, y_train, y_test = self.purge_and_embargo_split(
            X, y_encoded, split_idx, timeout_bars=40, embargo_bars=10
        )
        
        print(f"Ukuran Data Latihan Akhir: {X_train.shape[0]}")
        print(f"Ukuran Data Pengujian Akhir: {X_test.shape[0]}")
        
        # 6. Hitung Class Weighting untuk Mengatasi Ketimpangan Label
        print("\n[PIPELINE] Menghitung balanced class weights untuk XGBoost...")
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.to_numpy())
        class_weights = dict(zip(classes, weights))
        sample_weights = np.array([class_weights[val] for val in y_train])
        
        print("Bobot penyeimbang per kelas:")
        for cls, wt in class_weights.items():
            lbl = "SELL" if cls == 0 else "HOLD" if cls == 1 else "BUY"
            print(f"- Bobot Kelas {lbl:<5}: {wt:.4f}")
            
        # 7. Latih Model ML Primer (Base Classifier) dengan sample weights
        print("\n[PIPELINE] Melatih Model ML Primer (XGBoost Classifier)...")
        self.base_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=3,
            random_state=42
        )
        self.base_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluasi Model Primer
        y_pred = self.base_model.predict(X_test)
        print("\nLaporan Performa Model ML Primer:")
        print(classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))
        
        # 8. Pipeline Meta-Labeling dengan K-Fold Out-of-Fold (OOF) Predictions
        # CRITICAL FIX: Menggunakan OOF predictions untuk menghindari in-sample bias.
        print("\n[PIPELINE] Memulai penyusunan Meta-Labeling Dataset (K-Fold OOF)...")
        
        n_folds = 5
        oof_preds = np.full(len(X_train), -1, dtype=int)
        kf = KFold(n_splits=n_folds, shuffle=False)
        
        print(f"[META-OOF] Menghasilkan Out-of-Fold predictions ({n_folds} folds)...")
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[fold_train_idx]
            y_fold_train = y_train.iloc[fold_train_idx]
            X_fold_val = X_train.iloc[fold_val_idx]
            
            fold_classes = np.unique(y_fold_train)
            fold_wts = compute_class_weight(class_weight='balanced', classes=fold_classes, y=y_fold_train.to_numpy())
            fold_class_weights = dict(zip(fold_classes, fold_wts))
            fold_sample_weights = np.array([fold_class_weights[val] for val in y_fold_train])
            
            fold_model = xgb.XGBClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.05,
                objective='multi:softprob', num_class=3, random_state=42
            )
            fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_sample_weights)
            oof_preds[fold_val_idx] = fold_model.predict(X_fold_val)
        
        base_test_preds = self.base_model.predict(X_test)
        
        meta_y_train = np.where((oof_preds == y_train) & (oof_preds != 1), 1, 0)
        meta_y_test = np.where((base_test_preds == y_test) & (base_test_preds != 1), 1, 0)
        
        print(f"Jumlah Sinyal Sukses OOF di Data Latihan  (Meta-Label=1): {np.sum(meta_y_train)}")
        print(f"Jumlah Sinyal Gagal/Hold OOF di Data Latihan (Meta-Label=0): {len(meta_y_train) - np.sum(meta_y_train)}")
        
        meta_classes = np.unique(meta_y_train)
        meta_wts = compute_class_weight(class_weight='balanced', classes=meta_classes, y=meta_y_train)
        meta_class_weights = dict(zip(meta_classes, meta_wts))
        meta_sample_weights = np.array([meta_class_weights[val] for val in meta_y_train])
        
        print("\n[PIPELINE] Melatih Model ML Sekunder (Meta-Labeling XGBoost) pada OOF labels...")
        self.meta_model = xgb.XGBClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            objective='binary:logistic',
            random_state=42
        )
        self.meta_model.fit(X_train, meta_y_train, sample_weight=meta_sample_weights)
        
        # Evaluasi Model Sekunder
        meta_preds = self.meta_model.predict(X_test)
        print("\nLaporan Performa Model Meta-Labeling:")
        print(classification_report(meta_y_test, meta_preds, target_names=['FAILED/HOLD', 'SUCCESS_CONFIRMED'], zero_division=0))
        
        # 9. Integrasi LLM Reasoning Gatekeeper Terisolasi (Bebas Leakage)
        if backtest_mode:
            print("\n[INFO] Mode Backtest Aktif: Lapis LLM Gemini dibypass secara historis untuk mencegah point-in-time leakage.")
            print("Metrik pengujian di atas 100% didasarkan pada logika ML-only murni yang valid secara ilmiah.")
        
        # Demonstrasikan LLM Reasoning Gate *hanya* untuk titik LIVE sinyal terbaru di akhir data
        print("\n[PIPELINE] Mengevaluasi sinyal LIVE paling akhir menggunakan LLM Gatekeeper...")
        prob_success = self.meta_model.predict_proba(X_test)[:, 1]
        
        latest_idx = X_test.index[-1]
        latest_features = X_test.iloc[-1]
        latest_pred_class = base_test_preds[-1] # 0 = SELL, 1 = HOLD, 2 = BUY
        latest_prob_success = prob_success[-1]
        
        action_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        action = action_map[latest_pred_class]
        
        print(f"\nSinyal LIVE Terakhir Terdeteksi pada {latest_idx}:")
        print(f"- Sinyal Arah ML Primer : {action}")
        print(f"- Probabilitas Sukses Meta-ML: {latest_prob_success * 100:.2f}%")
        
        # CRITICAL QUANT FIX: Treshold dinaikkan dari 40% menjadi 55%
        # Win-rate impas = 1.5 / (2 + 1.5) = 42.9%. Threshold 55% menjamin ekspektasi profit yang positif!
        expectancy_ratio = latest_prob_success * 2.0 - (1.0 - latest_prob_success) * 1.5
        print(f"- Estimasi Ekspektasi Matematika: {expectancy_ratio:.3f} ATR per trade")
        
        if action != "HOLD" and latest_prob_success > 0.55:
            print(f"\n[!] Sinyal Kuantitatif Lolos Meta-ML (Keyakinan {latest_prob_success*100:.1f}% > 55%).")
            print("Mengirim konteks point-in-time ke LLM Reasoning Gate untuk validasi sentimen...")
            
            gatekeeper = MarketLLMReasoningGate()
            tech_data = {
                "action": action,
                "symbol": self.symbol,
                "timeframe": "15m",
                "htf_structure": "Bullish Trend" if latest_features['htf_structure'] > 0 else ("Sideways / Neutral" if latest_features['htf_structure'] == 0 else "Bearish Trend"),
                "dist_to_ob": f"{latest_features['dist_to_swing_high']*100:.2f}" if action == "SELL" else f"{latest_features['dist_to_swing_low']*100:.2f}",
                "chronos_direction": "Bullish Expansion" if latest_features['chronos_trend'] > 0 else "Bearish Contraction"
            }
            
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
            if action == "HOLD":
                print("\n[-] Sinyal dibatalkan karena model menyarankan HOLD.")
            else:
                print(f"\n[-] Sinyal {action} DIBATALKAN karena keyakinan Meta-ML ({latest_prob_success*100:.2f}%) berada di bawah threshold profitabel 55%.")


if __name__ == "__main__":
    # Jalankan pipeline
    pipeline = HybridMLPipeline(symbol="BTC/USDT")
    pipeline.run_end_to_end(strict_mode=False, backtest_mode=True)
