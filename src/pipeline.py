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
        
        # 3.5. Tambahkan fitur normalisasi tambahan (log-return & historical volatility)
        print("[PIPELINE] Menghitung fitur normalisasi tambahan (log-return & historical volatility)...")
        dataset['log_return'] = np.log(dataset['close'] / dataset['close'].shift(1))
        dataset['historical_volatility'] = dataset['log_return'].rolling(14).std()
        
        dataset['log_return'] = dataset['log_return'].fillna(0.0)
        dataset['historical_volatility'] = dataset['historical_volatility'].fillna(0.0)
        
        # 4. Definisikan fitur ML (SMC + OB Memory + Harmonics + Chronos)
        self.features_list = [
            'structure', 'dist_to_swing_high', 'dist_to_swing_low', 
            'upper_wick_ratio', 'lower_wick_ratio', 'bull_fvg', 'bear_fvg',
            'premium_discount_pct', 'in_discount', 'in_premium',
            'chronos_trend', 'chronos_volatility', 'chronos_skewness',
            'chronos_q90_end', 'chronos_q10_end',
            'chronos_breach_sh', 'chronos_breach_sl',
            'htf_structure', 'htf_premium_discount_pct', 'htf_in_discount', 'htf_in_premium',
            'dist_to_unmitigated_bullish_ob', 'dist_to_unmitigated_bearish_ob', 'ob_fvg_confluence',
            'log_return', 'historical_volatility',
            'bull_gartley', 'bear_gartley', 'bull_butterfly', 'bear_butterfly',
            'bull_bat', 'bear_bat', 'bull_crab', 'bear_crab', 'bull_cypher', 'bear_cypher',
            'harmonic_bullish_signal', 'harmonic_bearish_signal'
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
               # 5. Walk-Forward Cross-Validation (Purged & Embargoed) Folds
        print("\n[PIPELINE] Memulai Pembagian Jendela Kronologis Walk-Forward Cross-Validation (3 Folds)...")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        n_bars = len(dataset)
        
        # Definisikan interval uji chronological secara proporsional
        fold_splits = [
            (500, 650),
            (650, 800),
            (800, n_bars)
        ]
        
        all_test_preds = []
        all_test_prob_success = []
        all_test_indices = []
        
        for fold_idx, (train_end, test_end) in enumerate(fold_splits, 1):
            if test_end > n_bars:
                test_end = n_bars
            if train_end >= test_end:
                continue
                
            print(f"\n--- MENGEKSEKUSI WALK-FORWARD FOLD {fold_idx} ---")
            print(f"- Jendela Training : Baris 0 s.d {train_end - 40} (di-purge)")
            print(f"- Jendela Testing  : Baris {train_end + 10} s.d {test_end}")
            
            # Splitting dengan purging & embargoing
            X_tr, X_te, y_tr, y_te = self.purge_and_embargo_split(
                X, y_encoded, train_end, timeout_bars=40, embargo_bars=10
            )
            
            # Pangkas test set fold ini agar tepat sesuai dengan test_end
            te_end_idx = test_end - (train_end + 10)
            if te_end_idx > 0:
                X_te = X_te.iloc[:te_end_idx]
                y_te = y_te.iloc[:te_end_idx]
                
            if len(X_tr) == 0 or len(X_te) == 0:
                print(f"[-] Fold {fold_idx} dilewati karena data latihan/uji kosong.")
                continue
                
            # Class weighting untuk melatih model primer
            tr_classes = np.unique(y_tr)
            tr_wts = compute_class_weight(class_weight='balanced', classes=tr_classes, y=y_tr.to_numpy())
            tr_class_weights = dict(zip(tr_classes, tr_wts))
            tr_sample_weights = np.array([tr_class_weights[val] for val in y_tr])
            
            # Optuna Bayesian optimization per fold (7 trials agar cepat & efisien di CPU)
            print(f"[OPTUNA-FOLD {fold_idx}] Optimasi model primer...")
            def base_obj(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 30, 80),
                    'max_depth': trial.suggest_int('max_depth', 3, 6),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'random_state': 42,
                    'n_jobs': -1
                }
                val_sz = int(len(X_tr) * 0.2)
                X_t, X_v = X_tr.iloc[:-val_sz], X_tr.iloc[-val_sz:]
                y_t, y_v = y_tr.iloc[:-val_sz], y_tr.iloc[-val_sz:]
                
                t_cls = np.unique(y_t)
                t_wts = compute_class_weight('balanced', classes=t_cls, y=y_t.to_numpy())
                t_class_weights = dict(zip(t_cls, t_wts))
                t_sample_weights = np.array([t_class_weights[val] for val in y_t])
                
                m = xgb.XGBClassifier(**p)
                m.fit(X_t, y_t, sample_weight=t_sample_weights)
                pr = m.predict(X_v)
                from sklearn.metrics import f1_score
                return f1_score(y_v, pr, average='weighted', zero_division=0)
                
            study_b = optuna.create_study(direction='maximize')
            study_b.optimize(base_obj, n_trials=7)
            best_p = study_b.best_params
            best_p.update({'objective': 'multi:softprob', 'num_class': 3, 'random_state': 42, 'n_jobs': -1})
            
            fold_base_model = xgb.XGBClassifier(**best_p)
            fold_base_model.fit(X_tr, y_tr, sample_weight=tr_sample_weights)
            
            # OOF Predictions untuk training Meta-Labeling
            n_f = 3
            o_preds = np.full(len(X_tr), -1, dtype=int)
            k_f = KFold(n_splits=n_f, shuffle=False)
            for f_tr_idx, f_va_idx in k_f.split(X_tr):
                X_f_tr = X_tr.iloc[f_tr_idx]
                y_f_tr = y_tr.iloc[f_tr_idx]
                X_f_va = X_tr.iloc[f_va_idx]
                
                f_cls = np.unique(y_f_tr)
                f_wts = compute_class_weight(class_weight='balanced', classes=f_cls, y=y_f_tr.to_numpy())
                f_class_weights = dict(zip(f_cls, f_wts))
                f_sample_weights = np.array([f_class_weights[val] for val in y_f_tr])
                
                f_model = xgb.XGBClassifier(n_estimators=40, max_depth=4, learning_rate=0.05, objective='multi:softprob', num_class=3, random_state=42)
                f_model.fit(X_f_tr, y_f_tr, sample_weight=f_sample_weights)
                o_preds[f_va_idx] = f_model.predict(X_f_va)
                
            active_tr_idx = np.where(o_preds != 1)[0]
            X_m_tr = X_tr.iloc[active_tr_idx]
            y_m_tr = np.where(o_preds[active_tr_idx] == y_tr.iloc[active_tr_idx], 1, 0)
            
            # Optuna Meta Model Tuning
            print(f"[OPTUNA-FOLD {fold_idx}] Optimasi meta-model...")
            def meta_obj(trial):
                p = {
                    'n_estimators': trial.suggest_int('n_estimators', 20, 60),
                    'max_depth': trial.suggest_int('max_depth', 2, 4),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.12),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'objective': 'binary:logistic',
                    'random_state': 42,
                    'n_jobs': -1
                }
                if len(X_m_tr) < 6: return 0.5
                val_sz = int(len(X_m_tr) * 0.2)
                if val_sz < 2: val_sz = 2
                X_t, X_v = X_m_tr.iloc[:-val_sz], X_m_tr.iloc[-val_sz:]
                y_t, y_v = y_m_tr[:-val_sz], y_m_tr[-val_sz:]
                
                t_cls = np.unique(y_t)
                if len(t_cls) < 2:
                    m = xgb.XGBClassifier(**p)
                    m.fit(X_t, y_t)
                    pr = m.predict(X_v)
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y_v, pr)
                    
                t_wts = compute_class_weight('balanced', classes=t_cls, y=y_t)
                t_class_weights = dict(zip(t_cls, t_wts))
                t_sample_weights = np.array([t_class_weights[val] for val in y_t])
                
                m = xgb.XGBClassifier(**p)
                m.fit(X_t, y_t, sample_weight=t_sample_weights)
                pr = m.predict(X_v)
                from sklearn.metrics import f1_score
                return f1_score(y_v, pr, average='binary', zero_division=0)
                
            study_m = optuna.create_study(direction='maximize')
            study_m.optimize(meta_obj, n_trials=7)
            best_m_p = study_m.best_params
            best_m_p.update({'objective': 'binary:logistic', 'random_state': 42, 'n_jobs': -1})
            
            fold_meta_model = xgb.XGBClassifier(**best_m_p)
            m_classes = np.unique(y_m_tr)
            if len(m_classes) >= 2:
                m_wts = compute_class_weight(class_weight='balanced', classes=m_classes, y=y_m_tr)
                m_class_weights = dict(zip(m_classes, m_wts))
                m_sample_weights = np.array([m_class_weights[val] for val in y_m_tr])
                fold_meta_model.fit(X_m_tr, y_m_tr, sample_weight=m_sample_weights)
            else:
                fold_meta_model.fit(X_m_tr, y_m_tr)
                
            # Evaluasi fold out-of-sample
            te_preds = fold_base_model.predict(X_te)
            te_prob_success = fold_meta_model.predict_proba(X_te)[:, 1]
            
            all_test_preds.extend(te_preds)
            all_test_prob_success.extend(te_prob_success)
            all_test_indices.extend(X_te.index)
            
            # Simpan model fold terakhir sebagai model aktif untuk LIVE
            self.base_model = fold_base_model
            self.meta_model = fold_meta_model
            self.features_list = list(X.columns)
            
        # Rekonstruksi Dataset Pengujian Out-of-Sample Terpadu
        X_test = X.loc[all_test_indices]
        y_test = y_encoded.loc[all_test_indices]
        base_test_preds = np.array(all_test_preds)
        prob_success = np.array(all_test_prob_success)
        
        print(f"\n[PIPELINE] Selesai Walk-Forward Cross-Validation!")
        print(f"- Total Ukuran Data Uji Out-of-Sample Terpadu: {len(X_test)}")
        
        # Evaluasi Primer Terpadu
        print("\nLaporan Performa Model ML Primer Terpadu (All Folds OOS):")
        print(classification_report(y_test, base_test_preds, target_names=['SELL', 'HOLD', 'BUY'], zero_division=0))
        
        # Evaluasi Meta Terpadu (pada sinyal aktif)
        active_te_idx = np.where(base_test_preds != 1)[0]
        if len(active_te_idx) > 0:
            meta_y_test = np.where(base_test_preds[active_te_idx] == y_test.iloc[active_te_idx], 1, 0)
            meta_preds = np.where(prob_success[active_te_idx] >= 0.5, 1, 0)
            print("\nLaporan Performa Model Meta-Labeling Terpadu (sinyal aktif saja):")
            print(classification_report(meta_y_test, meta_preds, target_names=['SIGNAL_FAILED', 'SIGNAL_SUCCESS'], zero_division=0))
        else:
            print("\n[INFO] Tidak ada sinyal aktif di data uji untuk evaluasi meta-model.")
        
        # 8.5. Rigorous Financial Backtest using VectorBT
        print("\n[PIPELINE] Memulai Simulasi Backtesting Finansial dengan VectorBT...")
        try:
            import vectorbt as vbt
            import matplotlib
            matplotlib.use('Agg') # Mencegah error display pada Windows/non-GUI env
            import matplotlib.pyplot as plt
            
            close_test = dataset.loc[X_test.index, 'close']
            prob_success = self.meta_model.predict_proba(X_test)[:, 1]
            
            # Sinyal boolean berdasarkan threshold manajemen risiko
            entries = (base_test_preds == 2) & (prob_success >= 0.45) # 2 = BUY
            short_entries = (base_test_preds == 0) & (prob_success >= 0.60) # 0 = SELL
            
            # --- KELLY POSITION SIZING ---
            # Formula Kelly: f* = p - (1-p)/b
            # Di mana b = payout ratio (BUY: 2.0/1.5 = 1.3333, SELL: 1.5/2.0 = 0.75)
            kelly_sizes = np.zeros(len(close_test))
            for idx in range(len(close_test)):
                pred = base_test_preds[idx]
                prob = prob_success[idx]
                
                if pred == 2: # BUY
                    b = 2.0 / 1.5
                    f_star = prob - (1.0 - prob) / b
                elif pred == 0: # SELL
                    b = 1.5 / 2.0
                    f_star = prob - (1.0 - prob) / b
                else:
                    f_star = 0.0
                    
                # Gunakan Half-Kelly untuk proteksi resiko drawdown (standard profesional)
                half_kelly = 0.5 * f_star
                # Batasi maksimum allocation 10% dan minimum 0%
                kelly_sizes[idx] = np.clip(half_kelly, 0.0, 0.10)
                
            # Long-Only Portfolio simulation untuk mencegah kendala reversal pada SizeType.Percent
            portfolio = vbt.Portfolio.from_signals(
                close=close_test,
                entries=entries,
                exits=short_entries,
                size=kelly_sizes,
                size_type="percent",
                fees=0.0006, # 0.06% Binance spot VIP 0
                slippage=0.0001, # 0.01% slippage
                freq="15min",
                init_cash=10000.0
            )
            
            # Ekstrak Metrik
            total_return = portfolio.total_return() * 100
            sharpe_ratio = portfolio.sharpe_ratio()
            sortino_ratio = portfolio.sortino_ratio()
            max_drawdown = portfolio.max_drawdown() * 100
            
            # Menggunakan API trades dari vectorbt untuk mengambil metrik transaksi
            total_trades = portfolio.trades.count()
            win_rate = portfolio.trades.win_rate() * 100 if total_trades > 0 else 0.0
            profit_factor = portfolio.trades.profit_factor() if total_trades > 0 else 0.0
            
            print("=====================================================")
            print("         LAPORAN BACKTEST PORTOFOLIO VECTORBT        ")
            print("=====================================================")
            print(f" Total Return          : {total_return:+.2f}%")
            print(f" Sharpe Ratio          : {sharpe_ratio:.4f}")
            print(f" Sortino Ratio         : {sortino_ratio:.4f}")
            print(f" Max Drawdown          : {max_drawdown:.2f}%")
            print(f" Total Trades          : {total_trades}")
            print(f" Win Rate              : {win_rate:.2f}%")
            print(f" Profit Factor         : {profit_factor:.4f}")
            print("=====================================================")
            
            # Plot dan Simpan Equity Curve
            plt.figure(figsize=(10, 5))
            portfolio.value().plot(title=f"Hybrid Trading Strategy Equity Curve ({self.symbol})", color="#00ffcc")
            plt.xlabel("Waktu")
            plt.ylabel("Saldo (USD)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig("equity_curve.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("[VBT-BACKTEST] Grafik Equity Curve berhasil disimpan ke 'equity_curve.png'.")
            
        except Exception as e:
            print(f"[VBT-BACKTEST ERROR] Gagal menjalankan backtest VectorBT: {e}")
            import traceback
            traceback.print_exc()

        # 8.7. SHAP Model Interpretability (Explainable AI - XAI)
        print("\n[PIPELINE] Memulai Komputasi Nilai SHAP (TreeExplainer)...")
        try:
            import shap
            import matplotlib.pyplot as plt
            
            explainer = shap.TreeExplainer(self.base_model)
            shap_values = explainer.shap_values(X_test)
            
            # shap 1.0+ handles multi-class: shap_values is a list of arrays (one for each class)
            # or a single array of shape (num_samples, num_features, num_classes).
            plt.figure(figsize=(10, 6))
            
            # Extract target shap (class 2 = BUY)
            if isinstance(shap_values, list):
                target_shap = shap_values[2] if len(shap_values) > 2 else shap_values[0]
            elif len(shap_values.shape) == 3: # shape: (samples, features, classes)
                target_shap = shap_values[:, :, 2] # class 2 (BUY)
            else:
                target_shap = shap_values
                
            shap.summary_plot(target_shap, X_test, plot_type="bar", show=False)
            plt.title(f"SHAP Feature Importance Summary Plot (BUY Class - {self.symbol})")
            plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("[SHAP-EXPLAINER] Grafik SHAP Feature Importance berhasil disimpan ke 'shap_importance.png'.")
            
            # Cetak 5 fitur paling berpengaruh berdasarkan rata-rata nilai absolut SHAP
            mean_abs_shap = np.mean(np.abs(target_shap), axis=0)
            feature_importance = pd.Series(mean_abs_shap, index=self.features_list).sort_values(ascending=False)
            print("\n=== 5 FITUR UTAMA PALING BERPENGARUH (SHAP AI) ===")
            for rank, (feat, val) in enumerate(feature_importance.head(5).items(), 1):
                print(f" {rank}. {feat:<35} : {val:.6f} mean(|SHAP|)")
            print("==================================================")
            
        except Exception as e:
            print(f"[SHAP ERROR] Gagal menghitung nilai SHAP: {e}")
            import traceback
            traceback.print_exc()

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
        
        # ASYMMETRIC R:R FIX: Ekspektasi BERBEDA untuk BUY vs SELL karena Triple-Barrier
        # memiliki rasio Upper(2.0 ATR)/Lower(1.5 ATR) yang ASIMETRIS.
        # - BUY: TP = Upper = 2.0 ATR, SL = Lower = 1.5 ATR → R:R = 2.0/1.5 = 1.33x
        #   Win-rate impas BUY = 1.5 / (2.0 + 1.5) = 42.9%
        # - SELL: TP = Lower = 1.5 ATR, SL = Upper = 2.0 ATR → R:R = 1.5/2.0 = 0.75x
        #   Win-rate impas SELL = 2.0 / (1.5 + 2.0) = 57.1%
        if action == "BUY":
            reward_atr, risk_atr = 2.0, 1.5
            threshold = 0.45  # Floor aman di atas impas 42.9%
        elif action == "SELL":
            reward_atr, risk_atr = 1.5, 2.0
            threshold = 0.60  # Floor aman di atas impas 57.1%
        else:
            reward_atr, risk_atr = 0.0, 0.0
            threshold = 1.0  # HOLD tidak pernah lolos
        
        expectancy_ratio = (latest_prob_success * reward_atr) - ((1.0 - latest_prob_success) * risk_atr)
        breakeven_wr = risk_atr / (reward_atr + risk_atr) if (reward_atr + risk_atr) > 0 else 0
        print(f"- Arah Sinyal        : {action}")
        print(f"- Reward/Risk (ATR)  : {reward_atr}/{risk_atr}")
        print(f"- Win-Rate Impas     : {breakeven_wr*100:.1f}%")
        print(f"- Threshold Minimum  : {threshold*100:.1f}%")
        print(f"- Ekspektasi per Trade: {expectancy_ratio:+.3f} ATR")
        
        if action != "HOLD" and latest_prob_success > threshold and expectancy_ratio > 0:
            print(f"\n[!] Sinyal {action} Lolos Gate (P={latest_prob_success*100:.1f}% > {threshold*100:.0f}%, E[R]={expectancy_ratio:+.3f} ATR).")
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
            
            # Ambil berita makro aktual secara dinamis via Yahoo Finance RSS
            fundamental_news = gatekeeper.fetch_live_news()
            
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
            elif expectancy_ratio <= 0:
                print(f"\n[-] Sinyal {action} DIBATALKAN: Ekspektasi negatif ({expectancy_ratio:+.3f} ATR) pada P={latest_prob_success*100:.1f}%.")
            else:
                print(f"\n[-] Sinyal {action} DIBATALKAN: P={latest_prob_success*100:.1f}% < threshold {threshold*100:.0f}% untuk {action} (impas={breakeven_wr*100:.1f}%).")
                print(f"    Ekspektasi meskipun positif ({expectancy_ratio:+.3f}) belum cukup aman.")


if __name__ == "__main__":
    # Jalankan pipeline
    pipeline = HybridMLPipeline(symbol="BTC/USDT")
    pipeline.run_end_to_end(strict_mode=False, backtest_mode=True)
