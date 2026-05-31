"""
test_chronos_integration.py
Script verifikasi integrasi end-to-end untuk sistem trading hybrid (SMC + Chronos + LLM).
Menjalankan pipeline pemrosesan data, pembuatan fitur probabilistik Chronos,
pelabelan Triple-Barrier, pelatihan dual XGBoost, dan pengambilan keputusan LLM Gatekeeper.
"""

import sys
import os

# Tambahkan folder 'src' ke sys.path agar impor modul internal lancar
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipeline import HybridMLPipeline
    print("=====================================================================")
    print("   SISTEM PREDIKSI MARKET HYBRID (SMC + CHRONOS + LLM) - VERIFIKASI   ")
    print("=====================================================================")
    
    # Inisialisasi pipeline hibrida
    # Kita gunakan model 'amazon/chronos-t5-tiny' yang terbukti sangat ringan & cepat
    pipeline = HybridMLPipeline(symbol="BTC/USDT", chronos_model="amazon/chronos-t5-tiny")
    
    # Jalankan pipeline dari awal hingga pengambilan keputusan LLM
    pipeline.run_end_to_end()
    
    print("\n[SUKSES] Seluruh modul terintegrasi sempurna tanpa error!")
    print("SMC Features -> Chronos forecasting -> Triple-Barrier -> Dual XGBoost -> LLM reasoning.")
    print("=====================================================================")

except Exception as e:
    print("\n[ERROR] Terjadi kegagalan saat menjalankan verifikasi integrasi:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
