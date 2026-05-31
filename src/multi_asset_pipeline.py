"""
multi_asset_pipeline.py
Orkestrator Utama Portofolio Multi-Asset (BTC/USDT, EUR/USDT, XAU/USDT)
Menggunakan Alokasi Bobot Profesional Risk Parity (Inverse Volatility)
dan Kelly Position Sizing Terintegrasi di VectorBT.
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings

# Daftarkan direktori root dan src ke sys.path secara dinamis
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)
sys.path.append(os.path.dirname(curr_dir))

# Impor pipeline kuantitatif kita
from pipeline import HybridMLPipeline

def run_multi_asset_portfolio():
    print("=" * 80)
    print("      MEMULAI ORKESTRASI PORTOFOLIO MULTI-ASSET DENGAN RISK PARITY")
    print("=" * 80)
    
    symbols = ["BTC/USDT", "EUR/USDT", "XAU/USDT"]
    results = {}
    volatilities = {}
    
    # 1. Jalankan Pipeline untuk masing-masing aset secara terpisah
    for symbol in symbols:
        print(f"\n[MULTI-ASSET] Menjalankan pipeline kuantitatif untuk {symbol}...")
        pipeline = HybridMLPipeline(symbol=symbol)
        
        # Jalankan secara OOS (backtest_mode=True)
        res = pipeline.run_end_to_end(strict_mode=False, backtest_mode=True)
        
        if res and res["close_test"] is not None and len(res["close_test"]) > 0:
            results[symbol] = res
            
            # Hitung volatilitas historis out-of-sample dari log-return
            close = res["close_test"]
            log_returns = np.log(close / close.shift(1)).fillna(0.0)
            vol = log_returns.std()
            volatilities[symbol] = vol if vol > 0 else 1e-6
            print(f"[MULTI-ASSET] Volatilitas OOS {symbol}: {vol*100:.4f}% per bar.")
        else:
            print(f"[WARNING] Gagal mengambil data OOS untuk {symbol}. Aset dilewati.")
            
    if len(results) < 2:
        print("[ERROR] Data aset yang berhasil dimuat kurang dari 2. Alokasi portofolio dibatalkan.")
        return
        
    print("\n" + "=" * 50)
    print("         PERHITUNGAN BOBOT ALOKASI RISK PARITY")
    print("=" * 50)
    
    # 2. Hitung Bobot Risk Parity (Inverse Volatility: w_i = (1/sigma_i) / sum(1/sigma_j))
    inv_vols = {sym: 1.0 / vol for sym, vol in volatilities.items()}
    sum_inv_vols = sum(inv_vols.values())
    weights = {sym: inv_vol / sum_inv_vols for sym, inv_vol in inv_vols.items()}
    
    for sym in symbols:
        print(f" - {sym:<10} : Vol={volatilities[sym]*100:.4f}% | Bobot Alokasi={weights[sym]*100:.2f}%")
    print("=" * 50 + "\n")
    
    # 3. Selaraskan indeks waktu antar aset untuk Backtest Portofolio Terpadu
    # Temukan irisan waktu terpadu (common time index) agar sinkron
    common_index = None
    for sym, res in results.items():
        if common_index is None:
            common_index = res["close_test"].index
        else:
            common_index = common_index.intersection(res["close_test"].index)
            
    print(f"[MULTI-ASSET] Jumlah bar waktu sinkron out-of-sample: {len(common_index)}")
    if len(common_index) == 0:
        print("[ERROR] Indeks waktu antar aset tidak beririsan. Backtest portofolio dibatalkan.")
        return
        
    # Bangun DataFrame harga dan sinyal yang sinkron
    close_df = pd.DataFrame(index=common_index)
    entries_df = pd.DataFrame(index=common_index)
    short_entries_df = pd.DataFrame(index=common_index)
    sizes_df = pd.DataFrame(index=common_index)
    
    for sym in symbols:
        res = results[sym]
        close_df[sym] = res["close_test"].loc[common_index]
        entries_df[sym] = res["entries"].loc[common_index]
        short_entries_df[sym] = res["short_entries"].loc[common_index]
        
        # Lot sizing gabungan: Bobot Risk Parity dikalikan lot sizing Kelly internal
        # Ini menyetarakan resiko Kelly global antar instrumen
        raw_sizes = res["kelly_sizes"]
        # Ubah raw_sizes (numpy array) menjadi pandas Series agar aman dilokalisasi indeksnya
        sizes_series = pd.Series(raw_sizes, index=res["close_test"].index).loc[common_index]
        sizes_df[sym] = sizes_series * weights[sym]
        
    # 4. Eksekusi Backtest Portofolio Gabungan Terpadu via VectorBT
    print("[MULTI-ASSET] Memulai Simulasi Backtesting Portofolio Multi-Asset di VectorBT...")
    try:
        import vectorbt as vbt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Alokasikan modal awal $10,000 proporsional berdasarkan bobot Risk Parity
        init_cash_list = [10000.0 * weights[sym] for sym in symbols]
        
        portfolio = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=short_entries_df,
            size=sizes_df,
            size_type="percent",
            fees=0.0006, # 0.06% spot fee
            slippage=0.0001, # 0.01% slippage
            freq="15min",
            init_cash=init_cash_list
        )
        
        # Hitung Equity Curve masing-masing aset dan portofolio gabungan total
        individual_equities = portfolio.value()
        total_portfolio_equity = individual_equities.sum(axis=1)
        
        # Ekstrak Metrik Portofolio Gabungan
        final_value = total_portfolio_equity.iloc[-1]
        total_return = ((final_value - 10000.0) / 10000.0) * 100
        
        # Hitung drawdown portofolio total
        rolling_max = total_portfolio_equity.cummax()
        drawdowns = (total_portfolio_equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Cetak Laporan Portofolio
        print("=" * 60)
        print("     LAPORAN KINERJA GABUNGAN PORTOFOLIO MULTI-ASSET     ")
        print("=" * 60)
        print(f" Simbol Terbaca        : {', '.join(symbols)}")
        print(f" Modal Awal Global     : $10,000.00")
        print(f" Nilai Akhir Portofolio: ${final_value:,.2f}")
        print(f" Total Return          : {total_return:+.2f}%")
        print(f" Max Drawdown          : {max_drawdown:.2f}%")
        print("=" * 60)
        
        # 5. Gambarkan Grafik Perbandingan Premium & Simpan
        plt.figure(figsize=(12, 6))
        
        # Plot perkembangan ekuitas individu (diskala ke $10,000 agar mudah dibandingkan)
        for sym in symbols:
            norm_equity = (individual_equities[sym] / init_cash_list[symbols.index(sym)]) * 10000.0
            plt.plot(norm_equity, label=f"{sym} (Risk-Scaled Baseline)", alpha=0.6, linestyle="--")
            
        # Plot ekuitas portofolio gabungan total
        plt.plot(total_portfolio_equity, label="PORTFOLIO TOTAL (Risk Parity)", color="#00ffcc", linewidth=2.5)
        
        plt.title("Perbandingan Perkembangan Ekuitas Individu vs. Portofolio Risk Parity Multi-Asset", fontsize=14, color="white", weight="bold")
        plt.xlabel("Waktu", fontsize=11, color="white")
        plt.ylabel("Ekuitas Saldo (USD)", fontsize=11, color="white")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(facecolor="#1e1e1e", edgecolor="#00ffcc", labelcolor="white")
        
        # Estetika Dark Mode Premium
        fig = plt.gcf()
        fig.patch.set_facecolor('#121212')
        ax = plt.gca()
        ax.set_facecolor('#121212')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(colors='white')
        
        plt.savefig("multi_asset_equity_curve.png", dpi=150, bbox_inches="tight", facecolor='#121212')
        plt.close()
        print("\n[MULTI-ASSET] Grafik Multi-Asset Equity Curve berhasil disimpan ke 'multi_asset_equity_curve.png'.")
        
    except Exception as e:
        print(f"[ERROR] Gagal menjalankan backtest VectorBT Portofolio: {e}")
        import traceback
        traceback.print_exc()
        
    # 6. Tampilkan status Sinyal Live Terakhir untuk Ketiga Aset
    print("\n" + "=" * 50)
    print("      STATUS SINYAL LIVE TERAKHIR (DENGAN LLM GATE)")
    print("=" * 50)
    for sym in symbols:
        if sym in results:
            res = results[sym]
            print(f" - {sym:<10} pada {res['latest_idx']} : ML Sinyal={res['latest_action']:<5} | Prob Sukses={res['latest_prob_success']*100:.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_multi_asset_portfolio()
