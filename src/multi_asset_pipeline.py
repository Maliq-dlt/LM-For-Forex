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

    # Hanya proses aset yang benar-benar berhasil dimuat (cegah KeyError saat ada aset di-skip)
    active = [s for s in symbols if s in results]

    print("\n" + "=" * 50)
    print("         PERHITUNGAN BOBOT ALOKASI RISK PARITY")
    print("=" * 50)
    
    # 2. Hitung Bobot Risk Parity (Inverse Volatility: w_i = (1/sigma_i) / sum(1/sigma_j))
    inv_vols = {sym: 1.0 / vol for sym, vol in volatilities.items()}
    sum_inv_vols = sum(inv_vols.values())
    weights = {sym: inv_vol / sum_inv_vols for sym, inv_vol in inv_vols.items()}
    
    for sym in active:
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
    tp_stops_df = pd.DataFrame(index=common_index)
    sl_stops_df = pd.DataFrame(index=common_index)
    
    for sym in active:
        res = results[sym]
        close_df[sym] = res["close_test"].loc[common_index]
        entries_df[sym] = res["entries"].loc[common_index]
        short_entries_df[sym] = res["short_entries"].loc[common_index]
        
        # Lot sizing = Kelly fraction internal per-aset.
        # Bobot Risk Parity diterapkan SATU KALI saja, lewat alokasi init_cash di bawah,
        # bukan dikalikan lagi ke size (mencegah double-counting tilt).
        raw_sizes = res["kelly_sizes"]
        sizes_series = pd.Series(raw_sizes, index=res["close_test"].index).loc[common_index]
        sizes_df[sym] = sizes_series
        
        # Penyelarasan Bracket Exits ATR stops untuk masing-masing aset
        tp_stops_df[sym] = pd.Series(res["tp_stops"], index=res["close_test"].index).loc[common_index]
        sl_stops_df[sym] = pd.Series(res["sl_stops"], index=res["close_test"].index).loc[common_index]
        
    # 4. Eksekusi Backtest Portofolio Gabungan Terpadu via VectorBT
    print("[MULTI-ASSET] Memulai Simulasi Backtesting Portofolio Multi-Asset di VectorBT...")
    try:
        import vectorbt as vbt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Alokasikan modal awal $10,000 proporsional berdasarkan bobot Risk Parity.
        # Urutan HARUS mengikuti kolom close_df agar selaras dengan vectorbt.
        init_cash_list = [10000.0 * weights[sym] for sym in close_df.columns]
        
        # Coba jalankan simulasi Dual-Sided (Long & Short) secara terintegrasi dengan Bracket ATR Stops
        try:
            portfolio = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries_df,
                exits=short_entries_df | entries_df.shift(40).fillna(False),       # auto-exit 40 bar
                short_entries=short_entries_df,
                short_exits=entries_df | short_entries_df.shift(40).fillna(False),  # auto-exit 40 bar
                size=sizes_df,
                size_type="percent",
                tp_stop=tp_stops_df,
                sl_stop=sl_stops_df,
                fees=0.0006, # 0.06% spot fee
                slippage=0.0001, # 0.01% slippage
                freq="15min",
                init_cash=init_cash_list
            )
            print("[MULTI-ASSET] Berhasil menjalankan simulasi Dual-Sided (Long & Short) Portofolio!")
        except Exception as e:
            print(f"[MULTI-ASSET] Gagal membuat portfolio Dual-Sided ({e}). Menggunakan fallback Long-Only secara aman...")
            portfolio = vbt.Portfolio.from_signals(
                close=close_df,
                entries=entries_df,
                exits=short_entries_df | entries_df.shift(40).fillna(False),
                size=sizes_df,
                size_type="percent",
                tp_stop=tp_stops_df,
                sl_stop=sl_stops_df,
                fees=0.0006,
                slippage=0.0001,
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
        
        # --- BUY-AND-HOLD PASIF BENCHMARK PORTFOLIO ---
        portfolio_bh = vbt.Portfolio.from_holding(
            close=close_df,
            init_cash=init_cash_list,
            fees=0.0006,
            slippage=0.0001,
            freq="15min"
        )
        total_bh_equity = portfolio_bh.value().sum(axis=1)
        bh_final_value = total_bh_equity.iloc[-1]
        bh_return = ((bh_final_value - 10000.0) / 10000.0) * 100
        
        # Drawdown B&H
        bh_rolling_max = total_bh_equity.cummax()
        bh_drawdowns = (total_bh_equity - bh_rolling_max) / bh_rolling_max
        bh_max_drawdown = bh_drawdowns.min() * 100
        
        # Cetak Laporan Portofolio
        print("=" * 60)
        print("     LAPORAN KINERJA GABUNGAN PORTOFOLIO MULTI-ASSET     ")
        print("=" * 60)
        print(f" Simbol Terbaca        : {', '.join(active)}")
        print(f" Modal Awal Global     : $10,000.00")
        print(f" Nilai Akhir Portofolio: ${final_value:,.2f} vs. B&H B-mark: ${bh_final_value:,.2f}")
        print(f" Total Return          : {total_return:+.2f}% vs. B&H B-mark: {bh_return:+.2f}%")
        print(f" Max Drawdown          : {max_drawdown:.2f}% vs. B&H B-mark: {bh_max_drawdown:.2f}%")
        print("=" * 60)
        
        # 5. Gambarkan Grafik Perbandingan Premium & Simpan
        plt.figure(figsize=(12, 6))
        
        # Plot perkembangan ekuitas individu (diskala ke $10,000 agar mudah dibandingkan)
        for sym in active:
            norm_equity = (individual_equities[sym] / (10000.0 * weights[sym])) * 10000.0
            plt.plot(norm_equity, label=f"{sym} (Risk-Scaled Baseline)", alpha=0.4, linestyle="--")
            
        # Plot ekuitas portofolio gabungan total (Risk Parity Strategy)
        plt.plot(total_portfolio_equity, label="PORTFOLIO TOTAL (Risk Parity Strategy)", color="#00ffcc", linewidth=2.5)
        
        # Plot ekuitas portofolio benchmark pasif Buy-and-Hold
        plt.plot(total_bh_equity, label="PORTFOLIO B-MARK (Passive Buy-and-Hold)", color="#ff5555", linewidth=1.8, linestyle="-.")
        
        plt.title("Perbandingan Perkembangan Ekuitas: Risk Parity vs. Buy-and-Hold Pasif", fontsize=14, color="white", weight="bold")
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
    for sym in active:
        if sym in results:
            res = results[sym]
            print(f" - {sym:<10} pada {res['latest_idx']} : ML Sinyal={res['latest_action']:<5} | Prob Sukses={res['latest_prob_success']*100:.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_multi_asset_portfolio()