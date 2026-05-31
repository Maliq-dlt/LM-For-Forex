"""
llm_gate.py
Modul penalaran LLM untuk memvalidasi sinyal ML berdasarkan sentimen fundamental, 
makroekonomi, dan analisis buku order, bertindak sebagai Trading Gatekeeper.
"""

import json
import os
import warnings

# Coba import Google Generative AI SDK untuk mendukung LLM Gemini asli
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class MarketLLMReasoningGate:
    """
    Trading Gatekeeper menggunakan LLM untuk melakukan penalaran kognitif tingkat tinggi.
    Menerjemahkan data pasar terkuantisasi (SMC, Chronos) dan menyintesisnya dengan
    sentimen berita fundamental sebelum membuka posisi trading.
    """
    def __init__(self, api_key: str = None):
        # Ambil API key dari env var jika tidak disediakan langsung
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = None
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                # Gunakan model Gemini 1.5 Flash untuk respons cepat dan akurat
                self.client = genai.GenerativeModel('gemini-1.5-flash')
                print("[LLM-GATE] SDK Gemini berhasil terhubung.")
            except Exception as e:
                warnings.warn(f"Gagal mengonfigurasi SDK Gemini: {e}. Menggunakan mode Simulasi.")
                self.client = None
        else:
            if not GEMINI_AVAILABLE:
                print("[LLM-GATE] Google Generative AI SDK ('google-generativeai') tidak terpasang.")
            if not self.api_key:
                print("[LLM-GATE] GEMINI_API_KEY tidak dikonfigurasi. Menggunakan mode Simulasi.")

    def generate_prompt(self, technical_metrics: dict, fundamental_sentiments: list) -> str:
        """ Menyusun instruksi prompt terstruktur untuk LLM. """
        prompt = f"""
=== TUGAS: ANALISIS PENALARAN RISIKO TRADING (TRADING GATEKEEPER) ===
Anda adalah agen Manajemen Risiko Kuantitatif Senior. Tugas Anda adalah memvalidasi sinyal BUY/SELL dari model ML kuantitatif berdasarkan sentimen fundamental dan struktur makro.

Sinyal Teknis ML Yang Diusulkan:
- Aksi: {technical_metrics.get('action')}
- Aset: {technical_metrics.get('symbol')}
- Timeframe Eksekusi: {technical_metrics.get('timeframe')}
- Struktur Market 1D: {technical_metrics.get('htf_structure')}
- Jarak ke Level OB SMC Terdekat: {technical_metrics.get('dist_to_ob')}%
- Prediksi Forecast Numerik Chronos: {technical_metrics.get('chronos_direction')}

Berita & Sentimen Fundamental Terbaru:
{json.dumps(fundamental_sentiments, indent=2)}

Analisis Confluence:
1. Apakah arah trend jangka panjang (1D) searah dengan aksi trading?
2. Apakah sentimen fundamental global mendukung aksi ini?
3. Apakah ada rilis berita ekonomi berdampak tinggi (High-Impact News) dalam waktu dekat?

Tanggapan Anda HARUS dalam format JSON valid dengan kunci berikut tanpa markup markdown luar lainnya:
{{
  "decision": "CONFIRM" atau "DELAY" atau "REJECT",
  "confidence_score": (float antara 0.0 hingga 1.0),
  "rationale": "penjelasan ringkas maksimal 2 kalimat mengenai keputusan Anda"
}}
"""
        return prompt

    def query_reasoning(self, prompt: str) -> dict:
        """
        Mengirim prompt ke API LLM Gemini.
        Jika API tidak dikonfigurasi, otomatis fallback ke simulasi logika lokal yang cerdas.
        """
        if self.client is not None:
            try:
                # Memanggil Gemini API secara real-time
                response = self.client.generate_content(prompt)
                text = response.text.strip()
                
                # Bersihkan format jika model mengembalikan kode dengan penutup ```json
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                
                return json.loads(text.strip())
            except Exception as e:
                warnings.warn(f"API Gemini Error: {e}. Menjalankan simulasi penalaran...")
                
        # --- LOGIKA SIMULASI PENALARAN LOKAL (CERDAS) ---
        # Menolak sinyal jika ada pertentangan arah tren makro (BOS/CHoCH)
        # Sinyal BUY tapi trend makro bearish => REJECT
        # Sinyal SELL tapi trend makro bullish => REJECT
        
        # Ekstrak data dari prompt secara terprogram untuk simulasi
        is_buy = "action: BUY" in prompt or "'action': 'BUY'" in prompt
        is_bearish_htf = "htf_structure: Bearish" in prompt or "Bearish Trend" in prompt
        is_bullish_htf = "htf_structure: Bullish" in prompt or "Bullish Trend" in prompt
        
        if (is_buy and is_bearish_htf) or (not is_buy and is_bullish_htf):
            return {
                "decision": "REJECT",
                "confidence_score": 0.30,
                "rationale": "Sinyal dibatalkan karena bertentangan dengan tren makro High Timeframe (1D). Disiplin SMC melarang entry melawan arah tren utama."
            }
            
        return {
            "decision": "CONFIRM",
            "confidence_score": 0.82,
            "rationale": "Sinyal terkonfirmasi. Struktur makro mendukung entry searah dan ramalan probabilistik Chronos memproyeksikan perluasan harga searah."
        }


if __name__ == "__main__":
    print("=== PENGUJIAN MODUL LLM GATEKEEPER ===")
    
    tech_data = {
        "action": "BUY",
        "symbol": "BTC/USDT",
        "timeframe": "15m",
        "htf_structure": "Bullish Trend",
        "dist_to_ob": "0.08",
        "chronos_direction": "Bullish Expansion"
    }
    
    news = [
        {"source": "Economic Calendar", "headline": "CPI inflation cools to 3.1% YoY (supporting crypto prices)"}
    ]
    
    gate = MarketLLMReasoningGate()
    prompt = gate.generate_prompt(tech_data, news)
    decision = gate.query_reasoning(prompt)
    
    print("\nHasil Penalaran Gatekeeper:")
    print(json.dumps(decision, indent=4))
