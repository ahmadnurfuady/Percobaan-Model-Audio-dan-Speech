"""
UI Gradio untuk Pengujian Model SeamlessM4T v2
Menjalankan semua fitur: T2TT, T2ST, S2TT, S2ST, Demo, Daftar Bahasa
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from coba_seamlessm4t import SeamlessM4THelper, POPULAR_LANGUAGES

import gradio as gr

helper = None

def get_helper():
    global helper
    if helper is None:
        helper = SeamlessM4THelper()
    return helper

LANG_CHOICES = [f"{code} - {name}" for code, name in POPULAR_LANGUAGES.items()]

def parse_lang_code(lang_str):
    return lang_str.split(" - ")[0].strip()


# =============================================================================
# HANDLER FUNCTIONS
# =============================================================================

def handle_t2tt(text, src_lang, tgt_lang):
    if not text or not text.strip():
        return "Masukkan teks terlebih dahulu."
    src = parse_lang_code(src_lang)
    tgt = parse_lang_code(tgt_lang)
    try:
        h = get_helper()
        result = h.text_to_text(text.strip(), src_lang=src, tgt_lang=tgt)
        return result
    except Exception as e:
        return f"Error: {e}"


def handle_t2st(text, src_lang, tgt_lang):
    if not text or not text.strip():
        return None, "Masukkan teks terlebih dahulu."
    src = parse_lang_code(src_lang)
    tgt = parse_lang_code(tgt_lang)
    try:
        h = get_helper()
        out_path = os.path.join(tempfile.gettempdir(), "seamless_t2st_output.wav")
        h.text_to_speech(text.strip(), src_lang=src, tgt_lang=tgt, output_wav=out_path)
        return out_path, f"Audio berhasil dihasilkan ({src} -> {tgt})"
    except Exception as e:
        return None, f"Error: {e}"


def handle_s2tt(audio, tgt_lang):
    if audio is None:
        return "Upload atau rekam audio terlebih dahulu."
    tgt = parse_lang_code(tgt_lang)
    try:
        h = get_helper()
        if isinstance(audio, tuple):
            sr, data = audio
            import scipy.io.wavfile as wavfile
            tmp_path = os.path.join(tempfile.gettempdir(), "seamless_s2tt_input.wav")
            if data.dtype != np.int16 and data.dtype != np.float32:
                data = data.astype(np.float32)
            wavfile.write(tmp_path, rate=sr, data=data)
            audio_path = tmp_path
        else:
            audio_path = audio
        result = h.speech_to_text(audio_path, tgt_lang=tgt)
        return result if result else "(Tidak ada hasil transkripsi)"
    except Exception as e:
        return f"Error: {e}"


def handle_s2st(audio, tgt_lang):
    if audio is None:
        return None, "Upload atau rekam audio terlebih dahulu."
    tgt = parse_lang_code(tgt_lang)
    try:
        h = get_helper()
        if isinstance(audio, tuple):
            sr, data = audio
            import scipy.io.wavfile as wavfile
            tmp_path = os.path.join(tempfile.gettempdir(), "seamless_s2st_input.wav")
            if data.dtype != np.int16 and data.dtype != np.float32:
                data = data.astype(np.float32)
            wavfile.write(tmp_path, rate=sr, data=data)
            audio_path = tmp_path
        else:
            audio_path = audio
        out_path = os.path.join(tempfile.gettempdir(), "seamless_s2st_output.wav")
        h.speech_to_speech(audio_path, tgt_lang=tgt, output_wav=out_path)
        return out_path, f"Audio terjemahan berhasil dihasilkan (-> {tgt})"
    except Exception as e:
        return None, f"Error: {e}"


def handle_demo():
    try:
        h = get_helper()
        sample = "Halo, selamat datang! Ini adalah pengujian model SeamlessM4T."
        lines = [f"Teks Uji: \"{sample}\"\n"]

        lines.append("[1/3] Text-to-Text (ind -> eng)...")
        r1 = h.text_to_text(sample, src_lang="ind", tgt_lang="eng")
        lines.append(f"  Hasil: {r1}\n")

        lines.append("[2/3] Text-to-Text (ind -> jpn)...")
        r2 = h.text_to_text(sample, src_lang="ind", tgt_lang="jpn")
        lines.append(f"  Hasil: {r2}\n")

        lines.append("[3/3] Text-to-Speech (ind -> eng)...")
        out_path = os.path.join(tempfile.gettempdir(), "seamless_demo_t2st.wav")
        h.text_to_speech(sample, src_lang="ind", tgt_lang="eng", output_wav=out_path)
        lines.append(f"  Audio disimpan: {out_path}\n")

        lines.append("DEMO SELESAI DENGAN SUKSES!")
        return "\n".join(lines), out_path
    except Exception as e:
        return f"Error saat demo: {e}", None


def get_lang_table():
    rows = []
    for code, name in POPULAR_LANGUAGES.items():
        rows.append([code, name])
    return rows


# =============================================================================
# BUILD UI
# =============================================================================

def create_ui():
    with gr.Blocks(
        title="SeamlessM4T v2 - Uji Coba",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# SeamlessM4T v2 - Pengujian Model\n"
            "Antarmuka untuk mencoba semua fitur model Meta SeamlessM4T v2 (terjemahan teks & suara)."
        )

        with gr.Tab("Text-to-Text (T2TT)"):
            gr.Markdown("### Terjemah Teks antar Bahasa")
            with gr.Row():
                t2tt_src = gr.Dropdown(
                    choices=LANG_CHOICES, value="ind - Bahasa Indonesia",
                    label="Bahasa Sumber"
                )
                t2tt_tgt = gr.Dropdown(
                    choices=LANG_CHOICES, value="eng - English (Inggris)",
                    label="Bahasa Tujuan"
                )
            t2tt_input = gr.Textbox(
                label="Teks Input",
                placeholder="Ketik teks yang ingin diterjemahkan...",
                lines=3
            )
            t2tt_btn = gr.Button("Terjemahkan", variant="primary")
            t2tt_output = gr.Textbox(label="Hasil Terjemahan", lines=3, interactive=False)
            t2tt_btn.click(handle_t2tt, inputs=[t2tt_input, t2tt_src, t2tt_tgt], outputs=t2tt_output)

        with gr.Tab("Text-to-Speech (T2ST)"):
            gr.Markdown("### Teks ke Suara (.wav)")
            with gr.Row():
                t2st_src = gr.Dropdown(
                    choices=LANG_CHOICES, value="ind - Bahasa Indonesia",
                    label="Bahasa Teks Asal"
                )
                t2st_tgt = gr.Dropdown(
                    choices=LANG_CHOICES, value="eng - English (Inggris)",
                    label="Bahasa Suara Output"
                )
            t2st_input = gr.Textbox(
                label="Teks Input",
                placeholder="Ketik teks untuk diubah menjadi suara...",
                lines=3
            )
            t2st_btn = gr.Button("Hasilkan Suara", variant="primary")
            t2st_audio = gr.Audio(label="Audio Hasil", type="filepath")
            t2st_status = gr.Textbox(label="Status", interactive=False)
            t2st_btn.click(handle_t2st, inputs=[t2st_input, t2st_src, t2st_tgt], outputs=[t2st_audio, t2st_status])

        with gr.Tab("Speech-to-Text (S2TT)"):
            gr.Markdown("### Audio ke Teks (Transkripsi / Terjemahan)")
            s2tt_audio = gr.Audio(
                label="Upload atau Rekam Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            s2tt_tgt = gr.Dropdown(
                choices=LANG_CHOICES, value="eng - English (Inggris)",
                label="Bahasa Teks Output"
            )
            s2tt_btn = gr.Button("Transkripsi / Terjemahkan", variant="primary")
            s2tt_output = gr.Textbox(label="Hasil Transkripsi", lines=3, interactive=False)
            s2tt_btn.click(handle_s2tt, inputs=[s2tt_audio, s2tt_tgt], outputs=s2tt_output)

        with gr.Tab("Speech-to-Speech (S2ST)"):
            gr.Markdown("### Terjemahan Suara ke Suara")
            s2st_audio_in = gr.Audio(
                label="Upload atau Rekam Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            s2st_tgt = gr.Dropdown(
                choices=LANG_CHOICES, value="eng - English (Inggris)",
                label="Bahasa Audio Tujuan"
            )
            s2st_btn = gr.Button("Terjemahkan Audio", variant="primary")
            s2st_audio_out = gr.Audio(label="Audio Hasil Terjemahan", type="filepath")
            s2st_status = gr.Textbox(label="Status", interactive=False)
            s2st_btn.click(handle_s2st, inputs=[s2st_audio_in, s2st_tgt], outputs=[s2st_audio_out, s2st_status])

        with gr.Tab("Demo Otomatis"):
            gr.Markdown(
                "### Jalankan Demo Otomatis\n"
                "Menguji T2TT (ind->eng, ind->jpn) dan T2ST (ind->eng) secara otomatis."
            )
            demo_btn = gr.Button("Jalankan Demo", variant="primary")
            demo_log = gr.Textbox(label="Log Demo", lines=12, interactive=False)
            demo_audio = gr.Audio(label="Audio Demo (T2ST)", type="filepath")
            demo_btn.click(handle_demo, inputs=[], outputs=[demo_log, demo_audio])

        with gr.Tab("Daftar Bahasa"):
            gr.Markdown("### Daftar Kode Bahasa yang Didukung")
            gr.Dataframe(
                headers=["Kode (3 huruf)", "Nama Bahasa"],
                value=get_lang_table(),
                interactive=False
            )
            gr.Markdown("*Catatan: Bahasa Jawa ('jav') didukung untuk teks, tidak untuk speech.*")

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
