"""
UI Gradio untuk Pengujian Microsoft BEATs (Audio Tagging)
Fitur: Audio Tagging, Ekstraksi Fitur, Daftar & Pencarian Kelas AudioSet, Info Model
"""

import os
import sys
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from coba_beats import (
    BEATsHelper,
    CHECKPOINTS,
    daftar_kelas_audioset,
    TARGET_SR,
)

import gradio as gr

_helpers: dict = {}


def get_helper(checkpoint="finetuned") -> BEATsHelper:
    if checkpoint not in _helpers:
        _helpers[checkpoint] = BEATsHelper(checkpoint=checkpoint)
    return _helpers[checkpoint]


# =============================================================================
# HANDLERS
# =============================================================================

def handle_tagging(audio, top_k):
    if audio is None:
        return None, "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper("finetuned")
        results = h.tag_audio(audio, top_k=int(top_k))

        table = [[label, f"{prob:.2%}"] for label, prob in results]
        text = "\n".join(f"{prob:7.2%}  {label}" for label, prob in results)
        return table, text, f"Selesai. {len(results)} label teratas dari 527 kelas AudioSet."
    except Exception as e:
        return None, "", f"Error: {e}"


def handle_embedding(audio):
    if audio is None:
        return "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper("pretrained")
        feats = h.extract_embedding(audio)
        info = (
            f"Bentuk embedding : {tuple(feats.shape)}\n"
            f"  (time_steps x hidden_dim)\n\n"
            f"Rata-rata (mean) : {feats.mean().item():.6f}\n"
            f"Simpangan (std)  : {feats.std().item():.6f}\n"
            f"Nilai minimum    : {feats.min().item():.6f}\n"
            f"Nilai maksimum   : {feats.max().item():.6f}\n\n"
            f"8 nilai pertama  :\n{feats.flatten()[:8].tolist()}"
        )
        return info, f"Ekstraksi fitur berhasil. Shape: {tuple(feats.shape)}"
    except Exception as e:
        return "", f"Error: {e}"


def handle_search_class(keyword):
    rows = daftar_kelas_audioset()
    if not keyword or not keyword.strip():
        return [[idx, name] for idx, mid, name in rows[:50]]
    kw = keyword.strip().lower()
    hits = [[idx, name] for idx, mid, name in rows if kw in name.lower()]
    return hits if hits else [["-", f"Tidak ada kelas yang cocok dengan '{keyword}'"]]


def handle_model_info(checkpoint):
    try:
        h = get_helper(checkpoint)
        info = h.model_info()
        return "\n".join(f"{k:<20}: {v}" for k, v in info.items())
    except Exception as e:
        return f"Error: {e}"


def get_all_classes():
    return [[idx, name] for idx, mid, name in daftar_kelas_audioset()]


# =============================================================================
# UI BUILDER
# =============================================================================
def create_ui():
    with gr.Blocks(title="BEATs - Uji Coba", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Microsoft BEATs - Pengujian Audio Tagging\n"
            "BEATs **mengenali** suara (bukan membuat suara). Diberi audio, "
            "model menebak label dari **527 kelas AudioSet** — misal: Speech, Dog, "
            "Guitar, Rain, Car. Audio otomatis dikonversi ke "
            f"**{TARGET_SR} Hz mono**."
        )

        with gr.Tab("Audio Tagging"):
            gr.Markdown(
                "### Kenali suara dalam file audio\n"
                "Upload file audio atau rekam dari mikrofon, lalu klik Analisis."
            )
            t_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"],
            )
            t_topk = gr.Slider(3, 30, value=10, step=1, label="Tampilkan Top-K Label")
            t_btn = gr.Button("Analisis Audio", variant="primary")
            t_table = gr.Dataframe(
                headers=["Label", "Probabilitas"],
                label="Hasil Tagging",
                interactive=False,
            )
            t_text = gr.Textbox(label="Hasil (teks)", lines=12, interactive=False)
            t_status = gr.Textbox(label="Status", interactive=False)
            t_btn.click(
                handle_tagging,
                inputs=[t_audio, t_topk],
                outputs=[t_table, t_text, t_status],
            )

        with gr.Tab("Ekstraksi Fitur"):
            gr.Markdown(
                "### Ambil embedding / representasi vektor dari audio\n"
                "Memakai checkpoint **pretrained** (bukan finetuned). "
                "Embedding ini berguna sebagai input untuk model machine learning lain."
            )
            e_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"],
            )
            e_btn = gr.Button("Ekstrak Fitur", variant="primary")
            e_info = gr.Textbox(label="Informasi Embedding", lines=12, interactive=False)
            e_status = gr.Textbox(label="Status", interactive=False)
            e_btn.click(handle_embedding, inputs=[e_audio], outputs=[e_info, e_status])

        with gr.Tab("Cari Kelas"):
            gr.Markdown(
                "### Cari kelas AudioSet berdasarkan kata kunci\n"
                "Contoh kata kunci: `dog`, `music`, `rain`, `speech`, `engine`."
            )
            s_kw = gr.Textbox(label="Kata Kunci", placeholder="misal: dog")
            s_btn = gr.Button("Cari", variant="primary")
            s_table = gr.Dataframe(
                headers=["Index", "Nama Kelas"],
                label="Hasil Pencarian",
                interactive=False,
            )
            s_btn.click(handle_search_class, inputs=[s_kw], outputs=[s_table])
            s_kw.submit(handle_search_class, inputs=[s_kw], outputs=[s_table])

        with gr.Tab("Semua Kelas AudioSet"):
            gr.Markdown("### 527 Kelas AudioSet yang Dikenali BEATs")
            gr.Dataframe(
                headers=["Index", "Nama Kelas"],
                value=get_all_classes(),
                interactive=False,
            )

        with gr.Tab("Info Model"):
            gr.Markdown("### Detail Arsitektur & Checkpoint")
            i_ckpt = gr.Dropdown(
                choices=list(CHECKPOINTS.keys()),
                value="finetuned",
                label="Checkpoint",
            )
            i_btn = gr.Button("Muat Info Model", variant="primary")
            i_out = gr.Textbox(label="Info Model", lines=12, interactive=False)
            i_btn.click(handle_model_info, inputs=[i_ckpt], outputs=[i_out])

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7865,
        share=False,
        inbrowser=True,
    )
