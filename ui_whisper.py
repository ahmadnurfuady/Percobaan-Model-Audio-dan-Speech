"""
UI Gradio untuk Pengujian OpenAI Whisper
Fitur: Transkripsi, Terjemahan, Deteksi Bahasa, Timestamp
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

from coba_whisper import (
    WhisperHelper,
    POPULAR_LANGUAGES,
    WHISPER_MODELS,
    WHISPER_MODEL_SIZES,
)

import gradio as gr

# Cache helper per model name
_helpers: dict[str, WhisperHelper] = {}

def get_helper(model_name="base") -> WhisperHelper:
    if model_name not in _helpers:
        _helpers[model_name] = WhisperHelper(model_name=model_name)
    return _helpers[model_name]


LANG_CHOICES = ["(auto)"] + [f"{code} - {name}" for code, name in POPULAR_LANGUAGES.items()]

def parse_lang(s: str):
    if not s or s == "(auto)":
        return None
    return s.split(" - ")[0].strip()


# =============================================================================
# HANDLERS
# =============================================================================

def handle_transcribe(audio, language, model_name):
    if audio is None:
        return "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper(model_name)
        lang = parse_lang(language)
        result = h.transcribe(audio, language=lang)
        detected = result.get("language", "?")
        status = f"Selesai. Bahasa terdeteksi: {detected} | Model: whisper/{model_name}"
        return result["text"].strip(), status
    except Exception as e:
        return "", f"Error: {e}"


def handle_translate(audio, model_name):
    if audio is None:
        return "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper(model_name)
        result = h.translate_to_english(audio)
        detected = result.get("language", "?")
        status = f"Terjemahan selesai. Bahasa sumber: {detected} | Model: whisper/{model_name}"
        return result["text"].strip(), status
    except Exception as e:
        return "", f"Error: {e}"


def handle_detect(audio, model_name):
    if audio is None:
        return "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper(model_name)
        top = h.detect_language(audio)
        lines = []
        for lang, prob in top:
            bar = "█" * int(prob * 40)
            lang_name = POPULAR_LANGUAGES.get(lang, lang)
            lines.append(f"{lang:<6} {prob:.1%}  {bar}  ({lang_name})")
        return "\n".join(lines), f"Deteksi selesai. Model: whisper/{model_name}"
    except Exception as e:
        return "", f"Error: {e}"


def handle_timestamp(audio, language, model_name):
    if audio is None:
        return "", "", "Upload atau rekam audio terlebih dahulu."
    try:
        h = get_helper(model_name)
        lang = parse_lang(language)
        text, segments = h.transcribe_with_timestamps(audio, language=lang)

        seg_lines = []
        for seg in segments:
            t0 = f"{seg['start']:.2f}s"
            t1 = f"{seg['end']:.2f}s"
            seg_lines.append(f"[{t0} → {t1}]  {seg['text'].strip()}")

        detected = h._model and "?"
        status = f"{len(segments)} segmen | Model: whisper/{model_name}"
        return text.strip(), "\n".join(seg_lines), status
    except Exception as e:
        return "", "", f"Error: {e}"


def get_model_table():
    return [[m, WHISPER_MODEL_SIZES[m]] for m in WHISPER_MODELS]


def get_lang_table():
    return [[code, name] for code, name in POPULAR_LANGUAGES.items()]


# =============================================================================
# UI BUILDER
# =============================================================================
def create_ui():
    model_choices = WHISPER_MODELS
    default_model = "base"

    with gr.Blocks(title="OpenAI Whisper - Uji Coba", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# OpenAI Whisper - Pengujian ASR\n"
            "Transkripsi & terjemahan audio ke teks. Mendukung **99 bahasa** termasuk **Bahasa Indonesia**."
        )

        with gr.Tab("Transkripsi"):
            gr.Markdown(
                "### Transkripsi Audio → Teks\n"
                "Upload file audio atau rekam langsung dari mikrofon."
            )
            with gr.Row():
                tr_model = gr.Dropdown(
                    choices=model_choices, value=default_model, label="Model Whisper"
                )
                tr_lang = gr.Dropdown(
                    choices=LANG_CHOICES, value="(auto)", label="Bahasa (auto = deteksi otomatis)"
                )
            tr_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            tr_btn = gr.Button("Transkripsi", variant="primary")
            tr_text = gr.Textbox(label="Hasil Transkripsi", lines=6, interactive=False)
            tr_status = gr.Textbox(label="Status", interactive=False)
            tr_btn.click(
                handle_transcribe,
                inputs=[tr_audio, tr_lang, tr_model],
                outputs=[tr_text, tr_status]
            )

        with gr.Tab("Terjemahan ke Inggris"):
            gr.Markdown(
                "### Terjemahkan Audio → Bahasa Inggris\n"
                "Whisper akan transkripsi dan langsung menerjemahkan ke Inggris."
            )
            with gr.Row():
                tl_model = gr.Dropdown(
                    choices=model_choices, value=default_model, label="Model Whisper"
                )
            tl_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            tl_btn = gr.Button("Terjemahkan ke Inggris", variant="primary")
            tl_text = gr.Textbox(label="Hasil Terjemahan (Inggris)", lines=6, interactive=False)
            tl_status = gr.Textbox(label="Status", interactive=False)
            tl_btn.click(
                handle_translate,
                inputs=[tl_audio, tl_model],
                outputs=[tl_text, tl_status]
            )

        with gr.Tab("Deteksi Bahasa"):
            gr.Markdown(
                "### Deteksi Bahasa dari Audio\n"
                "Menampilkan probabilitas 5 bahasa teratas yang terdeteksi."
            )
            with gr.Row():
                dl_model = gr.Dropdown(
                    choices=model_choices, value=default_model, label="Model Whisper"
                )
            dl_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            dl_btn = gr.Button("Deteksi Bahasa", variant="primary")
            dl_text = gr.Textbox(label="Hasil Deteksi", lines=8, interactive=False)
            dl_status = gr.Textbox(label="Status", interactive=False)
            dl_btn.click(
                handle_detect,
                inputs=[dl_audio, dl_model],
                outputs=[dl_text, dl_status]
            )

        with gr.Tab("Transkripsi + Timestamp"):
            gr.Markdown(
                "### Transkripsi dengan Timestamp per Segmen\n"
                "Tampilkan waktu mulai dan akhir untuk setiap kalimat."
            )
            with gr.Row():
                ts_model = gr.Dropdown(
                    choices=model_choices, value=default_model, label="Model Whisper"
                )
                ts_lang = gr.Dropdown(
                    choices=LANG_CHOICES, value="(auto)", label="Bahasa"
                )
            ts_audio = gr.Audio(
                label="Audio Input",
                type="filepath",
                sources=["upload", "microphone"]
            )
            ts_btn = gr.Button("Transkripsi + Timestamp", variant="primary")
            ts_full = gr.Textbox(label="Teks Lengkap", lines=4, interactive=False)
            ts_segs = gr.Textbox(label="Segmen dengan Timestamp", lines=12, interactive=False)
            ts_status = gr.Textbox(label="Status", interactive=False)
            ts_btn.click(
                handle_timestamp,
                inputs=[ts_audio, ts_lang, ts_model],
                outputs=[ts_full, ts_segs, ts_status]
            )

        with gr.Tab("Daftar Bahasa"):
            gr.Markdown("### Bahasa Populer yang Didukung Whisper (total 99 bahasa)")
            gr.Dataframe(
                headers=["Kode", "Nama Bahasa"],
                value=get_lang_table(),
                interactive=False
            )

        with gr.Tab("Daftar Model"):
            gr.Markdown("### Perbandingan Model Whisper")
            gr.Dataframe(
                headers=["Model", "Ukuran File"],
                value=get_model_table(),
                interactive=False
            )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7862,
        share=False,
        inbrowser=True
    )
