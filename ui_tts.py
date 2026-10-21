"""
UI Gradio untuk Pengujian Coqui TTS
Fitur: Simple TTS, XTTS v2 Multilingual, Voice Cloning, Daftar Bahasa & Model
"""

import os
import sys
import tempfile
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_script_dir = str(Path(__file__).resolve().parent)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from coba_tts import (
    TTSHelper,
    XTTS_LANGUAGES,
    MODEL_XTTS_V2,
    MODEL_VITS_EN,
    MODEL_TACOTRON_EN,
    list_tts_models,
)

import gradio as gr

# Cache helper per model name
_helpers = {}

def get_helper(model_name):
    if model_name not in _helpers:
        _helpers[model_name] = TTSHelper(model_name=model_name)
    return _helpers[model_name]


LANG_CHOICES = [f"{code} - {name}" for code, name in XTTS_LANGUAGES.items()]

def parse_lang(s):
    return s.split(" - ")[0].strip()


# =============================================================================
# HANDLERS
# =============================================================================

def handle_simple(text, model_name):
    if not text or not text.strip():
        return None, "Masukkan teks terlebih dahulu."
    try:
        h = get_helper(model_name)
        out = os.path.join(tempfile.gettempdir(), "tts_simple_output.wav")
        h.tts_simple(text.strip(), output_wav=out)
        return out, f"Audio dihasilkan dengan model {model_name}."
    except Exception as e:
        return None, f"Error: {e}"


def handle_xtts(text, language, speaker):
    if not text or not text.strip():
        return None, "Masukkan teks terlebih dahulu."
    try:
        h = get_helper(MODEL_XTTS_V2)
        lang = parse_lang(language)
        out = os.path.join(tempfile.gettempdir(), "tts_xtts_output.wav")
        h.tts_multilingual(text.strip(), language=lang, output_wav=out, speaker=speaker or None)
        return out, f"Audio XTTS v2 ({lang}) dengan speaker {speaker or 'default'}."
    except Exception as e:
        return None, f"Error: {e}"


def handle_clone(text, language, speaker_audio):
    if not text or not text.strip():
        return None, "Masukkan teks terlebih dahulu."
    if not speaker_audio:
        return None, "Upload atau rekam audio referensi."
    try:
        h = get_helper(MODEL_XTTS_V2)
        lang = parse_lang(language)
        out = os.path.join(tempfile.gettempdir(), "tts_clone_output.wav")
        h.tts_voice_clone(text.strip(), speaker_wav=speaker_audio, language=lang, output_wav=out)
        return out, f"Voice clone berhasil ({lang})."
    except Exception as e:
        return None, f"Error: {e}"


def handle_list_speakers():
    try:
        h = get_helper(MODEL_XTTS_V2)
        speakers = h.list_speakers()
        if not speakers:
            return "(Model aktif tidak memiliki speaker preset)"
        return "\n".join(f"- {s}" for s in speakers)
    except Exception as e:
        return f"Error: {e}"


def handle_list_models():
    try:
        models = list_tts_models()
        return f"Total: {len(models)} model\n\n" + "\n".join(models)
    except Exception as e:
        return f"Error: {e}"


def get_lang_table():
    return [[code, name] for code, name in XTTS_LANGUAGES.items()]


# =============================================================================
# UI BUILDER
# =============================================================================
def create_ui():
    with gr.Blocks(title="Coqui TTS - Uji Coba", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Coqui TTS - Pengujian Model\n"
            "Antarmuka untuk mencoba sintesis suara Coqui TTS: XTTS v2 multilingual dan voice cloning."
        )

        with gr.Tab("TTS Sederhana"):
            gr.Markdown("### Text-to-Speech (model preset)")
            simple_model = gr.Dropdown(
                choices=[MODEL_VITS_EN, MODEL_TACOTRON_EN, MODEL_XTTS_V2],
                value=MODEL_VITS_EN,
                label="Model"
            )
            simple_text = gr.Textbox(
                label="Teks",
                placeholder="Ketik teks (default: bahasa Inggris untuk VITS/Tacotron)...",
                lines=3,
                value="Hello, this is a test of Coqui TTS."
            )
            simple_btn = gr.Button("Hasilkan Suara", variant="primary")
            simple_audio = gr.Audio(label="Audio Hasil", type="filepath")
            simple_status = gr.Textbox(label="Status", interactive=False)
            simple_btn.click(
                handle_simple,
                inputs=[simple_text, simple_model],
                outputs=[simple_audio, simple_status]
            )

        with gr.Tab("XTTS v2 Multilingual"):
            gr.Markdown("### XTTS v2 - 17 Bahasa, Speaker Preset")
            with gr.Row():
                xtts_lang = gr.Dropdown(
                    choices=LANG_CHOICES, value="en - English (Inggris)",
                    label="Bahasa"
                )
                xtts_speaker = gr.Textbox(
                    label="Speaker (kosongkan untuk default)",
                    placeholder="misal: Claribel Dervla"
                )
            xtts_text = gr.Textbox(
                label="Teks",
                placeholder="Ketik teks dalam bahasa yang dipilih...",
                lines=3,
                value="Hello world, this is XTTS version 2."
            )
            xtts_btn = gr.Button("Hasilkan Suara", variant="primary")
            xtts_audio = gr.Audio(label="Audio Hasil", type="filepath")
            xtts_status = gr.Textbox(label="Status", interactive=False)
            xtts_btn.click(
                handle_xtts,
                inputs=[xtts_text, xtts_lang, xtts_speaker],
                outputs=[xtts_audio, xtts_status]
            )

        with gr.Tab("Voice Cloning"):
            gr.Markdown(
                "### Clone Suara dari Audio Referensi\n"
                "Upload / rekam audio referensi (3-10 detik, suara bersih), lalu ketik teks."
            )
            clone_ref = gr.Audio(
                label="Audio Referensi",
                type="filepath",
                sources=["upload", "microphone"]
            )
            clone_lang = gr.Dropdown(
                choices=LANG_CHOICES, value="en - English (Inggris)",
                label="Bahasa Teks"
            )
            clone_text = gr.Textbox(
                label="Teks yang akan diucapkan oleh suara klona",
                lines=3,
                value="This is my cloned voice speaking."
            )
            clone_btn = gr.Button("Clone & Sintesis", variant="primary")
            clone_audio = gr.Audio(label="Audio Hasil Clone", type="filepath")
            clone_status = gr.Textbox(label="Status", interactive=False)
            clone_btn.click(
                handle_clone,
                inputs=[clone_text, clone_lang, clone_ref],
                outputs=[clone_audio, clone_status]
            )

        with gr.Tab("Speaker Preset"):
            gr.Markdown(
                "### Daftar Speaker Preset XTTS v2\n"
                "Nama-nama di bawah bisa dipakai di tab XTTS v2."
            )
            spk_btn = gr.Button("Muat Daftar Speaker", variant="primary")
            spk_output = gr.Textbox(label="Daftar Speaker", lines=15, interactive=False)
            spk_btn.click(handle_list_speakers, outputs=spk_output)

        with gr.Tab("Daftar Bahasa"):
            gr.Markdown("### Bahasa yang didukung XTTS v2")
            gr.Dataframe(
                headers=["Kode", "Nama Bahasa"],
                value=get_lang_table(),
                interactive=False
            )

        with gr.Tab("Daftar Model"):
            gr.Markdown("### Model TTS yang Tersedia")
            model_btn = gr.Button("Muat Daftar Model", variant="primary")
            model_output = gr.Textbox(label="Model", lines=20, interactive=False)
            model_btn.click(handle_list_models, outputs=model_output)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True
    )
