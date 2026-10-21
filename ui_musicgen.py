"""
UI Gradio untuk Pengujian Meta MusicGen
Fitur: Text-to-Music, Variasi, Contoh Prompt
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

from coba_musicgen import (
    MusicGenHelper,
    MUSICGEN_MODELS,
    EXAMPLE_PROMPTS,
    DEFAULT_MODEL,
)

import gradio as gr

# Cache helper per model key
_helpers: dict[str, MusicGenHelper] = {}

def get_helper(model_key=DEFAULT_MODEL) -> MusicGenHelper:
    if model_key not in _helpers:
        _helpers[model_key] = MusicGenHelper(model_key=model_key)
    return _helpers[model_key]


# =============================================================================
# HANDLERS
# =============================================================================

def handle_generate(prompt, duration, model_key):
    if not prompt or not prompt.strip():
        return None, "Masukkan deskripsi musik terlebih dahulu."
    try:
        h = get_helper(model_key)
        out = os.path.join(tempfile.gettempdir(), "musicgen_output.wav")
        results = h.generate(prompt.strip(), duration_seconds=int(duration), output_wav=out)
        return results[0], f"Berhasil generate {int(duration)}s audio. Model: {model_key}"
    except Exception as e:
        return None, f"Error: {e}"


def handle_variations(prompt, n_var, duration, model_key):
    if not prompt or not prompt.strip():
        return None, None, None, "Masukkan deskripsi musik terlebih dahulu."
    try:
        h = get_helper(model_key)
        n = int(n_var)
        out_dir = tempfile.gettempdir()
        results = h.generate_variations(prompt.strip(), n=n, duration_seconds=int(duration), output_dir=out_dir)
        # Kembalikan hingga 3 variasi
        outputs = results + [None] * (3 - len(results))
        status = f"{len(results)} variasi berhasil dibuat. Model: {model_key}"
        return outputs[0], outputs[1], outputs[2], status
    except Exception as e:
        return None, None, None, f"Error: {e}"


def fill_example(evt: gr.SelectData):
    return EXAMPLE_PROMPTS[evt.index]


def get_model_table():
    rows = []
    for key, hf_id in MUSICGEN_MODELS.items():
        rows.append([key, hf_id])
    return rows


# =============================================================================
# UI BUILDER
# =============================================================================
def create_ui():
    model_choices = list(MUSICGEN_MODELS.keys())

    with gr.Blocks(title="MusicGen - Uji Coba", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Meta MusicGen - Pengujian Text-to-Music\n"
            "Generate musik dari deskripsi teks menggunakan model MusicGen."
        )

        with gr.Tab("Generate Musik"):
            gr.Markdown("### Generate satu klip musik dari deskripsi teks")
            with gr.Row():
                gen_model = gr.Dropdown(
                    choices=model_choices, value=DEFAULT_MODEL, label="Model"
                )
                gen_duration = gr.Slider(
                    minimum=2, maximum=30, value=8, step=1,
                    label="Durasi (detik)"
                )
            gen_prompt = gr.Textbox(
                label="Deskripsi Musik",
                placeholder="misal: upbeat jazz piano with light drums",
                lines=3,
                value="upbeat jazz piano with light drums"
            )
            gen_btn = gr.Button("Generate Musik", variant="primary")
            gen_audio = gr.Audio(label="Hasil Audio", type="filepath")
            gen_status = gr.Textbox(label="Status", interactive=False)
            gen_btn.click(
                handle_generate,
                inputs=[gen_prompt, gen_duration, gen_model],
                outputs=[gen_audio, gen_status]
            )

        with gr.Tab("Variasi"):
            gr.Markdown("### Generate beberapa variasi dari satu prompt")
            with gr.Row():
                var_model = gr.Dropdown(
                    choices=model_choices, value=DEFAULT_MODEL, label="Model"
                )
                var_n = gr.Slider(minimum=1, maximum=3, value=2, step=1, label="Jumlah Variasi")
                var_duration = gr.Slider(minimum=2, maximum=30, value=8, step=1, label="Durasi (detik)")
            var_prompt = gr.Textbox(
                label="Deskripsi Musik",
                placeholder="misal: calm lo-fi hip hop with rain sounds",
                lines=3,
                value="calm lo-fi hip hop with rain sounds"
            )
            var_btn = gr.Button("Generate Variasi", variant="primary")
            with gr.Row():
                var_audio1 = gr.Audio(label="Variasi 1", type="filepath")
                var_audio2 = gr.Audio(label="Variasi 2", type="filepath")
                var_audio3 = gr.Audio(label="Variasi 3", type="filepath")
            var_status = gr.Textbox(label="Status", interactive=False)
            var_btn.click(
                handle_variations,
                inputs=[var_prompt, var_n, var_duration, var_model],
                outputs=[var_audio1, var_audio2, var_audio3, var_status]
            )

        with gr.Tab("Contoh Prompt"):
            gr.Markdown(
                "### Contoh Prompt Musik\n"
                "Klik baris untuk menyalin ke clipboard, atau copy manual."
            )
            example_list = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[[p] for p in EXAMPLE_PROMPTS],
                label="Klik untuk pakai prompt ini di tab Generate"
            )
            selected_prompt = gr.Textbox(label="Prompt Terpilih", interactive=True)
            example_list.select(fill_example, outputs=selected_prompt)

        with gr.Tab("Daftar Model"):
            gr.Markdown("### Model MusicGen yang Tersedia")
            gr.Dataframe(
                headers=["Kunci", "Model HuggingFace"],
                value=get_model_table(),
                interactive=False
            )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7863,
        share=False,
        inbrowser=True
    )
