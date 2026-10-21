"""
UI Gradio untuk Pengujian AudioLDM (Text-to-Audio)
Fitur: Generate Audio, Pengaturan Lanjutan, Variasi, Contoh Prompt
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

from coba_audioldm import (
    AudioLDMHelper,
    AUDIOLDM_MODELS,
    EXAMPLE_PROMPTS,
    DEFAULT_MODEL,
    SAMPLE_RATE,
)

import gradio as gr

_helpers: dict = {}


def get_helper(model_key=DEFAULT_MODEL) -> AudioLDMHelper:
    if model_key not in _helpers:
        _helpers[model_key] = AudioLDMHelper(model_key=model_key)
    return _helpers[model_key]


# =============================================================================
# HANDLERS
# =============================================================================

def handle_generate(prompt, duration, steps, model_key):
    if not prompt or not prompt.strip():
        return None, "Masukkan deskripsi suara terlebih dahulu."
    try:
        h = get_helper(model_key)
        out = os.path.join(tempfile.gettempdir(), "audioldm_simple.wav")
        paths = h.generate(
            prompt.strip(),
            duration_seconds=float(duration),
            num_inference_steps=int(steps),
            output_wav=out,
        )
        return paths[0], f"Berhasil: {duration}s @ {SAMPLE_RATE} Hz, {int(steps)} steps."
    except Exception as e:
        return None, f"Error: {e}"


def handle_advanced(prompt, negative, duration, steps, guidance, seed, model_key):
    if not prompt or not prompt.strip():
        return None, "Masukkan deskripsi suara terlebih dahulu."
    try:
        h = get_helper(model_key)
        out = os.path.join(tempfile.gettempdir(), "audioldm_advanced.wav")
        use_seed = int(seed) if seed is not None and int(seed) >= 0 else None
        paths = h.generate(
            prompt.strip(),
            duration_seconds=float(duration),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            negative_prompt=(negative or "").strip() or None,
            output_wav=out,
            seed=use_seed,
        )
        seed_info = f", seed={use_seed}" if use_seed is not None else ""
        return paths[0], f"Berhasil: {duration}s, {int(steps)} steps, guidance={guidance}{seed_info}."
    except Exception as e:
        return None, f"Error: {e}"


def handle_variations(prompt, n_var, duration, steps, model_key):
    if not prompt or not prompt.strip():
        return None, None, None, "Masukkan deskripsi suara terlebih dahulu."
    try:
        h = get_helper(model_key)
        out = os.path.join(tempfile.gettempdir(), "audioldm_var.wav")
        paths = h.generate(
            prompt.strip(),
            duration_seconds=float(duration),
            num_inference_steps=int(steps),
            n_variations=int(n_var),
            output_wav=out,
        )
        slots = paths + [None] * (3 - len(paths))
        return slots[0], slots[1], slots[2], f"{len(paths)} variasi berhasil dibuat."
    except Exception as e:
        return None, None, None, f"Error: {e}"


def pick_example(evt: gr.SelectData):
    return EXAMPLE_PROMPTS[evt.index]


def get_model_table():
    rows = []
    for key, hf_id in AUDIOLDM_MODELS.items():
        rows.append([key, hf_id])
    return rows


# =============================================================================
# UI BUILDER
# =============================================================================
def create_ui():
    model_choices = list(AUDIOLDM_MODELS.keys())

    with gr.Blocks(title="AudioLDM - Uji Coba", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# AudioLDM - Pengujian Text-to-Audio\n"
            "Membuat **suara / sound effect** dari deskripsi teks. "
            "Beda dengan MusicGen yang fokus musik, AudioLDM fokus suara umum "
            "(hujan, hewan, mesin, suasana)."
        )

        with gr.Tab("Generate Audio"):
            gr.Markdown("### Generate suara dari deskripsi teks")
            with gr.Row():
                g_model = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, label="Model")
                g_duration = gr.Slider(1, 30, value=5, step=1, label="Durasi (detik)")
                g_steps = gr.Slider(5, 50, value=10, step=1, label="Inference Steps (makin tinggi makin bagus/lambat)")
            g_prompt = gr.Textbox(
                label="Deskripsi Suara",
                placeholder="misal: heavy rain falling on a metal roof",
                lines=3,
                value="heavy rain falling on a metal roof",
            )
            g_btn = gr.Button("Generate Audio", variant="primary")
            g_audio = gr.Audio(label="Hasil Audio", type="filepath")
            g_status = gr.Textbox(label="Status", interactive=False)
            g_btn.click(
                handle_generate,
                inputs=[g_prompt, g_duration, g_steps, g_model],
                outputs=[g_audio, g_status],
            )

        with gr.Tab("Pengaturan Lanjutan"):
            gr.Markdown(
                "### Kontrol penuh: negative prompt, guidance scale, seed\n"
                "- **Negative prompt**: karakteristik yang dihindari\n"
                "- **Guidance scale**: seberapa patuh ke prompt (rendah = lebih kreatif)\n"
                "- **Seed**: isi angka >= 0 agar hasil bisa direproduksi, -1 untuk acak"
            )
            with gr.Row():
                a_model = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, label="Model")
                a_duration = gr.Slider(1, 30, value=5, step=1, label="Durasi (detik)")
                a_steps = gr.Slider(5, 50, value=20, step=1, label="Inference Steps")
            with gr.Row():
                a_guidance = gr.Slider(1.0, 5.0, value=2.5, step=0.1, label="Guidance Scale")
                a_seed = gr.Number(value=-1, label="Seed (-1 = acak)", precision=0)
            a_prompt = gr.Textbox(
                label="Deskripsi Suara",
                lines=3,
                value="birds chirping in a forest at sunrise",
            )
            a_negative = gr.Textbox(
                label="Negative Prompt (opsional)",
                placeholder="misal: low quality, distortion, noise",
                lines=2,
            )
            a_btn = gr.Button("Generate Audio", variant="primary")
            a_audio = gr.Audio(label="Hasil Audio", type="filepath")
            a_status = gr.Textbox(label="Status", interactive=False)
            a_btn.click(
                handle_advanced,
                inputs=[a_prompt, a_negative, a_duration, a_steps, a_guidance, a_seed, a_model],
                outputs=[a_audio, a_status],
            )

        with gr.Tab("Variasi"):
            gr.Markdown("### Generate beberapa variasi dari satu prompt")
            with gr.Row():
                v_model = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, label="Model")
                v_n = gr.Slider(1, 3, value=2, step=1, label="Jumlah Variasi")
                v_duration = gr.Slider(1, 20, value=5, step=1, label="Durasi (detik)")
                v_steps = gr.Slider(5, 30, value=10, step=1, label="Inference Steps")
            v_prompt = gr.Textbox(
                label="Deskripsi Suara",
                lines=3,
                value="ocean waves crashing on a rocky shore",
            )
            v_btn = gr.Button("Generate Variasi", variant="primary")
            with gr.Row():
                v_a1 = gr.Audio(label="Variasi 1", type="filepath")
                v_a2 = gr.Audio(label="Variasi 2", type="filepath")
                v_a3 = gr.Audio(label="Variasi 3", type="filepath")
            v_status = gr.Textbox(label="Status", interactive=False)
            v_btn.click(
                handle_variations,
                inputs=[v_prompt, v_n, v_duration, v_steps, v_model],
                outputs=[v_a1, v_a2, v_a3, v_status],
            )

        with gr.Tab("Contoh Prompt"):
            gr.Markdown("### Contoh prompt — klik salah satu untuk melihat teksnya")
            ex_list = gr.Dataset(
                components=[gr.Textbox(visible=False)],
                samples=[[p] for p in EXAMPLE_PROMPTS],
                label="Contoh Prompt AudioLDM",
            )
            ex_out = gr.Textbox(label="Prompt Terpilih (copy ke tab Generate)", interactive=True)
            ex_list.select(pick_example, outputs=ex_out)

        with gr.Tab("Daftar Model"):
            gr.Markdown("### Model AudioLDM yang Tersedia")
            gr.Dataframe(
                headers=["Kunci", "Model HuggingFace"],
                value=get_model_table(),
                interactive=False,
            )

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7864,
        share=False,
        inbrowser=True,
    )
