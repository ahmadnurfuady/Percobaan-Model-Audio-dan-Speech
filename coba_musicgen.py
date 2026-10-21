"""
=============================================================================
 Skrip Pengujian Meta MusicGen (Text-to-Music / Audio Generation)
 Proyek: Percobaan Generated
=============================================================================
Fitur:
1. Generate musik dari deskripsi teks (text-to-music)
2. Generate musik dengan durasi kustom
3. Generate beberapa variasi sekaligus (batch)
4. Daftar model MusicGen yang tersedia
=============================================================================
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Pastikan ffmpeg ada di PATH (dibutuhkan untuk encode audio output)
if sys.platform == "win32":
    _winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(_winget_base):
        for _d in os.listdir(_winget_base):
            if _d.startswith("Gyan.FFmpeg"):
                _pkg = os.path.join(_winget_base, _d)
                for _sub in os.listdir(_pkg):
                    _bin = os.path.join(_pkg, _sub, "bin")
                    if os.path.isfile(os.path.join(_bin, "ffmpeg.exe")):
                        if _bin not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

import torch
import numpy as np
import scipy.io.wavfile as wavfile

# =============================================================================
# KONSTANTA
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _SCRIPT_DIR / "Model Music Gen"

# Model lokal (hasil download)
MODEL_SMALL_LOCAL = str(MODEL_DIR / "musicgen-small")

# Nama model HuggingFace (fallback jika lokal tidak ada)
MUSICGEN_MODELS = {
    "small":         "facebook/musicgen-small",
    "medium":        "facebook/musicgen-medium",
    "large":         "facebook/musicgen-large",
    "stereo-small":  "facebook/musicgen-stereo-small",
    "stereo-medium": "facebook/musicgen-stereo-medium",
    "melody":        "facebook/musicgen-melody",
}

DEFAULT_MODEL = "small"

# Contoh prompt musik
EXAMPLE_PROMPTS = [
    "upbeat jazz piano with light drums",
    "calm lo-fi hip hop with rain sounds",
    "epic orchestral soundtrack for a fantasy film",
    "Indonesian gamelan music, traditional and slow",
    "electronic dance music with heavy bass drop",
    "acoustic guitar fingerpicking, calm and peaceful",
    "80s synth pop with retro vibes",
]


# =============================================================================
# KELAS HELPER
# =============================================================================
class MusicGenHelper:
    def __init__(self, model_key="small", device=None):
        self.model_key = model_key
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        self._processor = None

        # Tentukan path model
        if model_key == "small" and Path(MODEL_SMALL_LOCAL).exists():
            self.model_path = MODEL_SMALL_LOCAL
        else:
            self.model_path = MUSICGEN_MODELS.get(model_key, MUSICGEN_MODELS["small"])

        print(f"[*] Model    : {self.model_path}")
        print(f"[*] Perangkat: {self.device.upper()}")

    def load(self):
        if self._model is None:
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            print(f"[*] Memuat processor ...")
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            print(f"[*] Memuat model MusicGen ...")
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
            sample_rate = self._model.config.audio_encoder.sampling_rate
            print(f"[+] Model berhasil dimuat! Sample rate: {sample_rate} Hz")
        return self._processor, self._model

    def get_sample_rate(self):
        _, model = self.load()
        return model.config.audio_encoder.sampling_rate

    # ---- FITUR 1 & 2: GENERATE MUSIK ----
    def generate(self, prompts, duration_seconds=8, output_wav="output_music.wav"):
        """
        Generate musik dari satu atau beberapa prompt teks.
        prompts: str atau list of str
        duration_seconds: durasi audio yang dihasilkan
        """
        processor, model = self.load()

        if isinstance(prompts, str):
            prompts = [prompts]

        print(f"\n[*] Generating {len(prompts)} klip × {duration_seconds}s ...")
        for i, p in enumerate(prompts):
            print(f"    [{i+1}] {p}")

        inputs = processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # max_new_tokens ≈ 50 token/detik untuk musicgen-small (frame rate 50 Hz)
        max_tokens = int(duration_seconds * 50)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                guidance_scale=3.0,
            )

        sample_rate = model.config.audio_encoder.sampling_rate
        results = []

        for i, audio in enumerate(audio_values):
            # audio shape: [channels, samples]
            audio_np = audio.cpu().float().numpy()
            if audio_np.ndim == 2:
                # stereo → mono jika perlu, atau simpan stereo
                audio_out = audio_np[0]  # ambil channel pertama
            else:
                audio_out = audio_np

            # Normalize ke int16
            audio_int16 = (audio_out / np.abs(audio_out).max() * 32767).astype(np.int16)

            if len(prompts) == 1:
                out_path = output_wav
            else:
                stem = Path(output_wav).stem
                ext = Path(output_wav).suffix
                out_path = str(Path(output_wav).parent / f"{stem}_{i+1}{ext}")

            wavfile.write(out_path, sample_rate, audio_int16)
            duration_actual = len(audio_int16) / sample_rate
            print(f"[+] Audio [{i+1}] tersimpan: {Path(out_path).resolve()} ({duration_actual:.1f}s)")
            results.append(out_path)

        return results

    # ---- FITUR 3: BATCH VARIASI ----
    def generate_variations(self, prompt, n=3, duration_seconds=8, output_dir="."):
        """Generate n variasi dari satu prompt."""
        prompts = [prompt] * n
        out_wav = str(Path(output_dir) / "variation.wav")
        return self.generate(prompts, duration_seconds=duration_seconds, output_wav=out_wav)


# =============================================================================
# MENU INTERAKTIF
# =============================================================================
def interactive_menu(helper: MusicGenHelper):
    while True:
        print("\n" + "=" * 65)
        print("       PROGRAM UJI COBA META MUSICGEN")
        print("=" * 65)
        print(f" Model    : {helper.model_path}")
        print(f" Perangkat: {helper.device.upper()}")
        print("-" * 65)
        print(" [1] Generate musik dari teks")
        print(" [2] Generate musik dengan durasi kustom")
        print(" [3] Generate beberapa variasi dari satu prompt (batch)")
        print(" [4] Lihat contoh prompt")
        print(" [5] Lihat daftar model MusicGen")
        print(" [0] Keluar")
        print("=" * 65)

        pilihan = input("Pilih menu [0-5]: ").strip()
        if pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        try:
            if pilihan == "1":
                prompt = input("Deskripsi musik: ").strip()
                if not prompt:
                    print("[!] Deskripsi tidak boleh kosong.")
                    continue
                out = input("Nama file output [default output_music.wav]: ").strip() or "output_music.wav"
                helper.generate(prompt, output_wav=out)

            elif pilihan == "2":
                prompt = input("Deskripsi musik: ").strip()
                if not prompt:
                    print("[!] Deskripsi tidak boleh kosong.")
                    continue
                try:
                    dur = int(input("Durasi (detik, 1-30) [default 8]: ").strip() or "8")
                    dur = max(1, min(30, dur))
                except ValueError:
                    dur = 8
                out = input("Nama file output [default output_music.wav]: ").strip() or "output_music.wav"
                helper.generate(prompt, duration_seconds=dur, output_wav=out)

            elif pilihan == "3":
                prompt = input("Deskripsi musik: ").strip()
                if not prompt:
                    print("[!] Deskripsi tidak boleh kosong.")
                    continue
                try:
                    n = int(input("Jumlah variasi (1-5) [default 3]: ").strip() or "3")
                    n = max(1, min(5, n))
                except ValueError:
                    n = 3
                try:
                    dur = int(input("Durasi per klip (detik) [default 8]: ").strip() or "8")
                    dur = max(1, min(30, dur))
                except ValueError:
                    dur = 8
                helper.generate_variations(prompt, n=n, duration_seconds=dur)

            elif pilihan == "4":
                print("\n[+] Contoh prompt:")
                for i, p in enumerate(EXAMPLE_PROMPTS, 1):
                    print(f"  {i}. {p}")

            elif pilihan == "5":
                print("\n[+] Model MusicGen yang tersedia:")
                for key, hf_id in MUSICGEN_MODELS.items():
                    local = " [LOKAL]" if key == "small" and Path(MODEL_SMALL_LOCAL).exists() else ""
                    print(f"  {key:<16} → {hf_id}{local}")

            else:
                print("[!] Pilihan tidak valid.")
        except Exception as e:
            print(f"[!] Kesalahan: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Uji coba Meta MusicGen.")
    parser.add_argument("--mode", choices=["menu", "generate", "batch"], default="menu")
    parser.add_argument("--prompt", type=str, default="upbeat jazz with piano and drums")
    parser.add_argument("--duration", type=int, default=8, help="Durasi audio dalam detik")
    parser.add_argument("--n", type=int, default=3, help="Jumlah variasi (mode batch)")
    parser.add_argument("--output", type=str, default="output_music.wav")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MUSICGEN_MODELS.keys()))

    args = parser.parse_args()
    helper = MusicGenHelper(model_key=args.model)

    if args.mode == "generate":
        helper.generate(args.prompt, duration_seconds=args.duration, output_wav=args.output)
    elif args.mode == "batch":
        helper.generate_variations(args.prompt, n=args.n, duration_seconds=args.duration)
    else:
        interactive_menu(helper)


if __name__ == "__main__":
    main()
