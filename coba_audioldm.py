"""
=============================================================================
 Skrip Pengujian AudioLDM (Text-to-Audio Generation)
 Proyek: Percobaan Generated
=============================================================================
AudioLDM MEMBUAT suara dari deskripsi teks (kebalikan dari BEATs).
Cocok untuk sound effect & suara ambient: hujan, anjing menggonggong,
suara mesin, dll. Bedanya dengan MusicGen: MusicGen fokus musik,
AudioLDM fokus suara/efek audio umum.

Fitur:
1. Generate audio dari teks
2. Generate dengan durasi & kualitas (inference steps) kustom
3. Negative prompt (hindari karakteristik tertentu)
4. Generate beberapa variasi sekaligus
5. Contoh prompt
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

import torch
import numpy as np
import scipy.io.wavfile as wavfile

# =============================================================================
# KONSTANTA
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = _SCRIPT_DIR / "Model AudioLDM"

MODEL_LOCAL = MODEL_DIR / "audioldm-s-full-v2"

AUDIOLDM_MODELS = {
    "s-full-v2": "cvssp/audioldm-s-full-v2",   # small, default (lokal)
    "m-full":    "cvssp/audioldm-m-full",      # medium
    "l-full":    "cvssp/audioldm-l-full",      # large
}

DEFAULT_MODEL = "s-full-v2"
SAMPLE_RATE = 16000  # AudioLDM selalu 16 kHz

EXAMPLE_PROMPTS = [
    "heavy rain falling on a metal roof",
    "a dog barking in the distance",
    "birds chirping in a forest at sunrise",
    "ocean waves crashing on a rocky shore",
    "a car engine starting and revving",
    "footsteps walking on crunchy gravel",
    "thunder rumbling during a storm",
    "a crowded cafe with people talking",
    "keyboard typing sounds in a quiet office",
    "wind blowing through tall grass",
]


# =============================================================================
# KELAS HELPER
# =============================================================================
class AudioLDMHelper:
    def __init__(self, model_key=DEFAULT_MODEL, device=None):
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None

        # Pakai model lokal jika tersedia, kalau tidak ambil dari HuggingFace
        if model_key == DEFAULT_MODEL and MODEL_LOCAL.exists():
            self.model_path = str(MODEL_LOCAL)
        else:
            self.model_path = AUDIOLDM_MODELS.get(model_key, AUDIOLDM_MODELS[DEFAULT_MODEL])

        print(f"[*] Model    : {self.model_path}")
        print(f"[*] Perangkat: {self.device.upper()}")

    def load(self):
        if self._pipe is None:
            from diffusers import AudioLDMPipeline
            print(f"[*] Memuat pipeline AudioLDM ...")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._pipe = AudioLDMPipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
            )
            self._pipe = self._pipe.to(self.device)
            self._pipe.set_progress_bar_config(disable=None)
            print(f"[+] Pipeline berhasil dimuat! Sample rate: {SAMPLE_RATE} Hz")
        return self._pipe

    # ---- FITUR 1-4: GENERATE AUDIO ----
    def generate(
        self,
        prompt,
        duration_seconds=5.0,
        num_inference_steps=10,
        guidance_scale=2.5,
        negative_prompt=None,
        n_variations=1,
        output_wav="output_audioldm.wav",
        seed=None,
    ):
        """
        Generate audio dari deskripsi teks.
        duration_seconds    : panjang audio (detik)
        num_inference_steps : makin tinggi makin bagus tapi lambat (10-50)
        guidance_scale      : seberapa patuh ke prompt (1.0-5.0)
        negative_prompt     : karakteristik yang dihindari
        n_variations        : jumlah variasi yang dibuat
        """
        pipe = self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        print(f"\n[*] Generating {n_variations} klip x {duration_seconds}s ...")
        print(f"    Prompt: '{prompt}'")
        if negative_prompt:
            print(f"    Negative: '{negative_prompt}'")
        print(f"    Steps: {num_inference_steps}, guidance: {guidance_scale}")

        result = pipe(
            prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=int(num_inference_steps),
            audio_length_in_s=float(duration_seconds),
            guidance_scale=float(guidance_scale),
            num_waveforms_per_prompt=int(n_variations),
            generator=generator,
        )

        audios = result.audios
        paths = []

        for i, audio in enumerate(audios):
            audio = np.asarray(audio, dtype=np.float32)
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak
            audio_int16 = (audio * 32767).astype(np.int16)

            if len(audios) == 1:
                out_path = output_wav
            else:
                stem = Path(output_wav).stem
                ext = Path(output_wav).suffix or ".wav"
                out_path = str(Path(output_wav).parent / f"{stem}_{i + 1}{ext}")

            wavfile.write(out_path, SAMPLE_RATE, audio_int16)
            dur = len(audio_int16) / SAMPLE_RATE
            print(f"[+] Audio [{i + 1}] tersimpan: {Path(out_path).resolve()} ({dur:.1f}s)")
            paths.append(out_path)

        return paths


# =============================================================================
# MENU INTERAKTIF
# =============================================================================
def interactive_menu(helper: AudioLDMHelper):
    while True:
        print("\n" + "=" * 65)
        print("       PROGRAM UJI COBA AUDIOLDM (Text-to-Audio)")
        print("=" * 65)
        print(f" Model    : {helper.model_path}")
        print(f" Perangkat: {helper.device.upper()}")
        print("-" * 65)
        print(" [1] Generate audio dari teks (cepat, pengaturan default)")
        print(" [2] Generate dengan durasi & kualitas kustom")
        print(" [3] Generate dengan negative prompt")
        print(" [4] Generate beberapa variasi sekaligus")
        print(" [5] Lihat contoh prompt")
        print(" [6] Lihat daftar model AudioLDM")
        print(" [0] Keluar")
        print("=" * 65)

        pilihan = input("Pilih menu [0-6]: ").strip()
        if pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        try:
            if pilihan in ("1", "2", "3", "4"):
                prompt = input("Deskripsi suara: ").strip()
                if not prompt:
                    print("[!] Deskripsi tidak boleh kosong.")
                    continue

            if pilihan == "1":
                out = input("Nama file output [default output_audioldm.wav]: ").strip() or "output_audioldm.wav"
                helper.generate(prompt, output_wav=out)

            elif pilihan == "2":
                try:
                    dur = float(input("Durasi detik (1-30) [default 5]: ").strip() or "5")
                    dur = max(1.0, min(30.0, dur))
                except ValueError:
                    dur = 5.0
                try:
                    steps = int(input("Inference steps (10-50) [default 10]: ").strip() or "10")
                    steps = max(5, min(100, steps))
                except ValueError:
                    steps = 10
                out = input("Nama file output [default output_audioldm.wav]: ").strip() or "output_audioldm.wav"
                helper.generate(prompt, duration_seconds=dur, num_inference_steps=steps, output_wav=out)

            elif pilihan == "3":
                neg = input("Negative prompt (misal: low quality, noise): ").strip()
                out = input("Nama file output [default output_audioldm.wav]: ").strip() or "output_audioldm.wav"
                helper.generate(prompt, negative_prompt=neg or None, output_wav=out)

            elif pilihan == "4":
                try:
                    n = int(input("Jumlah variasi (1-5) [default 3]: ").strip() or "3")
                    n = max(1, min(5, n))
                except ValueError:
                    n = 3
                helper.generate(prompt, n_variations=n, output_wav="variasi.wav")

            elif pilihan == "5":
                print("\n[+] Contoh prompt:")
                for i, p in enumerate(EXAMPLE_PROMPTS, 1):
                    print(f"  {i:>2}. {p}")

            elif pilihan == "6":
                print("\n[+] Model AudioLDM yang tersedia:")
                for key, hf_id in AUDIOLDM_MODELS.items():
                    tag = "  [LOKAL]" if key == DEFAULT_MODEL and MODEL_LOCAL.exists() else ""
                    print(f"  {key:<12} -> {hf_id}{tag}")

            else:
                print("[!] Pilihan tidak valid.")
        except Exception as e:
            print(f"[!] Kesalahan: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Uji coba AudioLDM text-to-audio.")
    parser.add_argument("--mode", choices=["menu", "generate"], default="menu")
    parser.add_argument("--prompt", type=str, default="heavy rain falling on a metal roof")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--n", type=int, default=1, help="Jumlah variasi")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="output_audioldm.wav")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(AUDIOLDM_MODELS.keys()))

    args = parser.parse_args()
    helper = AudioLDMHelper(model_key=args.model)

    if args.mode == "generate":
        helper.generate(
            args.prompt,
            duration_seconds=args.duration,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt or None,
            n_variations=args.n,
            output_wav=args.output,
            seed=args.seed,
        )
    else:
        interactive_menu(helper)


if __name__ == "__main__":
    main()
