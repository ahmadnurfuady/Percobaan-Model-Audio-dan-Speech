"""
=============================================================================
 Skrip Pengujian OpenAI Whisper (Speech-to-Text / ASR)
 Proyek: Percobaan Generated
=============================================================================
Fitur:
1. Transkripsi audio ke teks (auto-detect bahasa)
2. Transkripsi dengan paksa bahasa tertentu (misal: Indonesia)
3. Terjemahan audio ke Bahasa Inggris (translate mode)
4. Deteksi bahasa dari audio
5. Transkripsi dengan timestamp per segmen
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

# Pastikan ffmpeg ada di PATH; dibutuhkan Whisper untuk decode audio.
if sys.platform == "win32":
    _ffmpeg_candidates = [
        r"C:\Program Files\ffmpeg\bin",
        r"C:\ffmpeg\bin",
    ]
    # WinGet install path (Gyan.FFmpeg)
    _winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    if os.path.isdir(_winget_base):
        for _d in os.listdir(_winget_base):
            if _d.startswith("Gyan.FFmpeg"):
                _pkg = os.path.join(_winget_base, _d)
                for _sub in os.listdir(_pkg):
                    _bin = os.path.join(_pkg, _sub, "bin")
                    if os.path.isfile(os.path.join(_bin, "ffmpeg.exe")):
                        _ffmpeg_candidates.insert(0, _bin)
    for _p in _ffmpeg_candidates:
        if os.path.isdir(_p) and _p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")
            break

import torch

# Folder lokal penyimpanan model Whisper (relatif terhadap lokasi skrip ini)
WHISPER_MODEL_DIR = str(Path(__file__).resolve().parent / "Model Whisper")
os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)

# =============================================================================
# KONSTANTA
# =============================================================================
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

WHISPER_MODEL_SIZES = {
    "tiny":     "~75 MB",
    "base":     "~145 MB",
    "small":    "~466 MB",
    "medium":   "~1.5 GB",
    "large":    "~2.9 GB",
    "large-v2": "~2.9 GB",
    "large-v3": "~2.9 GB",
}

# Bahasa-bahasa umum yang didukung Whisper (total 99 bahasa)
POPULAR_LANGUAGES = {
    "id": "Indonesian (Bahasa Indonesia)",
    "en": "English (Inggris)",
    "zh": "Chinese (Mandarin)",
    "ja": "Japanese (Jepang)",
    "ko": "Korean (Korea)",
    "ar": "Arabic (Arab)",
    "fr": "French (Prancis)",
    "de": "German (Jerman)",
    "es": "Spanish (Spanyol)",
    "pt": "Portuguese (Portugis)",
    "ru": "Russian (Rusia)",
    "it": "Italian (Italia)",
    "nl": "Dutch (Belanda)",
    "tr": "Turkish (Turki)",
    "hi": "Hindi (India)",
    "th": "Thai (Thailand)",
    "vi": "Vietnamese (Vietnam)",
    "ms": "Malay (Melayu)",
    "pl": "Polish (Polandia)",
    "sv": "Swedish (Swedia)",
}


# =============================================================================
# KELAS HELPER
# =============================================================================
class WhisperHelper:
    def __init__(self, model_name="base", device=None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        print(f"[*] Model    : whisper/{model_name}")
        print(f"[*] Perangkat: {self.device.upper()}")

    def load(self):
        if self._model is None:
            import whisper
            print(f"[*] Memuat model whisper/{self.model_name} ...")
            print(f"[*] Model dir: {WHISPER_MODEL_DIR}")
            self._model = whisper.load_model(
                self.model_name,
                device=self.device,
                download_root=WHISPER_MODEL_DIR,
            )
            print(f"[+] Model berhasil dimuat!")
        return self._model

    # ---- FITUR 1: TRANSKRIPSI AUTO-DETECT ----
    def transcribe(self, audio_path, language=None):
        """Transkripsi audio. language=None = auto-detect."""
        model = self.load()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")
        print(f"\n[*] Transkripsi: {audio_path}")
        opts = {}
        if language:
            opts["language"] = language
        result = model.transcribe(str(audio_path), **opts)
        return result

    # ---- FITUR 2: TRANSKRIPSI PAKSA BAHASA TERTENTU ----
    def transcribe_language(self, audio_path, language="id"):
        """Transkripsi dengan memaksa bahasa tertentu."""
        return self.transcribe(audio_path, language=language)

    # ---- FITUR 3: TERJEMAHAN KE BAHASA INGGRIS ----
    def translate_to_english(self, audio_path):
        """Transkripsi + terjemahkan ke Bahasa Inggris."""
        model = self.load()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")
        print(f"\n[*] Terjemahan ke Inggris: {audio_path}")
        result = model.transcribe(str(audio_path), task="translate")
        return result

    # ---- FITUR 4: DETEKSI BAHASA ----
    def detect_language(self, audio_path):
        """Deteksi bahasa dominan dari audio (30 detik pertama)."""
        import whisper
        model = self.load()
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")
        print(f"\n[*] Deteksi bahasa: {audio_path}")
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        top = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        return top

    # ---- FITUR 5: TRANSKRIPSI DENGAN TIMESTAMP ----
    def transcribe_with_timestamps(self, audio_path, language=None):
        """Transkripsi dengan detail timestamp per segmen."""
        result = self.transcribe(audio_path, language=language)
        segments = result.get("segments", [])
        return result["text"], segments


def tampilkan_bahasa():
    print("\n" + "=" * 60)
    print(" BAHASA POPULER YANG DIDUKUNG WHISPER (dari 99 total)")
    print("=" * 60)
    items = list(POPULAR_LANGUAGES.items())
    for i in range(0, len(items), 2):
        col1 = f"{items[i][0]}: {items[i][1]}"
        col2 = f"{items[i+1][0]}: {items[i+1][1]}" if i + 1 < len(items) else ""
        print(f"  {col1:<35} {col2}")
    print("=" * 60 + "\n")


def tampilkan_model():
    print("\n" + "=" * 50)
    print(" MODEL WHISPER YANG TERSEDIA")
    print("=" * 50)
    for m, size in WHISPER_MODEL_SIZES.items():
        print(f"  {m:<12} {size}")
    print("=" * 50 + "\n")


# =============================================================================
# MENU INTERAKTIF
# =============================================================================
def interactive_menu(helper: WhisperHelper):
    while True:
        print("\n" + "=" * 60)
        print("       PROGRAM UJI COBA OPENAI WHISPER")
        print("=" * 60)
        print(f" Model      : whisper/{helper.model_name}")
        print(f" Perangkat  : {helper.device.upper()}")
        print("-" * 60)
        print(" [1] Transkripsi audio (auto-detect bahasa)")
        print(" [2] Transkripsi dengan paksa bahasa tertentu")
        print(" [3] Terjemahan audio ke Bahasa Inggris")
        print(" [4] Deteksi bahasa dari audio")
        print(" [5] Transkripsi dengan timestamp per segmen")
        print(" [6] Daftar bahasa yang didukung")
        print(" [7] Daftar model Whisper")
        print(" [0] Keluar")
        print("=" * 60)

        pilihan = input("Pilih menu [0-7]: ").strip()
        if pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        try:
            if pilihan in ("1", "2", "3", "4", "5"):
                audio = input("Path file audio (.wav/.mp3/.m4a/dll): ").strip(' "\'')
                if not audio:
                    print("[!] Path audio tidak boleh kosong.")
                    continue

            if pilihan == "1":
                result = helper.transcribe(audio)
                print(f"\n[+] Bahasa terdeteksi: {result.get('language', '?')}")
                print(f"[+] Transkripsi:\n{result['text']}")

            elif pilihan == "2":
                lang = input("Kode bahasa (misal: id, en, ja): ").strip() or "id"
                result = helper.transcribe_language(audio, language=lang)
                print(f"\n[+] Transkripsi ({lang}):\n{result['text']}")

            elif pilihan == "3":
                result = helper.translate_to_english(audio)
                print(f"\n[+] Terjemahan (Inggris):\n{result['text']}")

            elif pilihan == "4":
                top = helper.detect_language(audio)
                print("\n[+] Top-5 bahasa terdeteksi:")
                for lang, prob in top:
                    bar = "█" * int(prob * 30)
                    print(f"  {lang:<6} {prob:.1%}  {bar}")

            elif pilihan == "5":
                lang = input("Kode bahasa (kosongkan = auto): ").strip() or None
                text, segments = helper.transcribe_with_timestamps(audio, language=lang)
                print(f"\n[+] Teks lengkap:\n{text}\n")
                print(f"[+] {len(segments)} segmen:")
                for seg in segments:
                    t0 = f"{seg['start']:.2f}s"
                    t1 = f"{seg['end']:.2f}s"
                    print(f"  [{t0} → {t1}] {seg['text'].strip()}")

            elif pilihan == "6":
                tampilkan_bahasa()

            elif pilihan == "7":
                tampilkan_model()

            else:
                print("[!] Pilihan tidak valid.")
        except Exception as e:
            print(f"[!] Kesalahan: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Uji coba OpenAI Whisper ASR.")
    parser.add_argument("--mode", choices=["menu", "transcribe", "translate", "detect", "timestamp"],
                        default="menu")
    parser.add_argument("--audio", type=str, default="", help="Path file audio")
    parser.add_argument("--lang", type=str, default=None, help="Kode bahasa (misal: id, en)")
    parser.add_argument("--model", type=str, default="base",
                        choices=WHISPER_MODELS, help="Ukuran model Whisper")

    args = parser.parse_args()
    helper = WhisperHelper(model_name=args.model)

    if args.mode == "menu":
        interactive_menu(helper)
    else:
        if not args.audio:
            print("[!] Perlu --audio <path>")
            import sys; sys.exit(1)
        if args.mode == "transcribe":
            r = helper.transcribe(args.audio, language=args.lang)
            print(f"Bahasa: {r.get('language')}\n{r['text']}")
        elif args.mode == "translate":
            r = helper.translate_to_english(args.audio)
            print(r['text'])
        elif args.mode == "detect":
            top = helper.detect_language(args.audio)
            for lang, prob in top:
                print(f"{lang}: {prob:.1%}")
        elif args.mode == "timestamp":
            text, segs = helper.transcribe_with_timestamps(args.audio, language=args.lang)
            print(text)
            for s in segs:
                print(f"[{s['start']:.2f}→{s['end']:.2f}] {s['text'].strip()}")


if __name__ == "__main__":
    main()
