"""
=============================================================================
 Skrip Pengujian Coqui TTS (Text-to-Speech)
 Proyek: Percobaan Generated
=============================================================================
Fitur:
1. Text-to-Speech sederhana (satu bahasa)             -> file .wav
2. Text-to-Speech XTTS v2 multilingual (16 bahasa)    -> file .wav
3. Voice Cloning (XTTS v2 dengan audio referensi)     -> file .wav
4. Daftar model TTS yang tersedia
5. Daftar bahasa yang didukung XTTS v2
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
os.environ["COQUI_TOS_AGREED"] = "1"  # Setuju otomatis dengan ToS XTTS

# Pastikan espeak-ng (Windows) ada di PATH; dibutuhkan oleh model VITS/Tacotron.
if sys.platform == "win32":
    for p in [r"C:\Program Files\eSpeak NG", r"C:\Program Files (x86)\eSpeak NG"]:
        if os.path.isdir(p) and p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
    for p in [r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
              r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"]:
        if os.path.isfile(p):
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", p)
            break

# =============================================================================
# 1. PATCH KOMPATIBILITAS: transformers 5.x menghapus isin_mps_friendly
# =============================================================================
def apply_compatibility_patches():
    import torch
    try:
        import transformers.pytorch_utils as pu
        if not hasattr(pu, "isin_mps_friendly"):
            def isin_mps_friendly(elements, test_elements):
                return torch.isin(elements, test_elements)
            pu.isin_mps_friendly = isin_mps_friendly
    except Exception as e:
        print(f"[!] Info patch transformers: {e}")

    # torch.load default weights_only=True di torch 2.6+; XTTS perlu False
    try:
        _orig_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = patched_load
    except Exception:
        pass

apply_compatibility_patches()

import torch

# =============================================================================
# 2. KONSTANTA MODEL & BAHASA
# =============================================================================
MODEL_XTTS_V2 = "tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_VITS_EN = "tts_models/en/ljspeech/vits"
MODEL_TACOTRON_EN = "tts_models/en/ljspeech/tacotron2-DDC"
MODEL_VITS_ID = "tts_models/multilingual/multi-dataset/xtts_v2"  # XTTS v2 mendukung ID

XTTS_LANGUAGES = {
    "en": "English (Inggris)",
    "es": "Spanish (Spanyol)",
    "fr": "French (Prancis)",
    "de": "German (Jerman)",
    "it": "Italian (Italia)",
    "pt": "Portuguese (Portugis)",
    "pl": "Polish (Polandia)",
    "tr": "Turkish (Turki)",
    "ru": "Russian (Rusia)",
    "nl": "Dutch (Belanda)",
    "cs": "Czech (Ceko)",
    "ar": "Arabic (Arab)",
    "zh-cn": "Chinese Mandarin",
    "ja": "Japanese (Jepang)",
    "hu": "Hungarian (Hungaria)",
    "ko": "Korean (Korea)",
    "hi": "Hindi (India)",
}

# =============================================================================
# 3. KELAS HELPER
# =============================================================================
class TTSHelper:
    def __init__(self, model_name=None, gpu=None):
        self.model_name = model_name or MODEL_XTTS_V2
        self.gpu = gpu if gpu is not None else torch.cuda.is_available()
        self.device = "cuda" if self.gpu else "cpu"
        print(f"[*] Model dipilih: {self.model_name}")
        print(f"[*] Perangkat     : {self.device.upper()}")
        self.tts = None

    def load(self):
        if self.tts is None:
            print(f"[*] Memuat model {self.model_name} ...")
            from TTS.api import TTS
            self.tts = TTS(model_name=self.model_name, progress_bar=True, gpu=self.gpu)
            print("[+] Model berhasil dimuat!")
        return self.tts

    # ---- FITUR 1: TTS SEDERHANA ----
    def tts_simple(self, text, output_wav="output_tts.wav"):
        tts = self.load()
        print(f"\n[*] Mensintesis: '{text}'")
        if getattr(tts, "is_multi_lingual", False):
            # XTTS perlu language + speaker. Gunakan default jika tidak ada.
            kwargs = {"language": "en"}
            if getattr(tts, "speakers", None):
                kwargs["speaker"] = tts.speakers[0]
            tts.tts_to_file(text=text, file_path=output_wav, **kwargs)
        else:
            tts.tts_to_file(text=text, file_path=output_wav)
        print(f"[+] Audio tersimpan di: {Path(output_wav).resolve()}")
        return output_wav

    # ---- FITUR 2: XTTS v2 MULTILINGUAL (SPEAKER PRESET) ----
    def tts_multilingual(self, text, language="en", output_wav="output_xtts.wav", speaker=None):
        tts = self.load()
        if not getattr(tts, "is_multi_lingual", False):
            raise RuntimeError(f"Model {self.model_name} bukan multilingual. Gunakan {MODEL_XTTS_V2}.")
        if speaker is None and getattr(tts, "speakers", None):
            speaker = tts.speakers[0]
        print(f"\n[*] XTTS ({language}, speaker={speaker}): '{text}'")
        tts.tts_to_file(text=text, file_path=output_wav, language=language, speaker=speaker)
        print(f"[+] Audio tersimpan di: {Path(output_wav).resolve()}")
        return output_wav

    # ---- FITUR 3: VOICE CLONING ----
    def tts_voice_clone(self, text, speaker_wav, language="en", output_wav="output_clone.wav"):
        tts = self.load()
        if not getattr(tts, "is_multi_lingual", False):
            raise RuntimeError(f"Voice cloning hanya didukung model XTTS. Model aktif: {self.model_name}")
        if not os.path.exists(speaker_wav):
            raise FileNotFoundError(f"File audio referensi tidak ditemukan: {speaker_wav}")
        print(f"\n[*] Voice cloning ({language}) dengan referensi: {speaker_wav}")
        print(f"    Teks: '{text}'")
        tts.tts_to_file(
            text=text,
            file_path=output_wav,
            speaker_wav=speaker_wav,
            language=language,
        )
        print(f"[+] Audio tersimpan di: {Path(output_wav).resolve()}")
        return output_wav

    def list_speakers(self):
        tts = self.load()
        spk = getattr(tts, "speakers", None) or []
        return list(spk)


def list_tts_models():
    """Return list of available TTS model names."""
    from TTS.api import TTS
    t = TTS()
    return t.list_models()


def tampilkan_bahasa_xtts():
    print("\n" + "=" * 55)
    print(" DAFTAR BAHASA XTTS v2")
    print("=" * 55)
    items = list(XTTS_LANGUAGES.items())
    for i in range(0, len(items), 2):
        col1 = f"{items[i][0]}: {items[i][1]}"
        col2 = f"{items[i+1][0]}: {items[i+1][1]}" if i + 1 < len(items) else ""
        print(f"  {col1:<28} {col2}")
    print("=" * 55 + "\n")


# =============================================================================
# 4. MENU INTERAKTIF
# =============================================================================
def interactive_menu(helper: TTSHelper):
    while True:
        print("\n" + "=" * 60)
        print("       PROGRAM UJI COBA COQUI TTS")
        print("=" * 60)
        print(" Model      :", helper.model_name)
        print(" Perangkat  :", helper.device.upper())
        print("-" * 60)
        print(" [1] TTS Sederhana (teks -> suara)")
        print(" [2] TTS Multilingual XTTS v2")
        print(" [3] Voice Cloning (XTTS v2 + audio referensi)")
        print(" [4] Lihat daftar speaker preset (XTTS)")
        print(" [5] Lihat daftar bahasa XTTS v2")
        print(" [6] Lihat daftar model TTS")
        print(" [0] Keluar")
        print("=" * 60)

        pilihan = input("Pilih menu [0-6]: ").strip()
        if pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        try:
            if pilihan == "1":
                text = input("Teks: ").strip()
                if not text:
                    print("[!] Teks tidak boleh kosong.")
                    continue
                out = input("Nama file output [default output_tts.wav]: ").strip() or "output_tts.wav"
                helper.tts_simple(text, output_wav=out)
            elif pilihan == "2":
                text = input("Teks: ").strip()
                if not text:
                    print("[!] Teks tidak boleh kosong.")
                    continue
                lang = input("Kode bahasa [default en]: ").strip() or "en"
                out = input("Nama file output [default output_xtts.wav]: ").strip() or "output_xtts.wav"
                helper.tts_multilingual(text, language=lang, output_wav=out)
            elif pilihan == "3":
                text = input("Teks: ").strip()
                if not text:
                    print("[!] Teks tidak boleh kosong.")
                    continue
                ref = input("Path audio referensi (.wav): ").strip(' "\'')
                lang = input("Kode bahasa [default en]: ").strip() or "en"
                out = input("Nama file output [default output_clone.wav]: ").strip() or "output_clone.wav"
                helper.tts_voice_clone(text, speaker_wav=ref, language=lang, output_wav=out)
            elif pilihan == "4":
                spk = helper.list_speakers()
                if not spk:
                    print("[!] Model aktif tidak memiliki speaker preset.")
                else:
                    print(f"[+] {len(spk)} speaker preset tersedia (menampilkan 20 pertama):")
                    for s in spk[:20]:
                        print("  -", s)
            elif pilihan == "5":
                tampilkan_bahasa_xtts()
            elif pilihan == "6":
                models = list_tts_models()
                print(f"[+] {len(models)} model tersedia (menampilkan 20 pertama):")
                for m in models[:20]:
                    print("  -", m)
            else:
                print("[!] Pilihan tidak valid.")
        except Exception as e:
            print(f"[!] Kesalahan: {e}")


# =============================================================================
# 5. ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Uji coba Coqui TTS.")
    parser.add_argument("--mode", choices=["menu", "simple", "xtts", "clone", "list"], default="menu")
    parser.add_argument("--text", type=str, default="Hello, this is a test of Coqui TTS.")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--speaker-wav", type=str, default="", help="Audio referensi untuk voice cloning")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--model", type=str, default=None, help="Nama model TTS (default XTTS v2)")

    args = parser.parse_args()
    helper = TTSHelper(model_name=args.model)

    if args.mode == "simple":
        helper.tts_simple(args.text, output_wav=args.output)
    elif args.mode == "xtts":
        helper.tts_multilingual(args.text, language=args.lang, output_wav=args.output)
    elif args.mode == "clone":
        if not args.speaker_wav:
            print("[!] Perlu --speaker-wav <path>")
            sys.exit(1)
        helper.tts_voice_clone(args.text, speaker_wav=args.speaker_wav, language=args.lang, output_wav=args.output)
    elif args.mode == "list":
        for m in list_tts_models():
            print(m)
    else:
        interactive_menu(helper)


if __name__ == "__main__":
    main()
