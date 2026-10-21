"""
=============================================================================
 Skrip Pengujian Model Meta SeamlessM4T (v2 Large / Medium)
 Proyek: Percobaan Generated
=============================================================================
Skrip ini dirancang untuk mencoba model SeamlessM4T dengan fitur lengkap:
1. Text-to-Text Translation (T2TT)       : Menerjemahkan teks antar bahasa
2. Text-to-Speech Translation (T2ST)     : Teks -> Suara (file audio .wav)
3. Speech-to-Text Translation (S2TT/ASR) : Audio -> Transkripsi/Terjemahan teks
4. Speech-to-Speech Translation (S2ST)   : Audio sumber -> Audio terjemahan

Fitur Tambahan:
- Kompatibilitas Windows: Patch memory-mapping dan C++ allocator untuk mencegah
  error 'paging file is too small' (os error 1455) & access violation torch_cpu.dll.
- Deteksi model lokal otomatis di folder 'Model SeamlessM4T v2'.
- Menu interaktif di terminal atau dapat dipanggil via baris perintah (CLI).
=============================================================================
"""

import os
import sys
import json
import struct
import argparse
import threading
import warnings
from pathlib import Path

# Fix encoding output di Windows agar bisa print karakter Unicode (Jepang, dll.)
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Sembunyikan peringatan yang tidak relevan
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =============================================================================
# 1. PATCH & OPTIMISASI WINDOWS / PYTORCH CPU
# =============================================================================
import numpy as np
import torch

DTYPE_MAP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": bool,
}

class FastFileSlice:
    """Pembaca slice tensor on-demand langsung dari file biner safetensors tanpa mmap."""
    def __init__(self, fh, lock, offset, length, shape, dtype):
        self.fh = fh
        self.lock = lock
        self.offset = offset
        self.length = length
        self.shape = shape
        self.dtype = dtype

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype

    def __getitem__(self, item):
        # transformers 5.x memuat bobot secara paralel; seek+read harus atomik
        with self.lock:
            self.fh.seek(self.offset)
            raw_bytes = self.fh.read(self.length)
        np_dtype = DTYPE_MAP.get(self.dtype, np.float32)
        arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(self.shape)
        if item is not Ellipsis and item != slice(None):
            arr = arr[item]
        if self.dtype == "BF16":
            t = torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
        else:
            t = torch.from_numpy(arr.copy())
        return t

class FastFileSafeOpen:
    """Pengganti safe_open bawaan untuk menghindari crash OS error 1455 di Windows."""
    def __init__(self, filename, framework="pt", device="cpu", **kwargs):
        self.fh = open(filename, "rb")
        self.lock = threading.Lock()
        header_len = struct.unpack("<Q", self.fh.read(8))[0]
        header_bytes = self.fh.read(header_len)
        self.header = json.loads(header_bytes.decode("utf-8"))
        self.base_offset = 8 + header_len
        self._keys = [k for k in self.header.keys() if k != "__metadata__"]

    def keys(self):
        return self._keys

    def get_slice(self, key):
        info = self.header[key]
        start, end = info["data_offsets"]
        return FastFileSlice(
            self.fh,
            self.lock,
            self.base_offset + start,
            end - start,
            info["shape"],
            info["dtype"]
        )

    def get_tensor(self, key):
        return self.get_slice(key)[:]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fh.close()

def apply_compatibility_patches():
    """Terapkan perbaikan kompatibilitas sebelum memuat transformers."""
    try:
        import safetensors
        import transformers.modeling_utils
        safetensors.safe_open = FastFileSafeOpen
        transformers.modeling_utils.safe_open = FastFileSafeOpen
    except Exception as e:
        print(f"[!] Info patch safetensors: {e}")

    try:
        from transformers.models.seamless_m4t_v2 import modeling_seamless_m4t_v2 as m
        orig_dec_init = m.SeamlessM4Tv2TextToUnitDecoder.__init__
        def patched_t2u_dec_init(self, *args, **kwargs):
            orig_dec_init(self, *args, **kwargs)
            for i, layer in enumerate(self.layers):
                layer.self_attn.layer_idx = i
        m.SeamlessM4Tv2TextToUnitDecoder.__init__ = patched_t2u_dec_init
    except Exception:
        pass

apply_compatibility_patches()

# =============================================================================
# 2. DAFTAR KODE BAHASA SEAMLESSM4T
# =============================================================================
POPULAR_LANGUAGES = {
    "ind": "Bahasa Indonesia",
    "eng": "English (Inggris)",
    "jav": "Basa Jawa (Javanese)",
    "zho": "Mandarin (Cina)",
    "jpn": "Japanese (Jepang)",
    "kor": "Korean (Korea)",
    "arb": "Arabic (Arab)",
    "fra": "Français (Prancis)",
    "deu": "Deutsch (Jerman)",
    "spa": "Español (Spanyol)",
    "rus": "Russian (Rusia)",
    "tha": "Thai (Thailand)",
    "vie": "Vietnamese (Vietnam)",
    "msa": "Bahasa Melayu",
    "hin": "Hindi (India)",
    "nld": "Dutch (Belanda)",
    "ita": "Italiano (Italia)",
    "por": "Português (Portugis)",
    "tur": "Türkçe (Turki)"
}

def tampilkan_daftar_bahasa():
    print("\n" + "=" * 55)
    print(" DAFTAR KODE BAHASA POPULER (3 Huruf)")
    print("=" * 55)
    items = list(POPULAR_LANGUAGES.items())
    for i in range(0, len(items), 2):
        col1 = f"{items[i][0]}: {items[i][1]}"
        col2 = f"{items[i+1][0]}: {items[i+1][1]}" if i + 1 < len(items) else ""
        print(f"  {col1:<28} {col2}")
    print("=" * 55)
    print(" Catatan: Bahasa Jawa ('jav') didukung untuk teks.")
    print("=" * 55 + "\n")

# =============================================================================
# 3. KELAS PENGELOLA MODEL (MODEL RUNNER)
# =============================================================================
class SeamlessM4THelper:
    def __init__(self, model_path_or_name=None):
        base_dir = Path(__file__).resolve().parent
        local_model_dir = base_dir / "Model SeamlessM4T v2"

        if model_path_or_name is None:
            if local_model_dir.exists() and (local_model_dir / "config.json").exists():
                self.model_path = str(local_model_dir)
                print(f"[*] Menggunakan model lokal: {self.model_path}")
            else:
                self.model_path = "facebook/seamless-m4t-v2-large"
                print(f"[*] Model lokal tidak ditemukan, akan memuat dari Hub: {self.model_path}")
        else:
            self.model_path = model_path_or_name
            print(f"[*] Jalur model disetel ke: {self.model_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Menggunakan perangkat: {self.device.upper()}")
        if self.device == "cpu":
            num_threads = max(1, os.cpu_count() - 2)
            torch.set_num_threads(num_threads)
            print(f"[*] Jumlah CPU Threads diatur ke: {num_threads}")

        self.processor = None
        self.model = None

    def load_processor(self):
        """Memuat processor dan tokenizer."""
        if self.processor is None:
            print("[*] Sedang memuat Processor & Tokenizer...")
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            print("[+] Processor berhasil dimuat!")
        return self.processor

    def load_model(self):
        """Memuat model SeamlessM4Tv2."""
        if self.model is None:
            self.load_processor()
            print("[*] Sedang memuat bobot model SeamlessM4T (bisa memakan waktu beberapa detik)...")
            from transformers import SeamlessM4Tv2Model
            self.model = SeamlessM4Tv2Model.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            self.model.eval()
            print("[+] Model berhasil dimuat ke memori!")
        return self.model

    # -------------------------------------------------------------------------
    # FITUR 1: TEXT-TO-TEXT TRANSLATION (T2TT)
    # -------------------------------------------------------------------------
    def text_to_text(self, text, src_lang="ind", tgt_lang="eng"):
        """Menerjemahkan teks dari bahasa sumber ke bahasa target."""
        processor = self.load_processor()
        model = self.load_model()

        print(f"\n[*] Menerjemahkan ({src_lang} -> {tgt_lang}): '{text}'")
        text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model.generate(
                **text_inputs,
                tgt_lang=tgt_lang,
                generate_speech=False
            )

        if hasattr(output, "sequences"):
            out_tokens = output.sequences
        elif isinstance(output, tuple):
            out_tokens = output[0]
        else:
            out_tokens = output

        if hasattr(out_tokens, "ndim") and out_tokens.ndim > 1:
            token_list = out_tokens[0].tolist()
        elif hasattr(out_tokens, "tolist"):
            token_list = out_tokens.tolist()
        else:
            token_list = list(out_tokens)

        translated = processor.decode(token_list, skip_special_tokens=True)
        print(f"[+] Hasil Terjemahan: {translated}")
        return translated

    # -------------------------------------------------------------------------
    # FITUR 2: TEXT-TO-SPEECH TRANSLATION (T2ST)
    # -------------------------------------------------------------------------
    def text_to_speech(self, text, src_lang="ind", tgt_lang="eng", output_wav="output_t2st.wav", speaker_id=0):
        """Mengubah teks menjadi file suara bicara (.wav)."""
        processor = self.load_processor()
        model = self.load_model()

        print(f"\n[*] Mensintesis suara ({src_lang} -> {tgt_lang}): '{text}'")
        text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = model.generate(
                **text_inputs,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id
            )

        # Ekstrak audio waveform
        if hasattr(out, "waveform"):
            audio_arr = out.waveform[0].cpu().numpy().squeeze()
        elif isinstance(out, tuple):
            audio_arr = out[0][0].cpu().numpy().squeeze()
        else:
            audio_arr = out[0].cpu().numpy().squeeze()

        sample_rate = getattr(model.config, "sampling_rate", 16000)

        import scipy.io.wavfile as wavfile
        wavfile.write(output_wav, rate=sample_rate, data=audio_arr)
        print(f"[+] Audio suara berhasil disimpan di: {Path(output_wav).resolve()}")
        print(f"[+] Sample rate: {sample_rate} Hz")
        return output_wav

    # -------------------------------------------------------------------------
    # FITUR 3: SPEECH-TO-TEXT TRANSLATION (S2TT / ASR)
    # -------------------------------------------------------------------------
    def speech_to_text(self, audio_file_path, tgt_lang="eng"):
        """Mentranskripsikan / menerjemahkan file audio ke teks."""
        import soundfile as sf
        import torchaudio

        processor = self.load_processor()
        model = self.load_model()

        if not os.path.exists(audio_file_path):
            print(f"[!] File audio tidak ditemukan: {audio_file_path}")
            return ""

        print(f"\n[*] Membaca file audio: {audio_file_path}")
        audio_data, orig_sr = sf.read(audio_file_path)

        # Konversi ke mono jika stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample ke 16000 Hz jika diperlukan
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            tensor_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
            audio_data = resampler(tensor_audio).squeeze(0).numpy()

        audio_inputs = processor(audio=audio_data, return_tensors="pt", sampling_rate=16000).to(self.device)

        print(f"[*] Mentranskripsi/menerjemahkan audio ke bahasa: {tgt_lang}...")
        with torch.no_grad():
            output = model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang,
                generate_speech=False
            )

        if hasattr(output, "sequences"):
            out_tokens = output.sequences
        elif isinstance(output, tuple):
            out_tokens = output[0]
        else:
            out_tokens = output
        transcription = processor.decode(out_tokens[0].tolist(), skip_special_tokens=True)
        print(f"[+] Hasil Transkripsi/Terjemahan: {transcription}")
        return transcription

    # -------------------------------------------------------------------------
    # FITUR 4: SPEECH-TO-SPEECH TRANSLATION (S2ST)
    # -------------------------------------------------------------------------
    def speech_to_speech(self, audio_file_path, tgt_lang="eng", output_wav="output_s2st.wav", speaker_id=0):
        """Menerjemahkan audio input langsung menjadi audio bahasa lain."""
        import soundfile as sf
        import torchaudio

        processor = self.load_processor()
        model = self.load_model()

        if not os.path.exists(audio_file_path):
            print(f"[!] File audio tidak ditemukan: {audio_file_path}")
            return ""

        print(f"\n[*] Membaca file audio input: {audio_file_path}")
        audio_data, orig_sr = sf.read(audio_file_path)

        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            tensor_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
            audio_data = resampler(tensor_audio).squeeze(0).numpy()

        audio_inputs = processor(audio=audio_data, return_tensors="pt", sampling_rate=16000).to(self.device)

        print(f"[*] Menerjemahkan audio ke audio bahasa ({tgt_lang})...")
        with torch.no_grad():
            out = model.generate(
                **audio_inputs,
                tgt_lang=tgt_lang,
                speaker_id=speaker_id
            )

        if hasattr(out, "waveform"):
            audio_arr = out.waveform[0].cpu().numpy().squeeze()
        elif isinstance(out, tuple):
            audio_arr = out[0][0].cpu().numpy().squeeze()
        else:
            audio_arr = out[0].cpu().numpy().squeeze()

        sample_rate = getattr(model.config, "sampling_rate", 16000)
        import scipy.io.wavfile as wavfile
        wavfile.write(output_wav, rate=sample_rate, data=audio_arr)
        print(f"[+] Audio hasil terjemahan disimpan di: {Path(output_wav).resolve()}")
        return output_wav

    # -------------------------------------------------------------------------
    # FITUR 5: DEMO CEPAT (QUICK TEST)
    # -------------------------------------------------------------------------
    def run_quick_demo(self):
        """Menjalankan pengujian otomatis untuk memverifikasi model."""
        print("\n" + "=" * 60)
        print("          MEMULAI PENGUJIAN OTOMATIS (DEMO)")
        print("=" * 60)
        sample_text = "Halo, selamat datang! Ini adalah pengujian model SeamlessM4T."
        print(f"Teks Uji Coba: \"{sample_text}\"")
        
        # 1. Text-to-Text: Indonesia -> Inggris
        print("\n[Langkah 1/2] Text-to-Text Translation (Indonesia -> Inggris)...")
        hasil_t2tt = self.text_to_text(sample_text, src_lang="ind", tgt_lang="eng")

        # 2. Text-to-Text: Indonesia -> Jepang
        print("\n[Langkah 2/2] Text-to-Text Translation (Indonesia -> Jepang)...")
        self.text_to_text(sample_text, src_lang="ind", tgt_lang="jpn")

        print("\n" + "=" * 60)
        print(" [+] DEMO SELESAI DENGAN SUKSES!")
        print("=" * 60 + "\n")


# =============================================================================
# 4. MENU TERMINAL INTERAKTIF
# =============================================================================
def interactive_menu(helper: SeamlessM4THelper):
    while True:
        print("\n" + "=" * 60)
        print("       PROGRAM UJI COBA MODEL SEAMLESSM4T (META AI)")
        print("=" * 60)
        print(" Lokasi Model :", helper.model_path)
        print(" Perangkat    :", helper.device.upper())
        print("-" * 60)
        print(" [1] Text-to-Text Translation (T2TT) - Terjemah Teks")
        print(" [2] Text-to-Speech (T2ST)          - Teks ke Suara (.wav)")
        print(" [3] Speech-to-Text (S2TT/ASR)       - Audio ke Teks")
        print(" [4] Speech-to-Speech (S2ST)         - Terjemah Suara ke Suara")
        print(" [5] Jalankan Demo Otomatis Cepat")
        print(" [6] Tampilkan Daftar Kode Bahasa")
        print(" [0] Keluar")
        print("=" * 60)

        pilihan = input("Pilih menu [0-6]: ").strip()

        if pilihan == "0":
            print("\nTerima kasih telah menggunakan skrip ini. Sampai jumpa!")
            break
        elif pilihan == "1":
            text = input("\nMasukkan teks yang ingin diterjemahkan: ").strip()
            if not text:
                print("[!] Teks tidak boleh kosong.")
                continue
            src = input("Kode bahasa sumber [default 'ind']: ").strip() or "ind"
            tgt = input("Kode bahasa tujuan [default 'eng']: ").strip() or "eng"
            try:
                helper.text_to_text(text, src_lang=src, tgt_lang=tgt)
            except Exception as e:
                print(f"[!] Terjadi kesalahan: {e}")
        elif pilihan == "2":
            text = input("\nMasukkan teks untuk disintesis menjadi suara: ").strip()
            if not text:
                print("[!] Teks tidak boleh kosong.")
                continue
            src = input("Kode bahasa teks asal [default 'ind']: ").strip() or "ind"
            tgt = input("Kode bahasa suara output [default 'eng']: ").strip() or "eng"
            out_file = input("Nama file output .wav [default 'output_t2st.wav']: ").strip() or "output_t2st.wav"
            try:
                helper.text_to_speech(text, src_lang=src, tgt_lang=tgt, output_wav=out_file)
            except Exception as e:
                print(f"[!] Terjadi kesalahan: {e}")
        elif pilihan == "3":
            audio_path = input("\nMasukkan path file audio (.wav / .mp3): ").strip(' "\'')
            if not audio_path:
                print("[!] Path audio tidak boleh kosong.")
                continue
            tgt = input("Kode bahasa teks hasil transkripsi [default 'eng']: ").strip() or "eng"
            try:
                helper.speech_to_text(audio_path, tgt_lang=tgt)
            except Exception as e:
                print(f"[!] Terjadi kesalahan: {e}")
        elif pilihan == "4":
            audio_path = input("\nMasukkan path file audio input: ").strip(' "\'')
            if not audio_path:
                print("[!] Path audio tidak boleh kosong.")
                continue
            tgt = input("Kode bahasa audio tujuan [default 'eng']: ").strip() or "eng"
            out_file = input("Nama file audio output [default 'output_s2st.wav']: ").strip() or "output_s2st.wav"
            try:
                helper.speech_to_speech(audio_path, tgt_lang=tgt, output_wav=out_file)
            except Exception as e:
                print(f"[!] Terjadi kesalahan: {e}")
        elif pilihan == "5":
            helper.run_quick_demo()
        elif pilihan == "6":
            tampilkan_daftar_bahasa()
        else:
            print("[!] Pilihan tidak valid, silakan coba lagi.")


# =============================================================================
# 5. ENTRY POINT & PARSER ARGUMEN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Skrip untuk mencoba model Meta SeamlessM4T (T2TT, T2ST, S2TT, S2ST)."
    )
    parser.add_argument("--mode", choices=["menu", "t2tt", "t2st", "s2tt", "s2st", "demo"], default="menu",
                        help="Mode eksekusi (menu, t2tt, t2st, s2tt, s2st, demo)")
    parser.add_argument("--text", type=str, default="Halo, apa kabar?", help="Teks input untuk T2TT atau T2ST")
    parser.add_argument("--audio", type=str, default="", help="Path file audio untuk S2TT atau S2ST")
    parser.add_argument("--src", type=str, default="ind", help="Kode bahasa sumber (default: ind)")
    parser.add_argument("--tgt", type=str, default="eng", help="Kode bahasa target (default: eng)")
    parser.add_argument("--output", type=str, default="output.wav", help="Nama file audio output .wav")
    parser.add_argument("--model", type=str, default=None, help="Jalur folder model lokal atau ID HuggingFace")

    args = parser.parse_args()

    helper = SeamlessM4THelper(model_path_or_name=args.model)

    if args.mode == "demo":
        helper.run_quick_demo()
    elif args.mode == "t2tt":
        helper.text_to_text(args.text, src_lang=args.src, tgt_lang=args.tgt)
    elif args.mode == "t2st":
        helper.text_to_speech(args.text, src_lang=args.src, tgt_lang=args.tgt, output_wav=args.output)
    elif args.mode == "s2tt":
        if not args.audio:
            print("[!] Harap sertakan --audio <path_file_audio>")
            sys.exit(1)
        helper.speech_to_text(args.audio, tgt_lang=args.tgt)
    elif args.mode == "s2st":
        if not args.audio:
            print("[!] Harap sertakan --audio <path_file_audio>")
            sys.exit(1)
        helper.speech_to_speech(args.audio, tgt_lang=args.tgt, output_wav=args.output)
    else:
        # Default: Mode menu interaktif
        interactive_menu(helper)

if __name__ == "__main__":
    main()
