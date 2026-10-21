"""
benchmark.py — Otomatis uji 6 model AI dan ekspor hasil ke CSV
Jalankan: python benchmark.py
Output  : folder hasil_benchmark/ berisi 6 file CSV + ringkasan
"""

import sys
import os
import csv
import time
import warnings
import tempfile
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
HASIL = ROOT / "hasil_benchmark"
HASIL.mkdir(exist_ok=True)

# Subfolder audio per model — dibuat saat pertama dibutuhkan
AUDIO_DIRS = {
    "SeamlessM4T": HASIL / "SeamlessM4T",
    "CoquiTTS":    HASIL / "CoquiTTS",
    "MusicGen":    HASIL / "MusicGen",
    "AudioLDM":    HASIL / "AudioLDM",
}

def audio_path(model_key: str, nama_file: str) -> str:
    folder = AUDIO_DIRS[model_key]
    folder.mkdir(exist_ok=True)
    return str(folder / nama_file)

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: tulis CSV
# ─────────────────────────────────────────────────────────────────────────────

def tulis_csv(nama_file: str, baris: list[dict], fieldnames: list[str]):
    path = HASIL / nama_file
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(baris)
    print(f"  [+] CSV tersimpan: {path}")
    return path


def baris_kosong(fields, **isi):
    row = {f: "" for f in fields}
    row.update(isi)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEAMLESSM4T
# ─────────────────────────────────────────────────────────────────────────────

SEAMLESS_FIELDS = [
    "ID_Uji", "Mode_Fitur", "Src_Lang", "Tgt_Lang",
    "Input_Teks_Transkrip_Audio_Asal", "Panjang",
    "Durasi_Audio_s", "Latensi_s", "RTF",
    "Hasil_Terjemahan", "BLEU_Score", "Skor_Makna_1_5",
    "Status", "Catatan_Kualitas_Analisis"
]

# (mode, src, tgt, input_teks, referensi, panjang_kategori)
# Panjang: Pendek=1-5 kata, Sedang=6-15 kata, Panjang=16+ kata
# Mode T2TT dan T2ST: input teks → hasil teks / audio
# Mode S2TT dan S2ST: input audio (dibuat via TTS placeholder) → hasil teks / audio
SEAMLESS_KASUS = [
    # ── T2TT (Text-to-Text Translation) ──────────────────────────────────────
    ("T2TT", "ind", "eng", "Terima kasih.",                                                                          "Thank you.",                                                                                    "Pendek"),
    ("T2TT", "ind", "eng", "Selamat pagi, apa kabar?",                                                              "Good morning, how are you?",                                                                    "Pendek"),
    ("T2TT", "ind", "eng", "Saya sedang belajar kecerdasan buatan.",                                                 "I am studying artificial intelligence.",                                                        "Sedang"),
    ("T2TT", "ind", "eng", "Teknologi kecerdasan buatan berkembang sangat pesat dalam dekade terakhir.",             "Artificial intelligence technology has developed very rapidly in the last decade.",              "Sedang"),
    ("T2TT", "ind", "eng", "Universitas ini memiliki fasilitas yang sangat lengkap dan modern untuk mendukung kegiatan penelitian mahasiswa tingkat akhir.", "This university has very complete and modern facilities to support final-year student research activities.", "Panjang"),
    ("T2TT", "eng", "ind", "Hello.",                                                                                 "Halo.",                                                                                         "Pendek"),
    ("T2TT", "eng", "ind", "The weather is very nice today.",                                                        "Cuaca sangat bagus hari ini.",                                                                  "Sedang"),
    ("T2TT", "eng", "ind", "Machine learning is a subset of artificial intelligence used to train models on data.",  "Pembelajaran mesin adalah bagian dari kecerdasan buatan yang digunakan untuk melatih model pada data.", "Panjang"),
    ("T2TT", "ind", "jpn", "Selamat datang.",                                                                        "いらっしゃいませ。",                                                                                "Pendek"),
    ("T2TT", "ind", "jpn", "Hari ini cuaca sangat cerah dan menyenangkan.",                                          "今日はとても晴れていて気持ちがいいです。",                                                              "Sedang"),
    ("T2TT", "ind", "fra", "Sampai jumpa besok.",                                                                    "À demain.",                                                                                     "Pendek"),
    ("T2TT", "ind", "fra", "Penelitian ini membahas penerapan model AI untuk pemrosesan bahasa alami.",              "Cette recherche traite de l'application des modèles d'IA au traitement du langage naturel.",      "Sedang"),
    ("T2TT", "eng", "arb", "Where is the train station?",                                                           "أين محطة القطار؟",                                                                             "Pendek"),
    ("T2TT", "eng", "arb", "Artificial intelligence is transforming many industries around the world.",              "الذكاء الاصطناعي يحول كثيراً من الصناعات حول العالم.",                                          "Sedang"),
    ("T2TT", "fra", "eng", "Bonjour, comment allez-vous aujourd'hui?",                                               "Hello, how are you today?",                                                                     "Sedang"),
    # ── T2ST (Text-to-Speech Translation) ────────────────────────────────────
    ("T2ST", "ind", "eng", "Halo.",                                                                                  "Hello.",                                                                                        "Pendek"),
    ("T2ST", "ind", "eng", "Selamat malam, semoga harimu menyenangkan.",                                             "Good evening, I hope your day was pleasant.",                                                   "Sedang"),
    ("T2ST", "ind", "eng", "Sistem pengenalan suara ini dirancang untuk mendukung berbagai bahasa di seluruh dunia.", "This speech recognition system is designed to support various languages around the world.",     "Panjang"),
    ("T2ST", "eng", "ind", "Thank you very much.",                                                                   "Terima kasih banyak.",                                                                          "Pendek"),
    ("T2ST", "eng", "jpn", "Nice to meet you.",                                                                      "はじめまして。",                                                                                "Pendek"),
    # ── S2TT (Speech-to-Text Translation) ────────────────────────────────────
    ("S2TT", "ind", "eng", "Terima kasih banyak.",                                                                   "Thank you very much.",                                                                          "Pendek"),
    ("S2TT", "ind", "eng", "Saya ingin memesan makanan untuk makan malam.",                                          "I would like to order food for dinner.",                                                        "Sedang"),
    ("S2TT", "ind", "eng", "Model kecerdasan buatan ini dilatih menggunakan dataset audio yang sangat besar sehingga mampu mengenali berbagai aksen.", "This AI model was trained using a very large audio dataset enabling it to recognize various accents.", "Panjang"),
    ("S2TT", "eng", "ind", "Hello, my name is Ahmad.",                                                               "Halo, nama saya Ahmad.",                                                                        "Pendek"),
    ("S2TT", "eng", "fra", "What time does the meeting start?",                                                      "À quelle heure commence la réunion?",                                                          "Sedang"),
    # ── S2ST (Speech-to-Speech Translation) ──────────────────────────────────
    ("S2ST", "ind", "eng", "Selamat pagi.",                                                                          "Good morning.",                                                                                 "Pendek"),
    ("S2ST", "ind", "eng", "Apa yang bisa saya bantu untuk kamu hari ini?",                                          "What can I help you with today?",                                                              "Sedang"),
    ("S2ST", "eng", "ind", "Please turn off the lights.",                                                            "Tolong matikan lampunya.",                                                                      "Pendek"),
    ("S2ST", "eng", "jpn", "I love Japanese food and culture.",                                                      "私は日本食と文化が大好きです。",                                                                    "Sedang"),
    ("S2ST", "ind", "arb", "Assalamualaikum warahmatullahi wabarakatuh.",                                            "السلام عليكم ورحمة الله وبركاته.",                                                             "Sedang"),
]


def _klasifikasi_panjang(teks: str) -> str:
    n = len(teks.split())
    if n <= 5:
        return "Pendek"
    elif n <= 15:
        return "Sedang"
    return "Panjang"


def _bleu_sederhana(referensi: str, hipotesis: str) -> float:
    """Unigram BLEU tanpa library eksternal (untuk indikasi kasar)."""
    if not referensi or not hipotesis:
        return 0.0
    import math
    ref_words = referensi.lower().split()
    hyp_words = hipotesis.lower().split()
    if not hyp_words:
        return 0.0
    ref_count = {}
    for w in ref_words:
        ref_count[w] = ref_count.get(w, 0) + 1
    match = 0
    for w in hyp_words:
        if ref_count.get(w, 0) > 0:
            match += 1
            ref_count[w] -= 1
    precision = match / len(hyp_words)
    bp = min(1.0, math.exp(1 - len(ref_words) / max(len(hyp_words), 1)))
    return round(bp * precision * 100, 2)


def _buat_audio_tts_placeholder(teks: str, lang: str) -> tuple[str, float]:
    """Buat file audio placeholder via TTS untuk mode S2TT / S2ST."""
    import numpy as np
    import scipy.io.wavfile as wv

    # Coba gTTS dulu (ringan, tidak perlu model besar)
    try:
        from gtts import gTTS
        out = tempfile.mktemp(suffix=".mp3")
        lang_gtts = "id" if lang == "ind" else lang[:2]
        gTTS(text=teks, lang=lang_gtts, slow=False).save(out)
        # Hitung durasi: estimasi dari panjang teks (gTTS ~2.5 kata/detik)
        durasi_est = round(len(teks.split()) / 2.5, 2)
        return out, durasi_est
    except Exception:
        pass

    # Fallback: silence WAV (agar pipeline tetap jalan, WER jadi 100%)
    out = tempfile.mktemp(suffix=".wav")
    sr = 16000
    durasi = max(2.0, len(teks.split()) * 0.4)
    data = np.zeros(int(sr * durasi), dtype=np.int16)
    wv.write(out, sr, data)
    return out, round(durasi, 2)


def benchmark_seamless():
    print("\n[1/6] Menguji SeamlessM4T v2...")
    try:
        from coba_seamlessm4t import SeamlessM4THelper
        h = SeamlessM4THelper()
    except Exception as e:
        print(f"  [!] Gagal load model: {e}")
        return

    baris = []
    for i, (mode, src, tgt, input_teks, referensi, panjang) in enumerate(SEAMLESS_KASUS, 1):
        id_uji = f"SM-{i:03d}"
        print(f"  [{i:02d}/{len(SEAMLESS_KASUS)}] {mode} {src}→{tgt} [{panjang}]: {input_teks[:35]}...")
        try:
            durasi_audio = ""
            rtf = ""

            t0 = time.perf_counter()

            nama_file_audio = f"{id_uji}_{mode}_{src}_{tgt}.wav"
            if mode == "T2TT":
                hasil_teks = h.text_to_text(input_teks, src_lang=src, tgt_lang=tgt)
                hasil_output = hasil_teks
            elif mode == "T2ST":
                out_wav = audio_path("SeamlessM4T", nama_file_audio)
                wav_path = h.text_to_speech(input_teks, src_lang=src, tgt_lang=tgt, output_wav=out_wav)
                hasil_output = f"[audio: {wav_path}]"
                try:
                    import scipy.io.wavfile as wv
                    sr_r, d_r = wv.read(wav_path)
                    durasi_audio = round(len(d_r) / sr_r, 3)
                except Exception:
                    pass
            elif mode == "S2TT":
                in_wav, durasi_audio = _buat_audio_tts_placeholder(input_teks, src)
                hasil_teks = h.speech_to_text(in_wav, tgt_lang=tgt)
                hasil_output = hasil_teks if isinstance(hasil_teks, str) else str(hasil_teks)
            elif mode == "S2ST":
                in_wav, durasi_audio = _buat_audio_tts_placeholder(input_teks, src)
                out_wav = audio_path("SeamlessM4T", nama_file_audio)
                wav_path = h.speech_to_speech(in_wav, tgt_lang=tgt, output_wav=out_wav)
                hasil_output = f"[audio: {wav_path}]"
                try:
                    import scipy.io.wavfile as wv
                    sr_r, d_r = wv.read(wav_path)
                    durasi_audio = round(len(d_r) / sr_r, 3)
                except Exception:
                    pass
            else:
                hasil_output = ""

            latensi = round(time.perf_counter() - t0, 3)
            if durasi_audio and float(durasi_audio) > 0:
                rtf = round(latensi / float(durasi_audio), 4)

            # BLEU hanya untuk output teks
            bleu = ""
            if mode in ("T2TT", "S2TT") and referensi and not hasil_output.startswith("[audio"):
                bleu = _bleu_sederhana(referensi, hasil_output)

            baris.append(baris_kosong(SEAMLESS_FIELDS,
                ID_Uji=id_uji, Mode_Fitur=mode, Src_Lang=src, Tgt_Lang=tgt,
                Input_Teks_Transkrip_Audio_Asal=input_teks, Panjang=panjang,
                Durasi_Audio_s=durasi_audio, Latensi_s=latensi, RTF=rtf,
                Hasil_Terjemahan=hasil_output, BLEU_Score=bleu,
                Status="OK", Catatan_Kualitas_Analisis=""))
        except Exception as e:
            baris.append(baris_kosong(SEAMLESS_FIELDS,
                ID_Uji=id_uji, Mode_Fitur=mode, Src_Lang=src, Tgt_Lang=tgt,
                Input_Teks_Transkrip_Audio_Asal=input_teks, Panjang=panjang,
                Status="ERROR", Catatan_Kualitas_Analisis=str(e)))

    tulis_csv(f"01_SeamlessM4T_{TIMESTAMP}.csv", baris, SEAMLESS_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COQUI TTS
# ─────────────────────────────────────────────────────────────────────────────

TTS_FIELDS = [
    "No", "Model_TTS", "Bahasa", "Teks_Input", "Panjang_Teks_kata",
    "Durasi_Audio_detik", "Sample_Rate_Hz", "Waktu_Generate_detik",
    "RTF", "Naturalness_1_5", "Keterangan"
]

TTS_KASUS = [
    # (model, lang, teks)
    ("tts_models/en/ljspeech/vits", "en", "Hello, welcome to our demonstration."),
    ("tts_models/en/ljspeech/vits", "en", "The quick brown fox jumps over the lazy dog."),
    ("tts_models/en/ljspeech/vits", "en", "Artificial intelligence is transforming the world."),
    ("tts_models/en/ljspeech/vits", "en", "Today is a beautiful day with clear blue skies and warm sunshine."),
    ("tts_models/en/ljspeech/vits", "en", "I am testing a text to speech system for my final college project."),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "en", "Hello, this is a multilingual speech synthesis test."),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "es", "Hola, bienvenido a nuestra demostración."),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "fr", "Bonjour, comment allez-vous aujourd'hui?"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "de", "Guten Morgen, wie geht es Ihnen?"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "ja", "こんにちは、今日はいい天気ですね。"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "zh-cn", "你好，欢迎来到我们的演示。"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "ko", "안녕하세요, 오늘 날씨가 좋네요."),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "hi", "नमस्ते, आप कैसे हैं?"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "pt", "Olá, como você está hoje?"),
    ("tts_models/multilingual/multi-dataset/xtts_v2", "ru", "Привет, как ты себя чувствуешь?"),
]


def benchmark_tts():
    print("\n[2/6] Menguji Coqui TTS...")
    try:
        from coba_tts import TTSHelper, MODEL_VITS_EN, MODEL_XTTS_V2
        import scipy.io.wavfile as wav
    except Exception as e:
        print(f"  [!] Gagal import: {e}")
        return

    # Load model sekali per jenis (VITS dan XTTS)
    helpers = {}
    for model_const in [MODEL_VITS_EN, MODEL_XTTS_V2]:
        try:
            helpers[model_const] = TTSHelper(model_name=model_const)
            helpers[model_const].load()
        except Exception as e:
            print(f"  [!] Gagal load {model_const}: {e}")

    baris = []
    for i, (model, lang, teks) in enumerate(TTS_KASUS, 1):
        model_pendek = model.split("/")[-1]
        print(f"  [{i:02d}/{len(TTS_KASUS)}] {model_pendek} [{lang}]: {teks[:40]}...")
        try:
            h = helpers.get(model)
            if h is None:
                raise RuntimeError(f"Model {model} tidak berhasil di-load")

            out = audio_path("CoquiTTS", f"TTS-{i:03d}_{model_pendek}_{lang}.wav")
            t0 = time.perf_counter()
            is_xtts = (model == MODEL_XTTS_V2)
            if is_xtts:
                h.tts_multilingual(teks, language=lang, output_wav=out)
            else:
                h.tts_simple(teks, output_wav=out)
            waktu_gen = round(time.perf_counter() - t0, 3)

            sr, data = wav.read(out)
            durasi = round(len(data) / sr, 3)
            rtf = round(waktu_gen / durasi, 4) if durasi > 0 else ""
            n_kata = len(teks.split())

            baris.append(baris_kosong(TTS_FIELDS,
                No=i, Model_TTS=model_pendek, Bahasa=lang,
                Teks_Input=teks, Panjang_Teks_kata=n_kata,
                Durasi_Audio_detik=durasi, Sample_Rate_Hz=sr,
                Waktu_Generate_detik=waktu_gen, RTF=rtf,
                Keterangan="auto"))
        except Exception as e:
            baris.append(baris_kosong(TTS_FIELDS,
                No=i, Model_TTS=model_pendek, Bahasa=lang,
                Teks_Input=teks, Keterangan=f"ERROR: {e}"))

    tulis_csv(f"02_CoquiTTS_{TIMESTAMP}.csv", baris, TTS_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WHISPER
# ─────────────────────────────────────────────────────────────────────────────

WHISPER_FIELDS = [
    "No", "Model_Size", "Bahasa", "Nama_Audio", "Durasi_Audio_detik",
    "Transkripsi_Referensi", "Transkripsi_Whisper",
    "Jumlah_Kata_Referensi", "Jumlah_Kata_Whisper",
    "WER_persen", "Waktu_Proses_detik", "RTF", "Keterangan"
]

# Kasus uji: generate audio sintetis dari teks, lalu transkripsi balik
# Untuk akurasi WER yang valid → idealnya gunakan audio rekaman manusia
# Script ini generate audio via TTS dulu sebagai placeholder
WHISPER_KASUS = [
    # (lang_whisper, model_size, teks_referensi)
    # Bahasa Indonesia
    ("id", "base", "Selamat pagi semua orang"),
    ("id", "base", "Saya sedang belajar kecerdasan buatan"),
    ("id", "base", "Teknologi berkembang dengan sangat cepat"),
    ("id", "base", "Universitas saya terletak di kota Bandung"),
    ("id", "base", "Cuaca hari ini sangat cerah dan menyenangkan"),
    ("id", "base", "Penelitian tugas akhir ini membahas model kecerdasan buatan untuk pemrosesan audio"),
    ("id", "base", "Python adalah bahasa pemrograman yang populer"),
    ("id", "base", "Kecerdasan buatan dapat membantu manusia dalam berbagai bidang"),
    # Bahasa Inggris
    ("en", "base", "Hello my name is John"),
    ("en", "base", "The weather is very nice today"),
    ("en", "base", "Artificial intelligence is changing the world"),
    ("en", "base", "I am testing a speech recognition system"),
    # Perbandingan model tiny vs base vs small (3 audio sama)
    ("id", "tiny",  "Selamat datang di sistem pengenalan suara"),
    ("id", "base",  "Selamat datang di sistem pengenalan suara"),
    ("id", "small", "Selamat datang di sistem pengenalan suara"),
]


def hitung_wer(ref: str, hyp: str) -> float:
    """WER sederhana tanpa library eksternal."""
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    n = len(ref_words)
    if n == 0:
        return 0.0
    m = len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return round(dp[m] / n * 100, 2)


def benchmark_whisper():
    print("\n[3/6] Menguji Whisper...")
    try:
        from coba_whisper import WhisperHelper
        h = WhisperHelper()
    except Exception as e:
        print(f"  [!] Gagal load model: {e}")
        return

    # Generate audio placeholder via gTTS atau TTS sintetis sederhana
    # Jika tidak ada TTS, buat file sin wave pendek sebagai placeholder
    def buat_audio_placeholder(teks, lang="id"):
        """Buat audio sintetis sederhana; gantikan dengan rekaman asli untuk WER valid."""
        try:
            from gtts import gTTS
            out = tempfile.mktemp(suffix=".mp3")
            gTTS(text=teks, lang=lang, slow=False).save(out)
            return out, "gtts"
        except Exception:
            pass
        try:
            from coba_tts import TTSHelper
            h2 = TTSHelper()
            out = tempfile.mktemp(suffix=".wav")
            h2.synthesize(teks, output_path=out)
            return out, "coqui"
        except Exception:
            pass
        # Fallback: silence
        import numpy as np, scipy.io.wavfile as wv
        out = tempfile.mktemp(suffix=".wav")
        data = np.zeros(16000 * 3, dtype=np.int16)
        wv.write(out, 16000, data)
        return out, "silence_placeholder"

    baris = []
    loaded_models = {}

    for i, (lang, model_size, referensi) in enumerate(WHISPER_KASUS, 1):
        print(f"  [{i:02d}/{len(WHISPER_KASUS)}] whisper-{model_size} [{lang}]: {referensi[:40]}...")
        try:
            # Load model sekali per size
            if model_size not in loaded_models:
                loaded_models[model_size] = WhisperHelper(model_name=model_size)
            hw = loaded_models[model_size]

            audio_in, sumber = buat_audio_placeholder(referensi, lang=lang)

            import scipy.io.wavfile as wv
            import numpy as np
            try:
                sr, data = wv.read(audio_in)
                durasi = round(len(data) / sr, 3)
            except Exception:
                durasi = 0

            t0 = time.perf_counter()
            result = hw.transcribe(audio_in, language=lang if lang != "id" else None)
            waktu = round(time.perf_counter() - t0, 3)

            transkripsi = result.get("text", "").strip()
            rtf = round(waktu / durasi, 4) if durasi > 0 else ""
            wer = hitung_wer(referensi, transkripsi)
            n_ref = len(referensi.split())
            n_hyp = len(transkripsi.split())

            baris.append(baris_kosong(WHISPER_FIELDS,
                No=i, Model_Size=model_size, Bahasa=lang,
                Nama_Audio=f"synth_{i:02d}_{sumber}",
                Durasi_Audio_detik=durasi,
                Transkripsi_Referensi=referensi,
                Transkripsi_Whisper=transkripsi,
                Jumlah_Kata_Referensi=n_ref, Jumlah_Kata_Whisper=n_hyp,
                WER_persen=wer, Waktu_Proses_detik=waktu, RTF=rtf,
                Keterangan=f"audio:{sumber}"))
        except Exception as e:
            baris.append(baris_kosong(WHISPER_FIELDS,
                No=i, Model_Size=model_size, Bahasa=lang,
                Transkripsi_Referensi=referensi, Keterangan=f"ERROR: {e}"))

    tulis_csv(f"03_Whisper_{TIMESTAMP}.csv", baris, WHISPER_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MUSICGEN
# ─────────────────────────────────────────────────────────────────────────────

MUSIC_FIELDS = [
    "No", "Prompt", "Genre_Target", "Durasi_Target_detik",
    "Durasi_Hasil_detik", "Waktu_Generate_detik",
    "Kecocokan_Prompt_1_5", "Kualitas_Audio_1_5", "Variasi_Musikal_1_5", "Keterangan"
]

MUSIC_KASUS = [
    # (prompt, genre, durasi)
    ("upbeat jazz piano with light drums", "Jazz", 8),
    ("calm lo-fi hip hop with rain sounds", "Lo-fi", 8),
    ("epic orchestral soundtrack for a fantasy film", "Orchestral", 10),
    ("electronic dance music with heavy bass and synth", "EDM", 8),
    ("classical piano solo, melancholic and slow", "Classical", 10),
    ("Indonesian gamelan music, traditional and slow", "Gamelan", 10),
    ("acoustic folk guitar with soft singing", "Folk", 8),
    ("heavy metal with distorted guitar and fast drums", "Metal", 8),
    ("smooth R&B with saxophone and slow tempo", "R&B", 8),
    ("ambient atmospheric music for meditation", "Ambient", 10),
    ("country music with guitar and harmonica", "Country", 8),
    ("hip hop beat with samples and vinyl crackle", "Hip Hop", 8),
    ("cinematic trailer music with brass and percussion", "Cinematic", 10),
    ("8-bit chiptune video game music", "8-bit", 8),
    ("bossa nova guitar and light jazz percussion", "Bossa Nova", 8),
    # Uji konsistensi (prompt sama diulang 2x)
    ("upbeat jazz piano with light drums", "Jazz", 8),
    ("calm lo-fi hip hop with rain sounds", "Lo-fi", 8),
    ("epic orchestral soundtrack for a fantasy film", "Orchestral", 10),
]


def benchmark_musicgen():
    print("\n[4/6] Menguji MusicGen...")
    try:
        from coba_musicgen import MusicGenHelper
        import scipy.io.wavfile as wav
        h = MusicGenHelper()
    except Exception as e:
        print(f"  [!] Gagal load model: {e}")
        return

    baris = []
    for i, (prompt, genre, durasi) in enumerate(MUSIC_KASUS, 1):
        print(f"  [{i:02d}/{len(MUSIC_KASUS)}] {genre}: {prompt[:40]}...")
        try:
            genre_slug = genre.replace(" ", "_").replace("/", "-")
            out = audio_path("MusicGen", f"MG-{i:03d}_{genre_slug}.wav")
            t0 = time.perf_counter()
            paths = h.generate(prompt, duration_seconds=durasi, output_wav=out)
            waktu = round(time.perf_counter() - t0, 3)

            sr, data = wav.read(paths[0])
            durasi_hasil = round(len(data) / sr, 3)

            ket = "konsistensi" if i > 15 else "auto"
            baris.append(baris_kosong(MUSIC_FIELDS,
                No=i, Prompt=prompt, Genre_Target=genre,
                Durasi_Target_detik=durasi, Durasi_Hasil_detik=durasi_hasil,
                Waktu_Generate_detik=waktu, Keterangan=ket))
        except Exception as e:
            baris.append(baris_kosong(MUSIC_FIELDS,
                No=i, Prompt=prompt, Genre_Target=genre,
                Durasi_Target_detik=durasi, Keterangan=f"ERROR: {e}"))

    tulis_csv(f"04_MusicGen_{TIMESTAMP}.csv", baris, MUSIC_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUDIOLDM
# ─────────────────────────────────────────────────────────────────────────────

AUDIOLDM_FIELDS = [
    "No", "Prompt", "Kategori", "Negative_Prompt",
    "Inference_Steps", "Guidance_Scale", "Durasi_detik",
    "Waktu_Generate_detik", "Kecocokan_Prompt_1_5", "Realistis_1_5", "Keterangan"
]

AUDIOLDM_KASUS = [
    # Alam
    ("heavy rain falling on a metal roof", "Alam", "music, talking", 20, 3.5, 5),
    ("ocean waves crashing on a rocky shore", "Alam", "music, speech", 20, 3.5, 5),
    ("strong wind blowing through trees", "Alam", "music, voices", 20, 3.5, 5),
    # Hewan
    ("a dog barking aggressively", "Hewan", "music, background noise", 20, 3.5, 4),
    ("birds chirping in a forest", "Hewan", "music, wind", 20, 3.5, 5),
    ("a cat meowing loudly", "Hewan", "music, talking", 20, 3.5, 4),
    # Mesin
    ("car engine revving loudly", "Mesin", "music, voices", 20, 3.5, 4),
    ("typewriter clicking rapidly", "Mesin", "music, echo", 20, 3.5, 4),
    ("electric drill buzzing and grinding", "Mesin", "music, speech", 20, 3.5, 4),
    # Manusia
    ("crowd applauding enthusiastically", "Manusia", "music, noise", 20, 3.5, 5),
    ("people laughing in a group", "Manusia", "music, wind", 20, 3.5, 5),
    ("footsteps walking on gravel", "Manusia", "music, talking", 20, 3.5, 4),
    # Suasana
    ("busy coffee shop with background chatter", "Suasana", "music, echo", 20, 3.5, 5),
    ("forest ambience with birds and wind", "Suasana", "music, traffic", 20, 3.5, 5),
    ("crowded market with vendors shouting", "Suasana", "music, echo", 20, 3.5, 5),
    # Pembanding steps (prompt sama, steps berbeda)
    ("heavy rain falling on a metal roof", "Alam-steps10", "music, talking", 10, 3.5, 5),
    ("heavy rain falling on a metal roof", "Alam-steps50", "music, talking", 50, 3.5, 5),
    ("a dog barking aggressively", "Hewan-steps10", "music, background noise", 10, 3.5, 4),
    ("a dog barking aggressively", "Hewan-steps50", "music, background noise", 50, 3.5, 4),
    ("busy coffee shop with background chatter", "Suasana-steps10", "music, echo", 10, 3.5, 5),
    ("busy coffee shop with background chatter", "Suasana-steps50", "music, echo", 50, 3.5, 5),
]


def benchmark_audioldm():
    print("\n[5/6] Menguji AudioLDM...")
    try:
        from coba_audioldm import AudioLDMHelper
        import scipy.io.wavfile as wav
        h = AudioLDMHelper()
    except Exception as e:
        print(f"  [!] Gagal load model: {e}")
        return

    baris = []
    for i, (prompt, kategori, neg, steps, guidance, durasi) in enumerate(AUDIOLDM_KASUS, 1):
        print(f"  [{i:02d}/{len(AUDIOLDM_KASUS)}] {kategori} steps={steps}: {prompt[:40]}...")
        try:
            kat_slug = kategori.replace(" ", "_").replace("/", "-")
            out = audio_path("AudioLDM", f"ADM-{i:03d}_{kat_slug}_steps{steps}.wav")
            t0 = time.perf_counter()
            paths = h.generate(
                prompt,
                negative_prompt=neg,
                duration_seconds=float(durasi),
                num_inference_steps=steps,
                guidance_scale=guidance,
                output_wav=out,
            )
            waktu = round(time.perf_counter() - t0, 3)

            baris.append(baris_kosong(AUDIOLDM_FIELDS,
                No=i, Prompt=prompt, Kategori=kategori,
                Negative_Prompt=neg, Inference_Steps=steps,
                Guidance_Scale=guidance, Durasi_detik=durasi,
                Waktu_Generate_detik=waktu, Keterangan="auto"))
        except Exception as e:
            baris.append(baris_kosong(AUDIOLDM_FIELDS,
                No=i, Prompt=prompt, Kategori=kategori,
                Inference_Steps=steps, Keterangan=f"ERROR: {e}"))

    tulis_csv(f"05_AudioLDM_{TIMESTAMP}.csv", baris, AUDIOLDM_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# 6. BEATs
# ─────────────────────────────────────────────────────────────────────────────

BEATS_FIELDS = [
    "No", "Nama_Audio", "Sumber_Audio", "Durasi_detik",
    "Label_Sebenarnya",
    "Top1_Label", "Top1_Prob_persen",
    "Top2_Label", "Top3_Label", "Top4_Label", "Top5_Label",
    "Benar_Top1", "Benar_Top5",
    "Waktu_Proses_detik", "Keterangan"
]

# Daftar audio sintetis: (nama, label_gt, freq_Hz, durasi_s)
# Gunakan file audio asli untuk akurasi lebih valid
BEATS_AUDIO_SINTETIS = [
    ("sine_440hz",   "Sine wave",    440,  3),
    ("sine_880hz",   "Sine wave",    880,  3),
    ("sine_220hz",   "Sine wave",    220,  3),
    ("noise_white",  "White noise",  None, 3),
    ("noise_pink",   "Pink noise",   None, 3),
    ("sine_1000hz",  "Sine wave",    1000, 4),
    ("sine_2000hz",  "Sine wave",    2000, 4),
    ("sine_100hz",   "Sine wave",    100,  4),
    ("chord_major",  "Music",        None, 4),
    ("silence",      "Silence",      0,    3),
]


def buat_audio_beats(nama, freq, durasi, sr=16000):
    import numpy as np
    import scipy.io.wavfile as wv
    t = float(durasi)
    n = int(sr * t)
    path = tempfile.mktemp(suffix=f"_{nama}.wav")

    if freq is None and "white" in nama.lower():
        data = (np.random.randn(n) * 8000).astype(np.int16)
    elif freq is None and "pink" in nama.lower():
        white = np.random.randn(n)
        # Simple pink noise approximation
        import scipy.signal
        b, a = scipy.signal.butter(1, 0.1, btype='low')
        data = (scipy.signal.lfilter(b, a, white) * 8000).astype(np.int16)
    elif freq is None and "chord" in nama.lower():
        data = sum(
            (8000 * 0.3 * np.sin(2 * np.pi * f * np.linspace(0, t, n))).astype(np.int16)
            for f in [261, 329, 392]
        ).astype(np.int16)
    elif freq == 0:
        data = np.zeros(n, dtype=np.int16)
    else:
        data = (8000 * np.sin(2 * np.pi * freq * np.linspace(0, t, n))).astype(np.int16)

    wv.write(path, sr, data)
    return path


def benchmark_beats():
    print("\n[6/6] Menguji BEATs...")
    try:
        from coba_beats import BEATsHelper
        import scipy.io.wavfile as wav
        h = BEATsHelper(checkpoint="finetuned")
    except Exception as e:
        print(f"  [!] Gagal load model: {e}")
        return

    baris = []
    for i, (nama, label_gt, freq, durasi) in enumerate(BEATS_AUDIO_SINTETIS, 1):
        print(f"  [{i:02d}/{len(BEATS_AUDIO_SINTETIS)}] {nama} (gt: {label_gt})...")
        try:
            audio_path = buat_audio_beats(nama, freq, durasi)

            sr_r, data_r = wav.read(audio_path)
            dur_aktual = round(len(data_r) / sr_r, 3)

            t0 = time.perf_counter()
            tags = h.tag_audio(audio_path, top_k=5)
            waktu = round(time.perf_counter() - t0, 3)

            top_labels = [t["label"] for t in tags]
            top_probs  = [round(t["probability"] * 100, 2) for t in tags]

            benar_top1 = "Ya" if label_gt.lower() in top_labels[0].lower() else "Tidak"
            benar_top5 = "Ya" if any(label_gt.lower() in lb.lower() for lb in top_labels) else "Tidak"

            baris.append(baris_kosong(BEATS_FIELDS,
                No=i, Nama_Audio=nama, Sumber_Audio="sintetis",
                Durasi_detik=dur_aktual, Label_Sebenarnya=label_gt,
                Top1_Label=top_labels[0] if len(top_labels) > 0 else "",
                Top1_Prob_persen=top_probs[0] if len(top_probs) > 0 else "",
                Top2_Label=top_labels[1] if len(top_labels) > 1 else "",
                Top3_Label=top_labels[2] if len(top_labels) > 2 else "",
                Top4_Label=top_labels[3] if len(top_labels) > 3 else "",
                Top5_Label=top_labels[4] if len(top_labels) > 4 else "",
                Benar_Top1=benar_top1, Benar_Top5=benar_top5,
                Waktu_Proses_detik=waktu, Keterangan="audio_sintetis"))
        except Exception as e:
            baris.append(baris_kosong(BEATS_FIELDS,
                No=i, Nama_Audio=nama, Label_Sebenarnya=label_gt,
                Keterangan=f"ERROR: {e}"))

    # Hitung akurasi
    total = len([b for b in baris if "ERROR" not in b.get("Keterangan", "")])
    top1_benar = sum(1 for b in baris if b.get("Benar_Top1") == "Ya")
    top5_benar = sum(1 for b in baris if b.get("Benar_Top5") == "Ya")
    if total > 0:
        baris.append(baris_kosong(BEATS_FIELDS,
            No="TOTAL", Keterangan=f"Top-1 Accuracy: {round(top1_benar/total*100,1)}% | Top-5 Accuracy: {round(top5_benar/total*100,1)}%"))

    tulis_csv(f"06_BEATs_{TIMESTAMP}.csv", baris, BEATS_FIELDS)
    print(f"  Selesai: {len(baris)} baris")


# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────────────────────────────────────────

RING_FIELDS = [
    "No", "Model", "Jenis_Tugas", "Ukuran_Model",
    "Waktu_Loading_detik", "Rata_Latency_detik", "Jumlah_Uji",
    "Tingkat_Keberhasilan_persen", "Status", "Keterangan"
]

ringkasan_data = {}


def benchmark_loading_time():
    """Ukur waktu loading setiap model."""
    global ringkasan_data

    models_info = [
        ("SeamlessM4T v2", "Translation",   "~1.2 GB", "coba_seamlessm4t", "SeamlessM4THelper", {}),
        ("Coqui TTS",      "TTS+VoiceClone","~1.8 GB", "coba_tts",         "TTSHelper",         {}),
        ("Whisper base",   "ASR",           "~145 MB", "coba_whisper",     "WhisperHelper",     {}),
        ("MusicGen small", "Text-to-Music", "~2 GB",   "coba_musicgen",    "MusicGenHelper",    {}),
        ("AudioLDM",       "Text-to-Audio", "~1.6 GB", "coba_audioldm",    "AudioLDMHelper",    {}),
        ("BEATs finetuned","Audio Tagging", "~346 MB", "coba_beats",       "BEATsHelper",       {"checkpoint": "finetuned"}),
    ]

    print("\n[+] Mengukur waktu loading model...")
    baris = []
    for i, (nama, tugas, ukuran, modul, kelas, kwargs) in enumerate(models_info, 1):
        print(f"  Load {nama}...")
        try:
            mod = __import__(modul)
            cls = getattr(mod, kelas)
            t0 = time.perf_counter()
            obj = cls(**kwargs)
            # Trigger actual load (lazy loaders)
            if hasattr(obj, "load"):
                obj.load()
            elif hasattr(obj, "load_model"):
                obj.load_model()
            load_time = round(time.perf_counter() - t0, 2)

            baris.append(baris_kosong(RING_FIELDS,
                No=i, Model=nama, Jenis_Tugas=tugas, Ukuran_Model=ukuran,
                Waktu_Loading_detik=load_time, Status="OK"))
            ringkasan_data[nama] = {"load": load_time}
        except Exception as e:
            baris.append(baris_kosong(RING_FIELDS,
                No=i, Model=nama, Jenis_Tugas=tugas, Ukuran_Model=ukuran,
                Status="GAGAL", Keterangan=str(e)))

    tulis_csv(f"00_Ringkasan_{TIMESTAMP}.csv", baris, RING_FIELDS)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

MENU = """
========================================
  BENCHMARK 6 MODEL AI AUDIO
  Output: hasil_benchmark/
========================================
Pilih model yang ingin diuji:
  1. SeamlessM4T v2  (translation)
  2. Coqui TTS       (text-to-speech)
  3. Whisper         (speech-to-text)
  4. MusicGen        (text-to-music)
  5. AudioLDM        (text-to-audio)
  6. BEATs           (audio tagging)
  7. Loading time semua model
  A. Jalankan SEMUA (1-6, otomatis)
  Q. Keluar
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(MENU)
    pilihan = input("Pilihan: ").strip().upper()

    if pilihan == "1":
        benchmark_seamless()
    elif pilihan == "2":
        benchmark_tts()
    elif pilihan == "3":
        benchmark_whisper()
    elif pilihan == "4":
        benchmark_musicgen()
    elif pilihan == "5":
        benchmark_audioldm()
    elif pilihan == "6":
        benchmark_beats()
    elif pilihan == "7":
        benchmark_loading_time()
    elif pilihan == "A":
        t_total = time.perf_counter()
        benchmark_seamless()
        benchmark_tts()
        benchmark_whisper()
        benchmark_musicgen()
        benchmark_audioldm()
        benchmark_beats()
        benchmark_loading_time()
        total = round(time.perf_counter() - t_total, 1)
        print(f"\n[+] Benchmark selesai dalam {total} detik")
        print(f"[+] Semua CSV tersimpan di: {HASIL}")
    elif pilihan == "Q":
        print("Keluar.")
    else:
        print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
