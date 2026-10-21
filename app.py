"""
=============================================================================
 SeamlessM4T Web Application Server (Flask Backend)
 Proyek: Percobaan Generated
=============================================================================
Menyediakan server API RESTful dan antarmuka web modern untuk model
Meta SeamlessM4T v2 / Medium.
=============================================================================
"""

import os
import sys
import json
import struct
import tempfile
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file

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
    def __init__(self, fh, offset, length, shape, dtype):
        self.fh = fh
        self.offset = offset
        self.length = length
        self.shape = shape
        self.dtype = dtype

    def get_shape(self):
        return self.shape

    def get_dtype(self):
        return self.dtype

    def __getitem__(self, item):
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
    def __init__(self, filename, framework="pt", device="cpu"):
        self.fh = open(filename, "rb")
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
    try:
        import safetensors
        import transformers.modeling_utils
        safetensors.safe_open = FastFileSafeOpen
        transformers.modeling_utils.safe_open = FastFileSafeOpen
    except Exception as e:
        print(f"[!] Warning safetensors patch: {e}")

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
# 2. INISIALISASI FLASK & MODEL MANAGER
# =============================================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = BASE_DIR / "Model SeamlessM4T v2"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"
TEMP_AUDIO_DIR.mkdir(exist_ok=True)

MODEL_LOCK = threading.Lock()
PROCESSOR = None
MODEL = None
MODEL_STATUS = "Belum Dimuat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGES = [
    {"code": "ind", "name": "Bahasa Indonesia", "flag": "🇮🇩"},
    {"code": "eng", "name": "English", "flag": "🇬🇧"},
    {"code": "jav", "name": "Basa Jawa", "flag": "🇮🇩"},
    {"code": "jpn", "name": "Japanese (日本語)", "flag": "🇯🇵"},
    {"code": "arb", "name": "Arabic (العربية)", "flag": "🇸🇦"},
    {"code": "zho", "name": "Mandarin (中文)", "flag": "🇨🇳"},
    {"code": "kor", "name": "Korean (한국어)", "flag": "🇰🇷"},
    {"code": "fra", "name": "Français", "flag": "🇫🇷"},
    {"code": "deu", "name": "Deutsch", "flag": "🇩🇪"},
    {"code": "spa", "name": "Español", "flag": "🇪🇸"},
    {"code": "rus", "name": "Russian (Русский)", "flag": "🇷🇺"},
    {"code": "tha", "name": "Thai (ไทย)", "flag": "🇹🇭"},
    {"code": "vie", "name": "Tiếng Việt", "flag": "🇻🇳"},
    {"code": "nld", "name": "Nederlands", "flag": "🇳🇱"},
    {"code": "ita", "name": "Italiano", "flag": "🇮🇹"},
    {"code": "por", "name": "Português", "flag": "🇵🇹"},
    {"code": "tur", "name": "Türkçe", "flag": "🇹🇷"},
]

def get_model_path():
    if LOCAL_MODEL_DIR.exists() and (LOCAL_MODEL_DIR / "config.json").exists():
        return str(LOCAL_MODEL_DIR)
    return "facebook/seamless-m4t-v2-large"

def get_processor():
    global PROCESSOR
    if PROCESSOR is None:
        from transformers import AutoProcessor
        model_path = get_model_path()
        print(f"[*] Memuat processor dari: {model_path}...")
        PROCESSOR = AutoProcessor.from_pretrained(model_path)
    return PROCESSOR

def get_model():
    global MODEL, MODEL_STATUS
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                MODEL_STATUS = "Sedang Memuat..."
                get_processor()
                from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2ForTextToText
                model_path = get_model_path()
                print(f"[*] Memuat model SeamlessM4T dari {model_path}...")
                try:
                    MODEL = SeamlessM4Tv2Model.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    print(f"[!] Fallback ke model Text-to-Text: {e}")
                    MODEL = SeamlessM4Tv2ForTextToText.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True
                    )
                MODEL.to(DEVICE)
                MODEL.eval()
                MODEL_STATUS = "Siap Digunakan (Ready)"
                print("[+] Model SeamlessM4T berhasil dimuat!")
    return MODEL

# =============================================================================
# 3. ROUTE & ENDPOINT API
# =============================================================================

@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES, device=DEVICE.upper())

@app.route("/api/status", methods=["GET"])
def api_status():
    global MODEL_STATUS
    return jsonify({
        "status": MODEL_STATUS,
        "device": DEVICE.upper(),
        "model_path": get_model_path(),
        "is_loaded": MODEL is not None
    })

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    try:
        get_model()
        return jsonify({"success": True, "message": "Model berhasil dimuat!"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/languages", methods=["GET"])
def api_languages():
    return jsonify(LANGUAGES)

@app.route("/api/t2tt", methods=["POST"])
def api_text_to_text():
    """Text-to-Text Translation"""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    src_lang = data.get("src_lang", "ind").strip()
    tgt_lang = data.get("tgt_lang", "eng").strip()

    if not text:
        return jsonify({"success": False, "error": "Teks tidak boleh kosong."}), 400

    try:
        processor = get_processor()
        model = get_model()

        with MODEL_LOCK:
            inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                if hasattr(model, "generate"):
                    out_tokens = model.generate(
                        **inputs,
                        tgt_lang=tgt_lang,
                        generate_speech=False if hasattr(model, "vocoder") else None
                    )
                else:
                    out_tokens = model.generate(**inputs, tgt_lang=tgt_lang)

            if isinstance(out_tokens, tuple):
                out_tokens = out_tokens[0]
            if out_tokens.ndim > 1:
                token_list = out_tokens[0].tolist()
            else:
                token_list = out_tokens.tolist()

            translated_text = processor.decode(token_list, skip_special_tokens=True)

        return jsonify({
            "success": True,
            "original_text": text,
            "translated_text": translated_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/t2st", methods=["POST"])
def api_text_to_speech():
    """Text-to-Speech Translation (mengembalikan file audio)"""
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    src_lang = data.get("src_lang", "ind").strip()
    tgt_lang = data.get("tgt_lang", "eng").strip()
    speaker_id = int(data.get("speaker_id", 0))

    if not text:
        return jsonify({"success": False, "error": "Teks tidak boleh kosong."}), 400

    try:
        processor = get_processor()
        model = get_model()

        with MODEL_LOCK:
            inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
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

            out_filename = f"t2st_{src_lang}_{tgt_lang}_{tempfile.mktemp().split(os.sep)[-1]}.wav"
            out_path = TEMP_AUDIO_DIR / out_filename
            import scipy.io.wavfile as wavfile
            wavfile.write(str(out_path), rate=sample_rate, data=audio_arr)

        return jsonify({
            "success": True,
            "audio_url": f"/api/audio/{out_filename}",
            "sample_rate": sample_rate
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/s2tt", methods=["POST"])
def api_speech_to_text():
    """Speech-to-Text / Transkripsi Audio"""
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "File audio tidak ditemukan."}), 400

    audio_file = request.files["audio"]
    tgt_lang = request.form.get("tgt_lang", "ind").strip()

    try:
        import soundfile as sf
        import torchaudio

        temp_input = TEMP_AUDIO_DIR / f"input_{tempfile.mktemp().split(os.sep)[-1]}.wav"
        audio_file.save(str(temp_input))

        audio_data, orig_sr = sf.read(str(temp_input))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            tensor_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
            audio_data = resampler(tensor_audio).squeeze(0).numpy()

        processor = get_processor()
        model = get_model()

        with MODEL_LOCK:
            audio_inputs = processor(audios=audio_data, return_tensors="pt", sampling_rate=16000).to(DEVICE)
            with torch.no_grad():
                out_tokens = model.generate(
                    **audio_inputs,
                    tgt_lang=tgt_lang,
                    generate_speech=False
                )

            if isinstance(out_tokens, tuple):
                out_tokens = out_tokens[0]
            transcription = processor.decode(out_tokens[0].tolist(), skip_special_tokens=True)

        # Hapus file input sementara
        try:
            temp_input.unlink(missing_ok=True)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "transcription": transcription,
            "tgt_lang": tgt_lang
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/s2st", methods=["POST"])
def api_speech_to_speech():
    """Speech-to-Speech Translation"""
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "File audio tidak ditemukan."}), 400

    audio_file = request.files["audio"]
    tgt_lang = request.form.get("tgt_lang", "eng").strip()
    speaker_id = int(request.form.get("speaker_id", 0))

    try:
        import soundfile as sf
        import torchaudio

        temp_input = TEMP_AUDIO_DIR / f"input_s2s_{tempfile.mktemp().split(os.sep)[-1]}.wav"
        audio_file.save(str(temp_input))

        audio_data, orig_sr = sf.read(str(temp_input))
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            tensor_audio = torch.from_numpy(audio_data).float().unsqueeze(0)
            audio_data = resampler(tensor_audio).squeeze(0).numpy()

        processor = get_processor()
        model = get_model()

        with MODEL_LOCK:
            audio_inputs = processor(audios=audio_data, return_tensors="pt", sampling_rate=16000).to(DEVICE)
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

            out_filename = f"s2st_{tgt_lang}_{tempfile.mktemp().split(os.sep)[-1]}.wav"
            out_path = TEMP_AUDIO_DIR / out_filename
            import scipy.io.wavfile as wavfile
            wavfile.write(str(out_path), rate=sample_rate, data=audio_arr)

        try:
            temp_input.unlink(missing_ok=True)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "audio_url": f"/api/audio/{out_filename}",
            "sample_rate": sample_rate
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/audio/<filename>")
def api_serve_audio(filename):
    audio_path = TEMP_AUDIO_DIR / filename
    if audio_path.exists():
        return send_file(str(audio_path), mimetype="audio/wav")
    return jsonify({"error": "File tidak ditemukan"}), 404

# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("      SERVER SEAMLESSM4T WEB UI BERJALAN")
    print("=" * 65)
    print(f"[*] Perangkat: {DEVICE.upper()}")
    print(f"[*] Folder Model: {get_model_path()}")
    print("[*] Buka browser Anda di: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
