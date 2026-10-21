# Dokumentasi Penggunaan - 6 Model AI Audio/Speech

Repo ini berisi 6 model AI untuk pemrosesan audio dan ucapan, lengkap dengan skrip CLI dan antarmuka web (Gradio) untuk pengujian.

---

## Daftar Isi

1. [Ikhtisar 6 Model](#ikhtisar-6-model)
2. [Persiapan Awal](#persiapan-awal)
3. [1. SeamlessM4T v2](#1-seamlessm4t-v2)
4. [2. Coqui TTS](#2-coqui-tts)
5. [3. OpenAI Whisper](#3-openai-whisper)
6. [4. Meta MusicGen](#4-meta-musicgen)
7. [5. AudioLDM](#5-audioldm)
8. [6. Microsoft BEATs](#6-microsoft-beats)
9. [Troubleshooting Umum](#troubleshooting-umum)

---

## Ikhtisar 6 Model

| # | Model | Jenis Tugas | Input | Output | Folder Model | Port UI |
|---|---|---|---|---|---|---|
| 1 | **SeamlessM4T v2** | Translation (teks/suara → teks/suara) | Teks atau audio | Teks atau audio | `Model SeamlessM4T v2/` | 7860 |
| 2 | **Coqui TTS** | Text-to-Speech + Voice Cloning | Teks | Audio ucapan | (download otomatis) | 7861 |
| 3 | **OpenAI Whisper** | Speech-to-Text (ASR) | Audio | Teks transkripsi | `Model Whisper/` | 7862 |
| 4 | **Meta MusicGen** | Text-to-Music | Teks | Audio musik | `Model Music Gen/` | 7863 |
| 5 | **AudioLDM** | Text-to-Audio (sound effect) | Teks | Audio efek/ambient | `Model AudioLDM/` | 7864 |
| 6 | **Microsoft BEATs** | Audio Tagging (klasifikasi) | Audio | Label kelas suara | `Model BEATs/` | 7865 |

---

## Persiapan Awal

### Virtual environment
Semua skrip menggunakan virtual environment di folder `.venv/`:

```bash
cd "D:\Projek Akhir\Percobaan Generated"
.venv\Scripts\activate
```

### Dependency utama
- `torch`, `torchaudio`, `numpy`, `scipy`, `librosa`, `soundfile`
- `transformers` 5.19.0 (SeamlessM4T, MusicGen, Whisper internal)
- `diffusers` 0.41.0 (AudioLDM)
- `openai-whisper` 20250625
- `coqui-tts` 0.27.5
- `gradio` 6.30.0

### Binary native (Windows)
- **ffmpeg** 9.0.2 — dibutuhkan Whisper & encoder audio (diinstall via `winget install Gyan.FFmpeg`)
- **espeak-ng** 1.52.0 — dibutuhkan model VITS/Tacotron untuk fonem (diinstall via `winget install eSpeak-NG.eSpeak-NG`)

Kedua binary ini sudah di-detect otomatis oleh skrip; tidak perlu restart terminal setelah install.

---

## 1. SeamlessM4T v2

Model Meta untuk terjemahan multibahasa teks dan suara. Mendukung 100+ bahasa termasuk Bahasa Indonesia.

### Fitur
- **T2TT** (Text-to-Text Translation): teks → teks bahasa lain
- **T2ST** (Text-to-Speech Translation): teks → audio bahasa lain
- **S2TT** (Speech-to-Text Translation): audio → teks bahasa lain
- **S2ST** (Speech-to-Speech Translation): audio → audio bahasa lain

### Cara pakai - CLI
```bash
python coba_seamlessm4t.py
```
Menu interaktif akan muncul; pilih fitur [1-6].

### Cara pakai - UI web
```bash
python ui_seamlessm4t.py
```
Buka http://127.0.0.1:7860

### Contoh
- T2TT: "Saya suka belajar bahasa." (ind) → "I like to learn languages." (eng)
- Daftar bahasa populer: ind, eng, jpn, kor, cmn, arb, fra, deu, spa, por, rus, dll.

---

## 2. Coqui TTS

Library TTS maintained-fork dari Coqui TTS yang discontinued. Fitur unggulan: XTTS v2 multilingual + voice cloning.

### Fitur
- **TTS Sederhana**: VITS/Tacotron (English only)
- **XTTS v2 Multilingual**: 17 bahasa (TIDAK termasuk Indonesia)
- **Voice Cloning**: tiru suara dari audio referensi 3-10 detik

### Cara pakai - CLI
```bash
python coba_tts.py
```

### Cara pakai - UI web
```bash
python ui_tts.py
```
Buka http://127.0.0.1:7861

### Catatan penting
- **XTTS v2 TIDAK mendukung Bahasa Indonesia** — hanya 17 bahasa: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
- Model XTTS v2 (~1.8 GB) akan auto-download saat pertama digunakan (perlu akses internet)
- Model VITS English (~146 MB) sudah dipakai dan terverifikasi

---

## 3. OpenAI Whisper

Automatic Speech Recognition (ASR) dari OpenAI. Mendukung **99 bahasa termasuk Bahasa Indonesia** - unggulan untuk konteks lokal.

### Fitur
- Transkripsi audio (auto-detect bahasa atau paksa bahasa tertentu)
- Terjemahan audio ke Bahasa Inggris
- Deteksi bahasa dari audio
- Transkripsi dengan timestamp per segmen

### Model yang tersedia
| Model | Ukuran | Kecepatan | Akurasi |
|---|---|---|---|
| tiny | ~75 MB | Sangat cepat | Rendah |
| **base** | **~145 MB** | **Cepat** | **Baik (default, sudah terdownload)** |
| small | ~466 MB | Sedang | Lebih baik |
| medium | ~1.5 GB | Lambat | Sangat baik |
| large/large-v2/large-v3 | ~2.9 GB | Sangat lambat | Terbaik |

### Cara pakai - CLI
```bash
python coba_whisper.py
```

### Cara pakai - UI web
```bash
python ui_whisper.py
```
Buka http://127.0.0.1:7862

### Lokasi model
`Model Whisper/base.pt` (dan model lain yang didownload otomatis)

---

## 4. Meta MusicGen

Generator musik dari deskripsi teks. Fokus pada musik, bukan efek suara umum.

### Fitur
- Generate musik dari teks (duration 1-30 detik)
- Generate beberapa variasi dari satu prompt
- Pilih model: small (~2 GB), medium, large, stereo variants, melody

### Cara pakai - CLI
```bash
python coba_musicgen.py
```

### Cara pakai - UI web
```bash
python ui_musicgen.py
```
Buka http://127.0.0.1:7863

### Lokasi model
`Model Music Gen/musicgen-small/`

### Contoh prompt
- "upbeat jazz piano with light drums"
- "calm lo-fi hip hop with rain sounds"
- "epic orchestral soundtrack for a fantasy film"
- "Indonesian gamelan music, traditional and slow"

### Catatan performa
Di CPU, generate 8 detik musik butuh ~1-2 menit. Dengan GPU jauh lebih cepat.

---

## 5. AudioLDM

Generator audio dari teks, fokus pada **sound effect dan suara ambient** (bukan musik). Beda dengan MusicGen yang untuk musik.

### Fitur
- Generate suara dari teks (hujan, hewan, mesin, suasana)
- Negative prompt (hindari karakteristik tertentu)
- Guidance scale (seberapa patuh ke prompt)
- Seed untuk reproducibility
- Variasi

### Cara pakai - CLI
```bash
python coba_audioldm.py
```

### Cara pakai - UI web
```bash
python ui_audioldm.py
```
Buka http://127.0.0.1:7864

### Lokasi model
`Model AudioLDM/audioldm-s-full-v2/` (~1.6 GB)

### Contoh prompt
- "heavy rain falling on a metal roof"
- "a dog barking in the distance"
- "ocean waves crashing on a rocky shore"
- "footsteps walking on crunchy gravel"

### Catatan
Pipeline `AudioLDMPipeline` sudah deprecated di diffusers 0.33.1+ tapi masih berfungsi. Penerus: AudioLDM2 (`cvssp/audioldm2`).

---

## 6. Microsoft BEATs

Model untuk **audio tagging** - mengenali suara apa yang ada dalam file audio. 527 kelas dari dataset AudioSet.

### Fitur
- **Audio Tagging**: diberi audio, keluarkan top-K label (misal "Speech 95%, Music 3%, Dog 1%")
- **Ekstraksi Fitur**: ambil embedding vektor untuk ML lanjutan
- Pencarian kelas AudioSet berdasarkan kata kunci

### Checkpoint tersedia
- `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` (~346 MB) - untuk audio tagging
- `BEATs_iter3_plus_AS2M.pt` (~347 MB) - untuk ekstraksi fitur

### Cara pakai - CLI
```bash
python coba_beats.py
```

### Cara pakai - UI web
```bash
python ui_beats.py
```
Buka http://127.0.0.1:7865

### Lokasi model
`Model BEATs/` berisi 2 checkpoint `.pt` + `class_labels_indices.csv` (daftar 527 label AudioSet)

### Catatan arsitektur
BEATs belum tersedia di library `transformers`, jadi kode modelnya diambil dari repo resmi [microsoft/unilm/beats](https://github.com/microsoft/unilm/tree/master/beats) dan disimpan di folder `beats/` (5 file Python).

---

## Troubleshooting Umum

### `ffmpeg: [WinError 2] The system cannot find the file specified`
→ Install ffmpeg: `winget install Gyan.FFmpeg`. Skrip akan auto-detect PATH.

### `No espeak backend found`
→ Install espeak-ng: `winget install eSpeak-NG.eSpeak-NG`.

### `ModuleNotFoundError: No module named 'X'`
→ Pastikan virtual env aktif: `.venv\Scripts\activate`

### Port UI sudah dipakai (`OSError: Cannot find empty port`)
→ Cari PID yang pakai port: `netstat -ano | findstr :786X`
→ Kill: `taskkill /F /PID <nomor>`

### Output menu CLI muncul karakter aneh (UnicodeEncodeError)
→ Sudah di-handle: skrip otomatis reconfigure stdout ke UTF-8 di Windows.

### MusicGen/AudioLDM lambat
→ Normal di CPU. Kurangi `duration` dan `steps` untuk eksplorasi cepat.
→ Dengan GPU (CUDA), kecepatan 10-50x lebih cepat.

---

## Struktur Direktori Proyek

```
Percobaan Generated/
├── .venv/                              # Virtual environment
├── beats/                              # Kode model BEATs (dari microsoft/unilm)
├── Model AudioLDM/
│   └── audioldm-s-full-v2/
├── Model BEATs/
│   ├── BEATs_iter3_plus_AS2M.pt
│   ├── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
│   └── class_labels_indices.csv
├── Model Music Gen/
│   └── musicgen-small/
├── Model SeamlessM4T v2/
├── Model Whisper/
│   └── base.pt
├── TTS/                                # Source code Coqui TTS (reference)
├── coba_seamlessm4t.py  +  ui_seamlessm4t.py
├── coba_tts.py          +  ui_tts.py
├── coba_whisper.py      +  ui_whisper.py
├── coba_musicgen.py     +  ui_musicgen.py
├── coba_audioldm.py     +  ui_audioldm.py
├── coba_beats.py        +  ui_beats.py
└── PENGGUNAAN.md                       # Dokumen ini
```
