# Percobaan Model AI Audio dan Speech 🎙️🎵

Repositori ini berisi kumpulan eksperimen, implementasi antarmuka interaktif (Gradio & Flask Web UI), serta pengujian performa (*benchmarking*) untuk **6 Model Kecerdasan Buatan (AI)** di bidang pemrosesan audio, suara, dan ucapan (*speech processing*).

Proyek ini dirancang untuk mengevaluasi kemampuan, kualitas, serta kecepatan inferensi masing-masing model dalam berbagai skenario penggunaan nyata (terjemahan multimodal, speech-to-text, text-to-speech, musik generatif, efek suara, dan klasifikasi audio).

---

## 📌 Daftar Model yang Diuji

| # | Model | Kategori / Tugas | Input | Output | Antarmuka & Port |
|---|---|---|---|---|---|
| 1 | **Meta SeamlessM4T v2** | Terjemahan Suara & Teks Multimodal (T2TT, T2ST, S2TT, S2ST) | Teks / Audio | Teks / Audio | Gradio (7860) & Flask (5000) |
| 2 | **Coqui TTS** | Text-to-Speech (Sintesis Suara & Voice Cloning) | Teks (+ Voice Ref) | Audio Ucapan | Gradio (7861) |
| 3 | **OpenAI Whisper** | Automatic Speech Recognition (ASR / Transkripsi Suara) | Audio | Teks Transkrip | Gradio (7862) |
| 4 | **Meta MusicGen** | Text-to-Music (Generasi Musik AI) | Prompt Teks | Audio Musik | Gradio (7863) |
| 5 | **AudioLDM** | Text-to-Audio (Generasi Efek Suara & Foley) | Prompt Teks | Audio Efek/Suasana | Gradio (7864) |
| 6 | **Microsoft BEATs** | Audio Event Classification & Tagging (Klasifikasi Kejadian Suara) | Audio | Label & Probabilitas Kelas | Gradio (7865) |

---

## 📊 Ringkasan Hasil Benchmark

Hasil pengujian otomatis (*automated benchmark*) tersimpan dalam format CSV pada folder [`hasil_benchmark/`](file:///d:/Projek%20Akhir/Percobaan%20Generated/hasil_benchmark). Berikut adalah ringkasan analisis performa dari masing-masing model:

### 1. SeamlessM4T v2 (`01_SeamlessM4T_20261019_145645.csv`)
Diuji pada **30 skenario** mencakup 4 mode terjemahan:
- **T2TT (Text-to-Text Translation - 15 kasus)**:
  - Menguji translasi lintas bahasa (Indonesia, Inggris, Jepang, Prancis, Arab).
  - Latensi inferensi sangat stabil di kisaran **1.02s – 2.76s** (kecuali kasus awal cold-start ~19.9s).
  - Skor BLEU mencapai **100.0%** pada percakapan umum seperti:
    - *"Selamat pagi, apa kabar?"* ➔ *"Good morning, how are you?"*
    - *"The weather is very nice today."* ➔ *"Cuaca sangat bagus hari ini."*
- **T2ST (Text-to-Speech Translation - 5 kasus)**:
  - Menghasilkan audio sintesis ucapan langsung dari teks asal.
  - Latensi berkisar antara **1.37s – 3.23s** dengan rata-rata **RTF (Real-Time Factor) ~1.15**.
- **S2TT (Speech-to-Text Translation - 5 kasus)**:
  - Menerjemahkan rekaman suara sumber langsung ke teks bahasa tujuan.
  - Latensi inferensi **2.27s – 6.01s** (RTF 0.93 – 2.65).
  - Rata-rata BLEU score mencapai **66.3%**, dengan akurasi semantik tinggi.
- **S2ST (Speech-to-Speech Translation - 5 kasus)**:
  - Menerjemahkan suara sumber langsung menjadi ucapan suara bahasa target.
  - Latensi **2.62s – 5.29s** dengan RTF ~1.34 – 2.73, seluruh sampel terjemahan berhasil dibuat.

---

### 2. Coqui TTS (`02_CoquiTTS_20261019_150415.csv`)
Diuji pada **15 skenario sintesis suara**:
- **Model VITS (English single-speaker, LJSpeech)**:
  - Performa sangat cepat dengan **RTF antara 0.38 – 0.48** (generasi audio jauh lebih cepat dari durasi audio aslinya / *faster than real-time*).
  - Menghasilkan audio berdurasi 3 hingga 5.3 detik hanya dalam waktu **1.38s – 2.07s**.
- **Model XTTS v2 (Multilingual & Voice Cloning)**:
  - Pada pengujian skrip batch *headless*, model memerlukan parameter sampel suara referensi pembicara (*speaker prompt audio*) dan dependensi kamus fonem per bahasa (seperti `cutlet` untuk bahasa Jepang dan `pypinyin` untuk Mandarin).
  - Pada pengujian melalui antarmuka interaktif [`ui_tts.py`](file:///d:/Projek%20Akhir/Percobaan%20Generated/ui_tts.py), fitur voice cloning dapat dijalankan dengan mengunggah sampel suara referensi secara langsung.

---

### 3. OpenAI Whisper (`03_Whisper_20261019_150715.csv`)
Diuji pada **15 sampel audio** ucapan (Bahasa Indonesia dan Bahasa Inggris):
- **Model Base (Bahasa Inggris)**:
  - Sangat cepat dan presisi, waktu proses rata-rata hanya **0.56s – 0.59s**.
  - Word Error Rate (WER) rendah (14% – 20%), di mana mayoritas perbedaan hanya berupa tanda baca (titik/koma).
- **Model Base (Bahasa Indonesia)**:
  - Waktu proses rata-rata **0.98s – 1.23s**.
  - WER berkisar 12.5% – 42.8%. Model terkadang mengadaptasi kata serapan ke ejaan fonetik lokal (misalnya *"Python"* menjadi *"Piton"*, atau *"pemrosesan"* menjadi *"pemrosisan"*).
- **Perbandingan Ukuran Model (Tiny vs Base vs Small)**:
  - Pada kalimat uji yang sama (*"Selamat datang di sistem pengenalan suara"*):
    - `tiny`: Waktu proses 17.54s (cold-start load), WER 16.67%.
    - `base`: Waktu proses **1.02s**, WER 16.67%.
    - `small`: Waktu proses 111.5s, **WER 0.0% (Akurasi Sempurna 100% kata per kata!)**.

---

### 4. Meta MusicGen (`04_MusicGen_20261019_151304.csv`)
Diuji pada **18 skenario generasi musik** dengan variasi genre (Jazz, Lo-fi, Orchestral, EDM, Classical, Gamelan Tradisional Indonesia, Folk, Heavy Metal, R&B, Ambient, Chiptune 8-bit, Bossa Nova):
- Durasi audio musik yang dihasilkan berkisar antara **7.94 detik** (target 8s) hingga **9.94 detik** (target 10s).
- Waktu generasi inferensi rata-rata berada pada kisaran **29.5s – 91.1s** (bergantung pada kompleksitas prompt dan durasi target).
- Model berhasil mengenali karakteristik instrumen dan suasana spesifik, termasuk musik tradisional seperti Gamelan maupun chiptune game retro.

---

### 5. AudioLDM (`05_AudioLDM_20261019_154142.csv`)
Diuji pada **21 skenario generasi efek suara** lintas 5 kategori utama (Alam, Hewan, Mesin, Manusia, dan Suasana Lingkungan):
- **Pengaruh Parameter `Inference Steps`**:
  - **10 Steps**: Waktu generasi sangat singkat (**7.0s – 8.5s**) untuk audio 4–5 detik. Cocok untuk preview cepat.
  - **20 Steps (Default)**: Waktu generasi **10.5s – 26.0s**. Menghasilkan keseimbangan optimal antara tekstur suara dan kecepatan proses.
  - **50 Steps**: Waktu generasi **21.5s – 26.0s**. Menghasilkan detail akustik frekuensi yang lebih padat dan tajam.
- Contoh efek suara yang berhasil disintesis meliputi suara hujan di atap seng, deburan ombak, gonggongan anjing, mesin mobil, hingga suasana kedai kopi (*coffee shop ambience*).

---

### 6. Microsoft BEATs (`06_BEATs_20261019_154857.csv`)
Diuji pada **10 skenario audio klasifikasi**:
- Model menggunakan arsitektur *Bidirectional Encoder representation from Audio Transformers* (BEATs) dengan fine-tuning pada dataset AudioSet (527 label kelas audio).
- Pengujian batch headless mendokumentasikan kebutuhan format indexing tensor output. Pada antarmuka interaktif [`ui_beats.py`](file:///d:/Projek%20Akhir/Percobaan%20Generated/ui_beats.py), model secara interaktif menampilkan visualisasi probabilitas **Top-5 prediksi kelas suara** untuk file audio yang diunggah atau direkam via mikrofon.

---

## 💻 Panduan Menjalankan Antarmuka (Web UI)

Tersedia antarmuka interaktif berbasis web untuk memudahkan pengujian secara visual:

### 1. SeamlessM4T v2
- **Gradio UI** (Port 7860):
  ```bash
  python ui_seamlessm4t.py
  ```
- **Flask Web UI** (Port 5000 - Tampilan Modern & Responsif):
  ```bash
  python app.py
  ```
  Buka browser di: `http://localhost:5000`

### 2. Coqui TTS (Port 7861)
```bash
python ui_tts.py
```

### 3. OpenAI Whisper (Port 7862)
```bash
python ui_whisper.py
```

### 4. Meta MusicGen (Port 7863)
```bash
python ui_musicgen.py
```

### 5. AudioLDM (Port 7864)
```bash
python ui_audioldm.py
```

### 6. Microsoft BEATs (Port 7865)
```bash
python ui_beats.py
```

### 7. Menjalankan Uji Benchmark Otomatis
Untuk menjalankan kembali seluruh pengujian otomatis ke format CSV:
```bash
python benchmark.py
```

---

## ⚙️ Prasyarat & Instalasi

### 1. Kebutuhan Sistem
- **Python**: Versi 3.10 – 3.12
- **FFmpeg**: Diperlukan untuk pengolahan file audio (`winget install Gyan.FFmpeg` pada Windows)
- **eSpeak-NG**: Diperlukan untuk modul fonemisasi TTS (`winget install eSpeak-NG.eSpeak-NG` pada Windows)

### 2. Instalasi Paket Dependensi
Aktifkan virtual environment Anda, lalu instal dependensi dari `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 📁 Struktur Direktori

```text
├── app.py                     # Server Flask Web UI (SeamlessM4T)
├── benchmark.py               # Skrip pengujian benchmarking otomatis untuk semua model
├── coba_seamlessm4t.py        # Modul backend SeamlessM4T v2
├── coba_tts.py                # Modul backend Coqui TTS
├── coba_whisper.py            # Modul backend OpenAI Whisper
├── coba_musicgen.py           # Modul backend Meta MusicGen
├── coba_audioldm.py           # Modul backend AudioLDM
├── coba_beats.py              # Modul backend Microsoft BEATs
├── ui_seamlessm4t.py          # Antarmuka Gradio SeamlessM4T v2 (Port 7860)
├── ui_tts.py                  # Antarmuka Gradio Coqui TTS (Port 7861)
├── ui_whisper.py              # Antarmuka Gradio OpenAI Whisper (Port 7862)
├── ui_musicgen.py             # Antarmuka Gradio Meta MusicGen (Port 7863)
├── ui_audioldm.py             # Antarmuka Gradio AudioLDM (Port 7864)
├── ui_beats.py                # Antarmuka Gradio Microsoft BEATs (Port 7865)
├── beats/                     # Source code arsitektur model BEATs
├── hasil_benchmark/           # Hasil CSV dan data pengujian benchmark
│   ├── 01_SeamlessM4T_*.csv
│   ├── 02_CoquiTTS_*.csv
│   ├── 03_Whisper_*.csv
│   ├── 04_MusicGen_*.csv
│   ├── 05_AudioLDM_*.csv
│   └── 06_BEATs_*.csv
├── templates/                 # Template HTML untuk antarmuka web Flask
├── static/                    # Berkas CSS dan JavaScript untuk Flask UI
├── PENGGUNAAN.md              # Dokumentasi teknis rinci penggunaan masing-masing model
└── requirements.txt           # Daftar pustaka Python yang digunakan
```

---

## 📄 Lisensi & Referensi
Projek ini dibuat untuk keperluan penelitian, pengujian performa, dan eksperimen kecerdasan buatan dalam domain audio dan wicara.
Model-model AI yang digunakan tunduk pada lisensi sumber terbuka dari masing-masing penyedia (Meta AI, OpenAI, Coqui, dan Microsoft).
