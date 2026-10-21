"""
=============================================================================
 Skrip Pengujian Microsoft BEATs (Audio Tagging / Audio Classification)
 Proyek: Percobaan Generated
=============================================================================
BEATs = Bidirectional Encoder representation from Audio Transformers.
Model ini MENGENALI suara (bukan membuat suara): diberi audio, ia menebak
label dari 527 kelas AudioSet (misal: Speech, Dog, Guitar, Rain, Car).

Fitur:
1. Audio tagging        -> daftar label + probabilitas
2. Ekstraksi fitur      -> embedding vektor (untuk ML lanjutan)
3. Daftar kelas AudioSet
4. Info model & checkpoint

Catatan: kode model BEATs ada di folder "beats/" (diambil dari repo resmi
microsoft/unilm) karena BEATs belum tersedia di library transformers.
=============================================================================
"""

import os
import sys
import csv
import argparse
import warnings
from pathlib import Path

if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_SCRIPT_DIR = Path(__file__).resolve().parent

# Kode model BEATs pakai import flat ("from backbone import ..."),
# jadi foldernya harus masuk sys.path.
_BEATS_CODE_DIR = _SCRIPT_DIR / "beats"
if str(_BEATS_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_BEATS_CODE_DIR))

import torch
import numpy as np

# =============================================================================
# KONSTANTA
# =============================================================================
MODEL_DIR = _SCRIPT_DIR / "Model BEATs"

CKPT_FINETUNED = MODEL_DIR / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
CKPT_PRETRAINED = MODEL_DIR / "BEATs_iter3_plus_AS2M.pt"
LABELS_CSV = MODEL_DIR / "class_labels_indices.csv"

CHECKPOINTS = {
    "finetuned": str(CKPT_FINETUNED),   # untuk audio tagging (527 kelas)
    "pretrained": str(CKPT_PRETRAINED),  # untuk ekstraksi fitur/embedding
}

TARGET_SR = 16000  # BEATs selalu butuh 16 kHz mono


# =============================================================================
# UTILITAS LABEL AUDIOSET
# =============================================================================
def load_audioset_labels():
    """Baca class_labels_indices.csv -> dict {mid: display_name}."""
    mid_to_name = {}
    if not LABELS_CSV.exists():
        return mid_to_name
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid_to_name[row["mid"]] = row["display_name"]
    return mid_to_name


def load_audio_16k_mono(audio_path):
    """Muat audio apa pun -> tensor float32 mono 16 kHz, range [-1, 1]."""
    import librosa
    wav, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    return torch.from_numpy(wav.astype(np.float32))


# =============================================================================
# KELAS HELPER
# =============================================================================
class BEATsHelper:
    def __init__(self, checkpoint="finetuned", device=None):
        self.checkpoint_key = checkpoint
        self.ckpt_path = CHECKPOINTS.get(checkpoint, CHECKPOINTS["finetuned"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._label_dict = None
        self._mid_to_name = load_audioset_labels()
        print(f"[*] Checkpoint: {Path(self.ckpt_path).name}")
        print(f"[*] Perangkat : {self.device.upper()}")

    def load(self):
        if self._model is None:
            from BEATs import BEATs, BEATsConfig

            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint tidak ditemukan: {self.ckpt_path}\n"
                    f"    Pastikan file .pt sudah ada di folder '{MODEL_DIR.name}'."
                )

            print(f"[*] Memuat checkpoint BEATs ...")
            # weights_only=False karena checkpoint berisi dict config, bukan tensor saja
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

            cfg = BEATsConfig(ckpt["cfg"])
            model = BEATs(cfg)
            model.load_state_dict(ckpt["model"])
            model.eval()
            model.to(self.device)

            self._model = model
            self._label_dict = ckpt.get("label_dict")

            n_params = sum(p.numel() for p in model.parameters())
            mode = "tagging (527 kelas)" if cfg.finetuned_model else "ekstraksi fitur"
            print(f"[+] Model dimuat! Mode: {mode}, {n_params:,} parameter")
        return self._model

    def is_finetuned(self):
        model = self.load()
        return model.cfg.finetuned_model

    # ---- FITUR 1: AUDIO TAGGING ----
    def tag_audio(self, audio_path, top_k=10):
        """Kenali suara dalam audio -> list (label, probabilitas)."""
        model = self.load()
        if not model.cfg.finetuned_model:
            raise RuntimeError(
                f"Checkpoint '{self.checkpoint_key}' bukan model finetuned, "
                f"jadi tidak punya label predictor. Pakai checkpoint 'finetuned'."
            )

        wav = load_audio_16k_mono(audio_path)
        batch = wav.unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(batch.shape, dtype=torch.bool, device=self.device)

        print(f"\n[*] Menganalisis: {audio_path}")
        with torch.no_grad():
            probs, _ = model.extract_features(batch, padding_mask=padding_mask)

        probs = probs[0].cpu()
        k = min(top_k, probs.shape[0])
        top_probs, top_idx = probs.topk(k)

        results = []
        for p, i in zip(top_probs.tolist(), top_idx.tolist()):
            results.append((self._index_to_name(i), p))
        return results

    def _index_to_name(self, idx):
        """Index prediksi -> nama kelas yang bisa dibaca manusia."""
        if self._label_dict:
            mid = self._label_dict.get(idx)
            if mid is not None:
                return self._mid_to_name.get(mid, mid)
        return f"class_{idx}"

    # ---- FITUR 2: EKSTRAKSI FITUR / EMBEDDING ----
    def extract_embedding(self, audio_path):
        """Ambil representasi vektor dari audio (untuk ML lanjutan)."""
        model = self.load()
        if model.cfg.finetuned_model:
            raise RuntimeError(
                "Checkpoint 'finetuned' mengeluarkan probabilitas label, bukan embedding. "
                "Pakai checkpoint 'pretrained' untuk ekstraksi fitur."
            )

        wav = load_audio_16k_mono(audio_path)
        batch = wav.unsqueeze(0).to(self.device)
        padding_mask = torch.zeros(batch.shape, dtype=torch.bool, device=self.device)

        print(f"\n[*] Ekstraksi fitur: {audio_path}")
        with torch.no_grad():
            feats, _ = model.extract_features(batch, padding_mask=padding_mask)

        feats = feats[0].cpu()  # [time_steps, hidden_dim]
        return feats

    def model_info(self):
        model = self.load()
        cfg = model.cfg
        return {
            "checkpoint": Path(self.ckpt_path).name,
            "mode": "audio tagging" if cfg.finetuned_model else "ekstraksi fitur",
            "encoder_layers": cfg.encoder_layers,
            "encoder_embed_dim": cfg.encoder_embed_dim,
            "attention_heads": cfg.encoder_attention_heads,
            "predictor_class": cfg.predictor_class if cfg.finetuned_model else "-",
            "parameter": f"{sum(p.numel() for p in model.parameters()):,}",
            "sample_rate": f"{TARGET_SR} Hz (mono)",
        }


def daftar_kelas_audioset(limit=None):
    """Return list (index, mid, display_name) dari semua kelas AudioSet."""
    rows = []
    if not LABELS_CSV.exists():
        return rows
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["index"]), row["mid"], row["display_name"]))
    return rows[:limit] if limit else rows


# =============================================================================
# MENU INTERAKTIF
# =============================================================================
def interactive_menu(helper: BEATsHelper):
    while True:
        print("\n" + "=" * 65)
        print("       PROGRAM UJI COBA MICROSOFT BEATs (Audio Tagging)")
        print("=" * 65)
        print(f" Checkpoint: {Path(helper.ckpt_path).name}")
        print(f" Perangkat : {helper.device.upper()}")
        print("-" * 65)
        print(" [1] Audio tagging (kenali suara dalam file audio)")
        print(" [2] Ekstraksi fitur / embedding (perlu checkpoint pretrained)")
        print(" [3] Lihat daftar kelas AudioSet")
        print(" [4] Cari kelas AudioSet berdasarkan kata kunci")
        print(" [5] Info model")
        print(" [0] Keluar")
        print("=" * 65)

        pilihan = input("Pilih menu [0-5]: ").strip()
        if pilihan == "0":
            print("\nTerima kasih! Sampai jumpa.")
            break
        try:
            if pilihan == "1":
                audio = input("Path file audio: ").strip(' "\'')
                if not audio:
                    print("[!] Path tidak boleh kosong.")
                    continue
                try:
                    k = int(input("Tampilkan top-K label [default 10]: ").strip() or "10")
                except ValueError:
                    k = 10
                results = helper.tag_audio(audio, top_k=k)
                print(f"\n[+] Top-{len(results)} label terdeteksi:")
                for label, prob in results:
                    bar = "#" * int(prob * 40)
                    print(f"  {prob:6.1%}  {label:<35} {bar}")

            elif pilihan == "2":
                audio = input("Path file audio: ").strip(' "\'')
                if not audio:
                    print("[!] Path tidak boleh kosong.")
                    continue
                feats = helper.extract_embedding(audio)
                print(f"\n[+] Bentuk embedding : {tuple(feats.shape)}")
                print(f"[+] Rata-rata (mean) : {feats.mean().item():.5f}")
                print(f"[+] Simpangan (std)  : {feats.std().item():.5f}")
                print(f"[+] 8 nilai pertama  : {feats.flatten()[:8].tolist()}")

            elif pilihan == "3":
                rows = daftar_kelas_audioset()
                print(f"\n[+] {len(rows)} kelas AudioSet (menampilkan 30 pertama):")
                for idx, mid, name in rows[:30]:
                    print(f"  {idx:>4}  {name}")

            elif pilihan == "4":
                kw = input("Kata kunci (misal: dog, music, rain): ").strip().lower()
                if not kw:
                    print("[!] Kata kunci tidak boleh kosong.")
                    continue
                rows = daftar_kelas_audioset()
                hits = [r for r in rows if kw in r[2].lower()]
                if not hits:
                    print(f"[!] Tidak ada kelas yang cocok dengan '{kw}'.")
                else:
                    print(f"\n[+] {len(hits)} kelas cocok:")
                    for idx, mid, name in hits:
                        print(f"  {idx:>4}  {name}")

            elif pilihan == "5":
                info = helper.model_info()
                print()
                for k, v in info.items():
                    print(f"  {k:<20}: {v}")

            else:
                print("[!] Pilihan tidak valid.")
        except Exception as e:
            print(f"[!] Kesalahan: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Uji coba Microsoft BEATs audio tagging.")
    parser.add_argument("--mode", choices=["menu", "tag", "embed", "info", "labels"], default="menu")
    parser.add_argument("--audio", type=str, default="", help="Path file audio")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS.keys()), default="finetuned")

    args = parser.parse_args()

    if args.mode == "labels":
        for idx, mid, name in daftar_kelas_audioset():
            print(f"{idx}\t{name}")
        return

    helper = BEATsHelper(checkpoint=args.checkpoint)

    if args.mode == "tag":
        if not args.audio:
            print("[!] Perlu --audio <path>")
            sys.exit(1)
        for label, prob in helper.tag_audio(args.audio, top_k=args.top_k):
            print(f"{prob:.4f}\t{label}")
    elif args.mode == "embed":
        if not args.audio:
            print("[!] Perlu --audio <path>")
            sys.exit(1)
        feats = helper.extract_embedding(args.audio)
        print("shape:", tuple(feats.shape))
    elif args.mode == "info":
        for k, v in helper.model_info().items():
            print(f"{k}: {v}")
    else:
        interactive_menu(helper)


if __name__ == "__main__":
    main()
