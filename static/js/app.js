/**
 * SeamlessM4T Studio - Frontend Logic
 * Menangani interaksi UI, Perekaman Audio Mikrofon, dan Komunikasi REST API
 */

document.addEventListener("DOMContentLoaded", () => {
    // =========================================================================
    // 1. ELEMEN DOM & STATE
    // =========================================================================
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanels = document.querySelectorAll(".tab-panel");
    const statusDot = document.getElementById("statusDot");
    const modelStatusText = document.getElementById("modelStatusText");
    const btnPreloadModel = document.getElementById("btnPreloadModel");

    const loadingOverlay = document.getElementById("loadingOverlay");
    const loadingTitle = document.getElementById("loadingTitle");
    const loadingSubtitle = document.getElementById("loadingSubtitle");
    const toastContainer = document.getElementById("toastContainer");

    // =========================================================================
    // 2. HELPER UI: TOAST & LOADING
    // =========================================================================
    function showToast(message, type = "success") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        const icon = type === "success" ? "ph-check-circle" : "ph-warning-circle";
        toast.innerHTML = `<i class="ph-bold ${icon}"></i><span>${message}</span>`;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(100%)";
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    function showLoading(title = "Memproses Model...", subtitle = "Harap tunggu, proses sedang berlangsung.") {
        loadingTitle.textContent = title;
        loadingSubtitle.textContent = subtitle;
        loadingOverlay.classList.remove("hidden");
    }

    function hideLoading() {
        loadingOverlay.classList.add("hidden");
    }

    // =========================================================================
    // 3. CEK STATUS MODEL & PRELOAD
    // =========================================================================
    async function checkModelStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            modelStatusText.textContent = `${data.status} (${data.device})`;
            if (data.is_loaded) {
                statusDot.className = "status-dot dot-ready";
                btnPreloadModel.style.display = "none";
            } else {
                statusDot.className = "status-dot dot-idle";
                btnPreloadModel.style.display = "inline-flex";
            }
        } catch (e) {
            modelStatusText.textContent = "Server Offline";
            statusDot.className = "status-dot dot-idle";
        }
    }

    btnPreloadModel.addEventListener("click", async () => {
        statusDot.className = "status-dot dot-busy";
        modelStatusText.textContent = "Sedang Memuat Bobot Model...";
        showLoading("Memuat Model SeamlessM4T ke Memori...", "Memuat bobot safetensors ke RAM. Proses ini memakan waktu beberapa detik.");
        try {
            const res = await fetch("/api/load_model", { method: "POST" });
            const data = await res.json();
            if (data.success) {
                showToast("Model SeamlessM4T berhasil dimuat ke memori!", "success");
            } else {
                showToast("Gagal memuat model: " + data.error, "error");
            }
        } catch (e) {
            showToast("Koneksi gagal saat memuat model.", "error");
        } finally {
            hideLoading();
            checkModelStatus();
        }
    });

    checkModelStatus();

    // =========================================================================
    // 4. NAVIGASI TAB
    // =========================================================================
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanels.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = `panel${btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)}`;
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) targetPanel.classList.add("active");
        });
    });

    // Quick text samples filler
    document.querySelectorAll(".btn-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const targetInput = document.getElementById(chip.dataset.target);
            if (targetInput) {
                targetInput.value = chip.dataset.text;
                targetInput.dispatchEvent(new Event("input"));
            }
        });
    });

    // =========================================================================
    // 5. FITUR 1: TEXT-TO-SPEECH (T2ST)
    // =========================================================================
    const t2stTextInput = document.getElementById("t2stTextInput");
    const t2stCharCount = document.getElementById("t2stCharCount");
    const btnRunT2st = document.getElementById("btnRunT2st");
    const t2stOutputBox = document.getElementById("t2stOutputBox");
    const t2stAudioPlayer = document.getElementById("t2stAudioPlayer");
    const t2stDownloadBtn = document.getElementById("t2stDownloadBtn");

    t2stTextInput.addEventListener("input", () => {
        t2stCharCount.textContent = `${t2stTextInput.value.length} karakter`;
    });

    btnRunT2st.addEventListener("click", async () => {
        const text = t2stTextInput.value.trim();
        if (!text) {
            showToast("Harap masukkan teks terlebih dahulu.", "error");
            t2stTextInput.focus();
            return;
        }

        const srcLang = document.getElementById("t2stSrcLang").value;
        const tgtLang = document.getElementById("t2stTgtLang").value;

        showLoading("Sintesis Suara (T2ST)...", `Menerjemahkan teks (${srcLang} ➔ ${tgtLang}) dan menghasilkan file suara.`);
        try {
            const res = await fetch("/api/t2st", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, src_lang: srcLang, tgt_lang: tgtLang })
            });
            const data = await res.json();
            if (data.success) {
                t2stAudioPlayer.src = data.audio_url;
                t2stDownloadBtn.href = data.audio_url;
                t2stOutputBox.classList.remove("hidden");
                t2stAudioPlayer.play().catch(() => {});
                showToast("Sintesis suara berhasil!", "success");
            } else {
                showToast("Gagal: " + (data.error || "Terjadi kesalahan"), "error");
            }
        } catch (e) {
            showToast("Koneksi server gagal.", "error");
        } finally {
            hideLoading();
            checkModelStatus();
        }
    });

    // =========================================================================
    // 6. FITUR 2: TEXT-TO-TEXT (T2TT)
    // =========================================================================
    const t2ttInputText = document.getElementById("t2ttInputText");
    const t2ttOutputText = document.getElementById("t2ttOutputText");
    const btnRunT2tt = document.getElementById("btnRunT2tt");
    const btnSwapLang = document.getElementById("btnSwapLang");
    const t2ttSrcLang = document.getElementById("t2ttSrcLang");
    const t2ttTgtLang = document.getElementById("t2ttTgtLang");
    const btnClearT2tt = document.getElementById("btnClearT2tt");
    const btnCopyT2tt = document.getElementById("btnCopyT2tt");

    btnSwapLang.addEventListener("click", () => {
        const temp = t2ttSrcLang.value;
        t2ttSrcLang.value = t2ttTgtLang.value;
        t2ttTgtLang.value = temp;

        const tempText = t2ttInputText.value;
        t2ttInputText.value = t2ttOutputText.value;
        t2ttOutputText.value = tempText;
    });

    btnClearT2tt.addEventListener("click", () => {
        t2ttInputText.value = "";
        t2ttOutputText.value = "";
        t2ttInputText.focus();
    });

    btnCopyT2tt.addEventListener("click", () => {
        const text = t2ttOutputText.value.trim();
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            showToast("Hasil terjemahan berhasil disalin ke clipboard!", "success");
        });
    });

    btnRunT2tt.addEventListener("click", async () => {
        const text = t2ttInputText.value.trim();
        if (!text) {
            showToast("Harap masukkan teks yang ingin diterjemahkan.", "error");
            t2ttInputText.focus();
            return;
        }

        const src = t2ttSrcLang.value;
        const tgt = t2ttTgtLang.value;

        showLoading("Menerjemahkan Teks...", `Menerjemahkan dari ${src} ke ${tgt}...`);
        try {
            const res = await fetch("/api/t2tt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text, src_lang: src, tgt_lang: tgt })
            });
            const data = await res.json();
            if (data.success) {
                t2ttOutputText.value = data.translated_text;
                showToast("Penerjemahan teks berhasil!", "success");
            } else {
                showToast("Gagal menerjemahkan: " + data.error, "error");
            }
        } catch (e) {
            showToast("Gagal menghubungi server.", "error");
        } finally {
            hideLoading();
            checkModelStatus();
        }
    });

    // =========================================================================
    // 7. PEREKAM MIKROFON & AUDIO HANDLING (SHARED)
    // =========================================================================
    function setupAudioRecorder(toggleBtnId, iconId, timerId, statusId, previewBoxId, previewAudioId) {
        let mediaRecorder = null;
        let audioChunks = [];
        let recordedBlob = null;
        let recordInterval = null;
        let recordSeconds = 0;

        const btnToggle = document.getElementById(toggleBtnId);
        const icon = document.getElementById(iconId);
        const timer = document.getElementById(timerId);
        const status = document.getElementById(statusId);
        const previewBox = document.getElementById(previewBoxId);
        const previewAudio = document.getElementById(previewAudioId);

        function updateTimer() {
            recordSeconds++;
            const m = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
            const s = String(recordSeconds % 60).padStart(2, "0");
            timer.textContent = `${m}:${s}`;
        }

        btnToggle.addEventListener("click", async () => {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                // Berhenti merekam
                mediaRecorder.stop();
                btnToggle.classList.remove("recording");
                icon.className = "ph-bold ph-microphone";
                status.textContent = "Rekaman selesai! Anda bisa mendengarkan preview di bawah.";
                clearInterval(recordInterval);
            } else {
                // Mulai merekam
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    audioChunks = [];
                    mediaRecorder = new MediaRecorder(stream);

                    mediaRecorder.ondataavailable = e => {
                        if (e.data.size > 0) audioChunks.push(e.data);
                    };

                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: "audio/wav" });
                        previewAudio.src = URL.createObjectURL(recordedBlob);
                        previewBox.classList.remove("hidden");
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start();
                    btnToggle.classList.add("recording");
                    icon.className = "ph-bold ph-stop";
                    status.textContent = "Sedang merekam suara Anda... Klik tombol untuk berhenti.";
                    recordSeconds = 0;
                    timer.textContent = "00:00";
                    recordInterval = setInterval(updateTimer, 1000);
                } catch (err) {
                    showToast("Tidak dapat mengakses mikrofon: " + err.message, "error");
                }
            }
        });

        return {
            getBlob: () => recordedBlob,
            clear: () => {
                recordedBlob = null;
                audioChunks = [];
                previewBox.classList.add("hidden");
                timer.textContent = "00:00";
                status.textContent = "Klik tombol mikrofon untuk mulai merekam";
            }
        };
    }

    const s2ttRecorder = setupAudioRecorder(
        "btnToggleRecord", "recordIcon", "recordTimer", "recordStatusText",
        "recordedAudioPreviewBox", "recordedAudioPreview"
    );

    const s2stRecorder = setupAudioRecorder(
        "btnToggleRecordS2st", "recordIconS2st", "recordTimerS2st", "recordStatusTextS2st",
        "recordedAudioPreviewBoxS2st", "recordedAudioPreviewS2st"
    );

    // =========================================================================
    // 8. FITUR 3: SPEECH-TO-TEXT (S2TT)
    // =========================================================================
    const btnSwitchMic = document.getElementById("btnSwitchMic");
    const btnSwitchUpload = document.getElementById("btnSwitchUpload");
    const boxMicSection = document.getElementById("boxMicSection");
    const boxUploadSection = document.getElementById("boxUploadSection");
    const audioFileInput = document.getElementById("audioFileInput");
    const btnBrowseAudio = document.getElementById("btnBrowseAudio");
    const audioDropzone = document.getElementById("audioDropzone");
    const chosenFileName = document.getElementById("chosenFileName");

    let currentS2ttMethod = "mic"; // 'mic' | 'upload'
    let uploadedFile = null;

    btnSwitchMic.addEventListener("click", () => {
        currentS2ttMethod = "mic";
        btnSwitchMic.classList.add("active");
        btnSwitchUpload.classList.remove("active");
        boxMicSection.classList.add("active");
        boxUploadSection.classList.remove("active");
    });

    btnSwitchUpload.addEventListener("click", () => {
        currentS2ttMethod = "upload";
        btnSwitchUpload.classList.add("active");
        btnSwitchMic.classList.remove("active");
        boxUploadSection.classList.add("active");
        boxMicSection.classList.remove("active");
    });

    btnBrowseAudio.addEventListener("click", () => audioFileInput.click());

    audioFileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadedFile = e.target.files[0];
            chosenFileName.textContent = `File terpilih: ${uploadedFile.name} (${(uploadedFile.size / 1024).toFixed(1)} KB)`;
            chosenFileName.classList.remove("hidden");
        }
    });

    audioDropzone.addEventListener("dragover", e => { e.preventDefault(); audioDropzone.style.borderColor = "var(--primary-solid)"; });
    audioDropzone.addEventListener("dragleave", e => { e.preventDefault(); audioDropzone.style.borderColor = ""; });
    audioDropzone.addEventListener("drop", e => {
        e.preventDefault();
        audioDropzone.style.borderColor = "";
        if (e.dataTransfer.files.length > 0) {
            uploadedFile = e.dataTransfer.files[0];
            chosenFileName.textContent = `File terpilih: ${uploadedFile.name} (${(uploadedFile.size / 1024).toFixed(1)} KB)`;
            chosenFileName.classList.remove("hidden");
        }
    });

    const btnRunS2tt = document.getElementById("btnRunS2tt");
    const s2ttOutputBox = document.getElementById("s2ttOutputBox");
    const s2ttResultText = document.getElementById("s2ttResultText");
    const btnCopyS2tt = document.getElementById("btnCopyS2tt");

    btnCopyS2tt.addEventListener("click", () => {
        navigator.clipboard.writeText(s2ttResultText.value).then(() => {
            showToast("Hasil transkripsi disalin!", "success");
        });
    });

    btnRunS2tt.addEventListener("click", async () => {
        const formData = new FormData();
        const tgtLang = document.getElementById("s2ttTgtLang").value;
        formData.append("tgt_lang", tgtLang);

        if (currentS2ttMethod === "mic") {
            const blob = s2ttRecorder.getBlob();
            if (!blob) {
                showToast("Harap rekam suara Anda terlebih dahulu.", "error");
                return;
            }
            formData.append("audio", blob, "recording.wav");
        } else {
            if (!uploadedFile) {
                showToast("Harap pilih atau seret file audio terlebih dahulu.", "error");
                return;
            }
            formData.append("audio", uploadedFile);
        }

        showLoading("Transkripsi Audio (S2TT)...", `Memproses audio dan menerjemahkan ke ${tgtLang}...`);
        try {
            const res = await fetch("/api/s2tt", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                s2ttResultText.value = data.transcription;
                s2ttOutputBox.classList.remove("hidden");
                showToast("Transkripsi audio berhasil!", "success");
            } else {
                showToast("Gagal: " + (data.error || "Gagal mentranskripsi"), "error");
            }
        } catch (e) {
            showToast("Gagal menghubungi server.", "error");
        } finally {
            hideLoading();
            checkModelStatus();
        }
    });

    // =========================================================================
    // 9. FITUR 4: SPEECH-TO-SPEECH (S2ST)
    // =========================================================================
    const btnRunS2st = document.getElementById("btnRunS2st");
    const s2stOutputBox = document.getElementById("s2stOutputBox");
    const s2stAudioPlayer = document.getElementById("s2stAudioPlayer");
    const s2stDownloadBtn = document.getElementById("s2stDownloadBtn");

    btnRunS2st.addEventListener("click", async () => {
        const blob = s2stRecorder.getBlob();
        if (!blob) {
            showToast("Harap rekam suara terlebih dahulu.", "error");
            return;
        }

        const tgtLang = document.getElementById("s2stTgtLang").value;
        const formData = new FormData();
        formData.append("audio", blob, "input_s2st.wav");
        formData.append("tgt_lang", tgtLang);

        showLoading("Speech-to-Speech (S2ST)...", `Menerjemahkan audio ke suara bahasa ${tgtLang}...`);
        try {
            const res = await fetch("/api/s2st", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.success) {
                s2stAudioPlayer.src = data.audio_url;
                s2stDownloadBtn.href = data.audio_url;
                s2stOutputBox.classList.remove("hidden");
                s2stAudioPlayer.play().catch(() => {});
                showToast("Terjemahan suara berhasil!", "success");
            } else {
                showToast("Gagal: " + (data.error || "Gagal terjemah suara"), "error");
            }
        } catch (e) {
            showToast("Gagal menghubungi server.", "error");
        } finally {
            hideLoading();
            checkModelStatus();
        }
    });
});
