[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/zaidhassan2/Offline-transriber/pulls)
---

# Transcriber  

**AI transcription without the cloud.**  
An open-source, **privacy-first** web application powered by **OpenAI Whisper** and **FastAPI**. Transcribe YouTube videos or local audio/video files instantly on your machine — no data leaks, no subscription fees.

Developed by [Zaid Hassan](https://zaidhassan.me).

Built with **FastAPI**, **WebSockets**, **Alpine.js**, and **Tailwind CSS** for a modern, real-time experience.

![Preview](./preview.png)

**Keywords:** AI transcription, speech-to-text, YouTube transcription, offline transcription, Whisper AI, voice recognition, audio transcription, video transcription, privacy-first AI, local AI, open source transcription

---

## ✨ Features

* 🎥 **YouTube & Local Support:** Transcribe directly from YouTube URLs or upload `.mp3`, `.mp4`, `.wav`, and more.
* 🚀 **Real-Time Progress:** Watch the transcription status live via **WebSockets** — no more guessing when it finishes.
* 📚 **Smart Library:** Manage your media and transcriptions in one place. Auto-correlates audio files with their transcripts and allows one-click cleanup of paired files.
* 🛡️ **Filename Sanitization:** Automatic handling of special characters, accents, and spaces in filenames for robust cross-platform compatibility.
* 🤖 **Dual AI Engine:**
  * **openai-whisper**: High accuracy (default when FFmpeg is available).
  * **faster-whisper**: Blazing fast inference with seamless fallback.
* ⚡ **GPU Acceleration:** Automatic **CUDA** support for NVIDIA GPUs, with robust CPU fallback.
* 🛠 **Zero-Config Setup:**
  * **Auto-FFmpeg:** Automatically detects or installs FFmpeg locally (Windows/Linux/Mac).
  * **yt-dlp Python API:** Robust media downloading without external binary dependencies.
* 🐳 **Production Ready:** Optimized **Docker** image (multi-stage build, non-root user, secure).
* 🌓 **Modern UI:** Dark/light mode, responsive design, history management, and file library.

---

## 🛠 Tech Stack

* **Backend:** Python 3.13+, FastAPI, Uvicorn, WebSockets
* **Frontend:** Jinja2 Templates, Alpine.js (Reactive UI), Tailwind CSS
* **AI & Media:** 
  * `openai-whisper` & `faster-whisper` (ASR Models)
  * `yt-dlp` (Python API)
  * `imageio-ffmpeg` (Auto-setup)
  * `torch` (PyTorch with CUDA 12.4 support)
* **DevOps:** Docker (Multi-stage), GitHub Actions (CI), Pytest
* **Storage:** Local filesystem with smart correlation (Library)

---

## 📂 Project Structure

```txt
app/
 ├─ main.py                # App entry point & router registration
 ├─ core/
 │   ├─ config.py          # Environment settings (Pydantic)
 │   └─ theme.py           # UI theming logic
 ├─ routers/
 │   ├─ home.py            # UI: Homepage
 │   ├─ upload.py          # API: Handle file/YouTube uploads
 │   ├─ websocket.py       # API: Real-time progress updates
 │   ├─ library.py         # API: Manage audio/transcription files
 │   └─ history.py         # UI: Transcription history
 ├─ services/
 │   ├─ progress.py        # Task state management
 │   ├─ youtube.py         # yt-dlp integration
 │   ├─ transcriber.py     # Whisper engine & FFmpeg logic
 │   └─ file_manager.py    # File I/O operations & Sanitization
 ├─ scripts/
 │   └─ setup_ffmpeg.py    # Auto-installer for FFmpeg
 ├─ templates/             # Jinja2 + Alpine.js templates
 └─ static/                # Assets
storage/
 ├─ uploads/               # Temporary media files
 └─ transcriptions/        # Generated .txt files
tests/                     # Unit & Integration tests
```

---

## ⚙️ Requirements

* **Python 3.10 to 3.12** (Highly Recommended for GPU support)
* **FFmpeg**: Installed automatically on first run (or via system PATH).
* **NVIDIA GPU** (Optional): For faster transcription.

> **⚠️ Important Note on Python 3.13+:** The official PyTorch binaries with CUDA support often lag behind the latest Python releases. If you are using Python 3.13 or 3.14, `pip install torch` might fallback to the CPU-only version. For the best experience with NVIDIA GPUs, please use Python 3.10, 3.11, or 3.12.

---

## 🚀 Quickstart

### Local Development

**Windows (PowerShell):**

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Run the automated installer (installs GPU support + dependencies)
python install.py

# Start the application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**macOS/Linux:**

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Run the automated installer
python install.py

# Start the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open: [http://localhost:8000](http://localhost:8000)

---

## 🐳 Run with Docker

Build an optimized, secure container:

```bash
docker build -t transcriber .
docker run --rm -p 8000:8000 transcriber
```

**For GPU Support (NVIDIA):**
Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) is installed.

```bash
docker run --rm --gpus all -p 8000:8000 transcriber
```

---

## 📖 Usage

1. **Home:** Paste a YouTube link or drag & drop an audio/video file.
2. **Process:** Watch the real-time progress bar as the AI processes your media.
3. **Result:** View the transcript instantly and download it as `.txt`.
4. **History:** Access your past transcriptions anytime.

**Key Endpoints:**

* `GET /` → Web Interface
* `POST /transcribe/youtube` → Start YouTube job
* `POST /transcribe/upload` → Start File job
* `WS /ws/progress/{task_id}` → Real-time status stream

---

## 🧪 Testing

The project uses `pytest` with extensive mocking to ensure stability without downloading large models/files during tests.

```bash
pytest
```

---


---

## 🧩 Troubleshooting

* **FFmpeg Error:** The app requires FFmpeg. The `install.py` script attempts to download it automatically. If it fails, you may need to install FFmpeg manually and add it to your system's PATH.
* **Slow Transcription / CPU instead of GPU:** If you have an NVIDIA GPU but the app is still using the CPU, it's likely that pip installed the CPU-only version of PyTorch. Run the included setup script to force-install the CUDA version:
  ```bash
  python install.py
  ```
* **WebSocket Error:** If behind a proxy (Nginx/Apache), ensure WebSocket upgrades are allowed.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

---

## 🔍 SEO & Discovery

This project is optimized for search engines with relevant keywords:
- AI transcription
- Speech-to-text
- YouTube transcription
- Offline transcription
- Whisper AI
- Voice recognition
- Audio transcription
- Video transcription
- Privacy-first AI
- Local AI processing
- Open source transcription
- Real-time transcription
- GPU-accelerated transcription

## 🌐 Live Demo

Try the live version at: [https://transcriber.zaidhassan.me](https://transcriber.zaidhassan.me)

## 📧 Contact

For questions or support, visit [zaidhassan.me](https://zaidhassan.me) or open an issue on GitHub.

---
