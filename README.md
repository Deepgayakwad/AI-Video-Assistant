# 🎬 AI Video Assistant

An intelligent meeting assistant powered by **Mistral AI** and **Sarvam AI** that transcribes, summarises, and lets you chat with your video/audio meetings.

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔊 **Audio Extraction** | Download YouTube videos or use local MP4/MP3 files |
| 📝 **Transcription** | Powered by Sarvam AI (supports English & Hinglish) |
| 📋 **Summarisation** | AI-generated concise meeting summaries |
| ✅ **Action Items** | Automatically extract tasks and owners |
| 🔑 **Key Decisions** | Highlight important decisions made |
| ❓ **Open Questions** | Surface unresolved questions |
| 💬 **RAG Chat** | Ask anything about your meeting via Retrieval-Augmented Generation |
| ⬇ **Export** | Download summary and transcript as `.txt` |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your PATH
- A [Mistral AI](https://console.mistral.ai/) API key
- A [Sarvam AI](https://dashboard.sarvam.ai/) API key

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-video-assistant.git
cd ai-video-assistant

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

### Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🗂 Project Structure

```
ai-video-assistant/
├── app.py               # Main Streamlit application
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
│
├── core/
│   ├── extractor.py     # Action items, decisions & questions extraction
│   ├── rag_engine.py    # RAG chain (ChromaDB + LangChain)
│   ├── summarizer.py    # Meeting summarisation
│   ├── transcriber.py   # Audio transcription via Sarvam AI
│   └── vector_store.py  # Vector DB management
│
└── utils/
    └── audio_processor.py  # YouTube download & audio chunking
```

---

## ⚙️ Environment Variables

Copy `.env.example` → `.env` and configure:

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral AI API key |
| `SARVAM_API_KEY` | Your Sarvam AI API key |
| `SARVAM_STT_MODEL` | STT model to use (default: `saaras:v2.5`) |

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set your **Secrets** in the Streamlit Cloud dashboard:
   ```toml
   MISTRAL_API_KEY = "your_key"
   SARVAM_API_KEY = "your_key"
   SARVAM_STT_MODEL = "saaras:v2.5"
   ```
5. Click **Deploy** 🎉

---

## 🛠 Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[Mistral AI](https://mistral.ai)** — LLM for summarisation & chat
- **[Sarvam AI](https://sarvam.ai)** — Indian language speech-to-text
- **[LangChain](https://langchain.com)** — RAG pipeline orchestration
- **[ChromaDB](https://trychroma.com)** — Vector store
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — YouTube audio extraction
- **[Faster Whisper](https://github.com/SYSTRAN/faster-whisper)** — Fast transcription fallback

---

## 📄 License

MIT © 2026
