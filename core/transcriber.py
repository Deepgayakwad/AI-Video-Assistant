import os
import subprocess

import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

# =====================================================
# CONFIGURATION
# =====================================================

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None


# =====================================================
# LOAD FASTER WHISPER MODEL
# =====================================================

def load_model():
    global _model

    if _model is None:
        print(f"\nLoading Faster-Whisper model: {WHISPER_MODEL}...")

        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
        )

        print("Using CPU.")
        print("Faster-Whisper model loaded.\n")

    return _model


# =====================================================
# FASTER WHISPER TRANSCRIPTION
# =====================================================

def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = load_model()

    segments, info = model.transcribe(
        chunk_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    return transcript.strip()


# =====================================================
# SARVAM HELPERS
# =====================================================

def _send_to_sarvam(piece_path: str) -> str:
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav",
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\nSarvam Error ({response.status_code})")
        print(response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


# =====================================================
# SARVAM TRANSCRIPTION
# =====================================================

def _get_duration(wav_path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr.decode(errors='replace')}")
    return float(result.stdout.decode().strip())


def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not found in .env")

    total_seconds = _get_duration(chunk_path)
    piece_seconds = SARVAM_PIECE_SECONDS
    transcript = ""
    i = 0
    start = 0.0
    total = int(total_seconds / piece_seconds) + 1

    while start < total_seconds:
        piece_path = f"{chunk_path}_piece_{i}.wav"

        # Extract piece using ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(piece_seconds),
            "-i", chunk_path,
            "-ac", "1",
            "-ar", "16000",
            piece_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg piece extraction failed at piece {i}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )

        try:
            print(f"Processing Sarvam piece {i+1}/{total}...")
            transcript += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

        start += piece_seconds
        i += 1

    return transcript.strip()


# =====================================================
# ROUTER
# =====================================================

def transcribe_chunk(chunk_path: str, language: str = "english") -> str:

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# =====================================================
# TRANSCRIBE ALL CHUNKS
# =====================================================

def transcribe_all(chunks: list[str], language: str = "english") -> str:

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Faster Whisper"

    print(f"\nUsing {engine}\n")

    transcript = ""

    total = len(chunks)

    for i, chunk in enumerate(chunks):

        print(f"Chunk {i+1}/{total}")

        text = transcribe_chunk(chunk, language)

        transcript += text + " "

    print("\nTranscription Complete.\n")

    return transcript.strip()


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    chunks = [
        "sample_chunk.wav"
    ]

    text = transcribe_all(
        chunks,
        language="english"
    )

    print(text)