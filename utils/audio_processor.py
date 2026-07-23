import os
import subprocess
import yt_dlp

# Directory to store downloaded audio
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def _ffmpeg_to_wav(input_path: str, output_path: str) -> str:
    """
    Use ffmpeg subprocess to convert any audio/video file to
    16 kHz mono WAV. Avoids pydub / audioop dependency that was
    removed in Python 3.13+.
    """
    cmd = [
        "ffmpeg",
        "-y",                  # overwrite output without asking
        "-i", input_path,
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz sample rate (Whisper-friendly)
        "-vn",                 # strip video
        output_path,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg conversion failed:\n{result.stderr.decode(errors='replace')}"
        )
    return output_path


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to WAV format.
    Returns the path to the downloaded WAV file.
    """
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        # Use Android client to bypass 403 blocks on cloud server IPs
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Linux; Android 13; Pixel 7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Mobile Safari/537.36"
            ),
        },
        "quiet": True,
        "noplaylist": True,
        "no_warnings": False,
        "age_limit": None,
        "geo_bypass": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_file = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    return wav_file


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio/video file to 16 kHz mono WAV
    using ffmpeg directly (no pydub / audioop).
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    return _ffmpeg_to_wav(input_path, output_path)


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """
    Split a WAV file into chunks of `chunk_minutes` using ffmpeg.
    Returns a list of chunk file paths.
    """
    # Get total duration via ffprobe
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        wav_path,
    ]
    probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if probe.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed:\n{probe.stderr.decode(errors='replace')}"
        )

    total_seconds = float(probe.stdout.decode().strip())
    chunk_seconds = chunk_minutes * 60
    base_name = os.path.splitext(wav_path)[0]
    chunks = []

    start = 0.0
    i = 1
    while start < total_seconds:
        chunk_path = f"{base_name}_chunk_{i}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-i", wav_path,
            "-ac", "1",
            "-ar", "16000",
            chunk_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg chunking failed at chunk {i}:\n"
                f"{result.stderr.decode(errors='replace')}"
            )
        chunks.append(chunk_path)
        start += chunk_seconds
        i += 1

    return chunks


def process_input(source: str) -> list:
    """
    Accepts either:
    - A YouTube URL
    - A local audio/video file

    Returns:
        List of WAV chunk paths.
    """
    if source.startswith(("http://", "https://")):
        print("🎥 Detected YouTube URL.")
        print("⬇️ Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")

        print("📁 Detected local file.")
        print("🔄 Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("✂️ Chunking audio...")
    chunks = chunk_audio(wav_path)

    print(f"✅ Audio ready! Created {len(chunks)} chunk(s).")

    return chunks


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()

    try:
        chunks = process_input(source)

        print("\nGenerated Chunks:")
        for chunk in chunks:
            print(chunk)

    except Exception as e:
        print(f"\n❌ Error: {e}")