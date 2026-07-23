import os
import yt_dlp
from pydub import AudioSegment

# Directory to store downloaded audio
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL and convert it to WAV format.
    Returns the path to the downloaded WAV file.
    """
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_file = os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"

    return wav_file


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio/video file to
    16 kHz mono WAV (Whisper-friendly).
    """
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """
    Split a WAV file into chunks of `chunk_minutes`.
    Returns a list of chunk file paths.
    """
    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    base_name = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i + 1}.wav"

        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list[str]:
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