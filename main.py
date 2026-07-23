import time
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import (
    build_rag_chain,
    ask_question,
)

load_dotenv()


# =====================================================
# Main Pipeline
# =====================================================

def run_pipeline(source: str, language: str = "english") -> dict:
    """
    Complete AI Meeting Assistant Pipeline.
    """

    start_time = time.time()

    print("\n" + "=" * 60)
    print("🚀 Starting AI Video Assistant")
    print("=" * 60)

    # -------------------------------------------------
    # Audio Processing
    # -------------------------------------------------

    print("\n🎧 Processing Audio...\n")
    chunks = process_input(source)

    # -------------------------------------------------
    # Transcription
    # -------------------------------------------------

    print("\n🎙️ Transcribing Meeting...\n")

    transcript = transcribe_all(
        chunks,
        language=language,
    )

    print("\n✅ Transcription Completed.\n")

    print("Transcript Preview:\n")
    print(transcript[:300] + "...\n")

    with open(
        "transcript.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(transcript)

    # -------------------------------------------------
    # Title
    # -------------------------------------------------

    print("📝 Generating Meeting Title...\n")
    title = generate_title(transcript)

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print("📄 Generating Summary...\n")
    summary = summarize(transcript)

    with open(
        "summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)

    # -------------------------------------------------
    # Information Extraction
    # -------------------------------------------------

    print("📌 Extracting Action Items...\n")
    action_items = extract_action_items(transcript)

    print("📌 Extracting Key Decisions...\n")
    key_decisions = extract_key_decisions(transcript)

    print("📌 Extracting Open Questions...\n")
    open_questions = extract_questions(transcript)

    # -------------------------------------------------
    # RAG
    # -------------------------------------------------

    print("📚 Building Vector Store...\n")

    rag_chain = build_rag_chain(transcript)

    elapsed = time.time() - start_time

    print(f"\n✅ Pipeline completed in {elapsed:.2f} seconds.")

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    try:

        source = input(
            "Enter YouTube URL or Local File Path:\n> "
        ).strip()

        language = input(
            "Language (english / hinglish): "
        ).strip().lower()

        if not language:
            language = "english"

        result = run_pipeline(
            source,
            language,
        )

        print("\n" + "=" * 70)
        print("📋 AI Meeting Report")
        print("=" * 70)

        print(f"\n📌 Meeting Title\n")
        print(result["title"])

        print("\n" + "-" * 70)

        print("\n📝 Meeting Summary\n")
        print(result["summary"])

        print("\n" + "-" * 70)

        print("\n✅ Action Items\n")
        print(result["action_items"])

        print("\n" + "-" * 70)

        print("\n📌 Key Decisions\n")
        print(result["key_decisions"])

        print("\n" + "-" * 70)

        print("\n❓ Open Questions\n")
        print(result["open_questions"])

        print("\n" + "=" * 70)

        # =====================================================
        # Chat Mode
        # =====================================================

        print("\n💬 Chat with your meeting")
        print("Type 'exit' to quit.\n")

        rag_chain = result["rag_chain"]

        while True:

            question = input("You: ").strip()

            if question.lower() in [
                "exit",
                "quit",
                "q",
            ]:
                print("\n👋 Goodbye!")
                break

            if not question:
                continue

            answer = ask_question(
                rag_chain,
                question,
            )

            print(f"\n🤖 Assistant:\n{answer}\n")

    except KeyboardInterrupt:

        print("\n\nProgram Interrupted.")

    except Exception as e:

        print("\n❌ Error:")
        print(e)