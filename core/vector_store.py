import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# Configuration
# =====================================================

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =====================================================
# Embedding Model
# =====================================================

def get_embeddings():
    """
    Load HuggingFace embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =====================================================
# Build Vector Store
# =====================================================

def build_vector_store(transcript: str) -> Chroma:
    """
    Build a Chroma vector database from the meeting transcript.
    """

    print("\nBuilding Vector Store...")

    # Remove previous vector database
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_text(transcript)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": i + 1,
                "source": "meeting_transcript",
            },
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Indexed {len(documents)} chunks.")
    print("Vector Store created successfully.\n")

    return vector_store


# =====================================================
# Load Existing Vector Store
# =====================================================

def load_vector_store() -> Chroma:
    """
    Load an existing Chroma database.
    """

    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            "Vector database not found. "
            "Run build_vector_store() first."
        )

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


# =====================================================
# Retriever
# =====================================================

def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Create a retriever from the vector database.
    """

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# =====================================================
# Search Helper
# =====================================================

def search_documents(query: str, k: int = 4):
    """
    Search the vector database directly.
    """

    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        query,
        k=k,
    )

    return docs


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    transcript = """
    John will prepare the project report by Friday.

    Sarah will review the budget.

    The team decided to migrate to AWS.

    The deployment date is still undecided.

    Another backend developer may be hired.
    """

    db = build_vector_store(transcript)

    print("\nSearching...\n")

    results = search_documents("Who will prepare the report?")

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}")
        print(doc.page_content)
        print("-" * 60)