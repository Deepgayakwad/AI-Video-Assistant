import os
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)

load_dotenv()

# =====================================================
# LLM
# =====================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.3,
)


# =====================================================
# Prompt
# =====================================================

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert AI Meeting Assistant.

Answer the user's question ONLY using the meeting transcript context.

Rules:
- Do not make up information.
- If the answer is not present, reply:
  "I could not find this information in the meeting transcript."
- Keep answers concise.
- Mention people's names whenever possible.
- If multiple people discussed the topic, summarize their viewpoints.

Meeting Transcript Context:

{context}
""",
        ),
        ("human", "{question}"),
    ]
)


# =====================================================
# Helper
# =====================================================

def format_docs(docs):
    """
    Combine retrieved documents into one context string.
    """
    return "\n\n--------------------\n\n".join(
        doc.page_content for doc in docs
    )


# =====================================================
# Build New RAG Chain
# =====================================================

def build_rag_chain(transcript: str):
    """
    Build a new vector store from the transcript
    and return a RAG chain.
    """

    print("Building Vector Store...")

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =====================================================
# Load Existing RAG Chain
# =====================================================

def load_rag_chain():
    """
    Load an already existing vector database.
    """

    print("Loading Vector Store...")

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =====================================================
# Ask Question
# =====================================================

def ask_question(rag_chain, question: str) -> str:
    """
    Ask a question about the meeting transcript.
    """

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print("\nAnswer:")
    print(answer)

    return answer


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    transcript = """
    John will prepare the project report before Friday.

    Sarah will review the budget.

    The team decided to migrate the application to AWS.

    The deployment date is still undecided.

    Amit suggested hiring another backend developer.
    """

    rag = build_rag_chain(transcript)

    print()

    ask_question(
        rag,
        "Who will prepare the project report?"
    )

    ask_question(
        rag,
        "What decision was made?"
    )

    ask_question(
        rag,
        "When is the deployment?"
    )