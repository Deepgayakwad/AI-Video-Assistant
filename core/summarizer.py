from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

load_dotenv()


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


llm = get_llm()


def split_transcript(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """
    Generate a final meeting summary using map-reduce summarization.
    """

    chunks = split_transcript(transcript)

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant.

Summarize this transcript chunk into concise bullet points.

Focus on:
- Important discussions
- Decisions made
- Action items
- Deadlines
- Key takeaways

CRITICAL RULE: Only report a decision, action item, owner, or deadline if it
was explicitly stated in this chunk. Do not infer or invent one just because
the chunk discusses a topic that could plausibly lead to a decision or task.
If this chunk contains none of these, say so plainly rather than omitting
the point silently or fabricating one.
""",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    partial_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        partial_summaries.append(
            map_chain.invoke({"text": chunk})
        )

    combined_summary = "\n\n".join(partial_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant.

Combine the partial summaries into one final meeting summary.

CRITICAL RULE: Only include a decision or action item if it was explicitly
stated in the source material — a real task assigned to a real owner, or an
explicit decision that was actually made. Never invent owners, deadlines, or
decisions just to fill out a section, and never generalize a discussion
topic into a fake decision or task. If a section has nothing genuine to
report, write exactly "None identified in this discussion." under that
heading instead of fabricating content. It is completely normal and expected
for a conceptual, explanatory, or lecture-style discussion to have no formal
decisions or action items — do not force structure onto a discussion that
did not include them.

Return:

## Meeting Summary

## Key Discussion Points

## Decisions Taken

## Action Items

## Next Steps
""",
            ),
            ("human", "{text}"),
        ]
    )

    reduce_chain = (
        RunnableLambda(lambda x: {"text": x})
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    return reduce_chain.invoke(combined_summary)


def generate_title(transcript: str) -> str:
    """
    Generate a short meeting title.
    """

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate a professional meeting title.

Rules:
- Maximum 8 words
- No quotation marks
- No punctuation
- Return only the title
""",
            ),
            ("human", "{text}"),
        ]
    )

    title_chain = (
        RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:3000])