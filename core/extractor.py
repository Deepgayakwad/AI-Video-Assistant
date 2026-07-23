from dotenv import load_dotenv
import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()


# =====================================================
# LLM
# =====================================================

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
)


# =====================================================
# Generic Chain Builder
# =====================================================

def build_chain(system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    return (
        RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


# =====================================================
# Action Items
# =====================================================

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract every action item from the meeting transcript.

For each action item provide:

1. Task
2. Owner
3. Deadline

If Owner or Deadline is not mentioned, write "Not specified".

Return the result in this format:

1.
Task:
Owner:
Deadline:

If there are no action items, return:

No action items found.
"""
    )

    return chain.invoke(transcript)


# =====================================================
# Key Decisions
# =====================================================

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all important decisions made during the meeting.

For each decision include:

- Decision
- Reason (if mentioned)

Return as a numbered list.

If no decisions were made, return:

No key decisions found.
"""
    )

    return chain.invoke(transcript)


# =====================================================
# Open Questions / Follow-ups
# =====================================================

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract:

- Unanswered questions
- Pending discussions
- Risks
- Blockers
- Topics requiring follow-up

Return as a numbered list.

If none exist, return:

No open questions found.
"""
    )

    return chain.invoke(transcript)


# =====================================================
# Extract Everything
# =====================================================

def extract_all(transcript: str) -> dict:
    """
    Returns all meeting insights in one dictionary.
    """

    print("Extracting Action Items...")
    action_items = extract_action_items(transcript)

    print("Extracting Key Decisions...")
    decisions = extract_key_decisions(transcript)

    print("Extracting Open Questions...")
    questions = extract_questions(transcript)

    return {
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
    }


# =====================================================
# Test
# =====================================================

if __name__ == "__main__":

    transcript = """
    John will prepare the project report by Friday.

    Sarah will review the budget.

    The team decided to migrate to AWS.

    We still need to finalize the deployment date.

    Should we hire another backend developer?
    """

    results = extract_all(transcript)

    print("\n========== ACTION ITEMS ==========\n")
    print(results["action_items"])

    print("\n========== DECISIONS ==========\n")
    print(results["decisions"])

    print("\n========== QUESTIONS ==========\n")
    print(results["questions"])