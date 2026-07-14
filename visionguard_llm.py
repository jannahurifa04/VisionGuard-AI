import requests
from smart_search import smart_search_events, format_events_for_llm
from memory import search_memory
from rag_engine import build_rag_context

MODEL_NAME = "qwen2.5:7b"


def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            },
            timeout=90
        )
        return response.json().get("response", "").strip()

    except Exception as e:
        return f"Ollama error: {e}"


def is_small_talk(q):
    q = q.lower().strip()
    return q in ["hi", "hello", "hey", "hai", "halo"]


def clean_answer(answer):
    banned_phrases = [
        "the user wants",
        "i will analyze",
        "i need to analyze",
        "based on my thinking",
        "let me analyze",
        "step-by-step",
        "malicious intent",
        "crime",
        "criminal",
        "home address",
    ]

    cleaned = answer.strip()

    for phrase in banned_phrases:
        cleaned = cleaned.replace(phrase, "")
        cleaned = cleaned.replace(phrase.capitalize(), "")

    return cleaned.replace("\n", "<br>")


def get_memory_context(question, limit=5):
    try:
        memory_results = search_memory(question, limit=limit)
        memory_docs = memory_results.get("documents", [[]])[0]

        if not memory_docs:
            return "No relevant long-term memory found."

        memory_context = ""

        for doc in memory_docs:
            memory_context += doc + "\n"

        return memory_context

    except Exception as e:
        return f"Memory search error: {e}"


def ask_rag_llm(question):
    q = question.strip()

    if is_small_talk(q):
        return (
            "Hi 👋 I am VisionGuard AI Assistant.<br>"
            "You can talk to me naturally. I specialize in CCTV events, "
            "loitering, person ID, evidence, memory, watchlist, and security summaries."
        )

    result = smart_search_events(q)
    rows = result["rows"]
    database_context = format_events_for_llm(result)

    memory_context = get_memory_context(q, limit=5)
    rag_context = build_rag_context(q)

    has_memory = (
        memory_context
        and memory_context != "No relevant long-term memory found."
        and not memory_context.startswith("Memory search error")
    )

    if not rows and not has_memory:
        return (
            f"No CCTV records were found for {result['date_label']}.<br><br>"
            "There are no matching database events or long-term memory records."
        )

    prompt = f"""
You are VisionGuard AI Assistant, a CCTV security investigation assistant.

You have three trusted sources:
1. Exact CCTV database records from SQLite.
2. Relevant long-term memory from ChromaDB.
3. Additional verified RAG context.

STRICT RULES:
- Use ONLY the CCTV database records, memory, and RAG context shown below.
- NEVER mention information not explicitly present in database, memory, or RAG context.
- NEVER mention addresses unless exact address exists in records.
- NEVER speculate about crime, malicious intent, motive, identity, gender, or guilt.
- If uncertain, reply: "Not enough CCTV evidence."
- Do NOT invent events.
- Do NOT invent security guards.
- Do NOT invent reports from staff.
- Do NOT invent locations.
- Do NOT invent names.
- Do NOT say someone is guilty or dangerous.
- Do NOT say "the user wants".
- Do NOT say "I will analyze".
- Do NOT show your reasoning process.
- Do NOT mention chain-of-thought.
- If Person ID is None, say "unknown person".
- Treat shirt color as approximate because camera lighting may be inaccurate.
- Use concise security language.
- Use HTML <br> for line breaks.
- Answer directly.

HOW TO ANSWER:
- First give a short direct answer.
- Then list key CCTV evidence.
- Include exact Person ID, event type, timestamp, snapshot, and video clip when available.
- For risk or suspicious activity, focus only on:
  loitering count, repeated visits, watchlist status, timestamps, snapshots, and video clips.
- Do not make conclusions beyond the records.

User question:
{q}

Search metadata:
Date range: {result['date_label']}
Event type: {result['event_type']}
Person ID: {result['person_id']}

Exact CCTV database records:
{database_context}

Relevant long-term memory:
{memory_context}

Additional verified RAG context:
{rag_context}

Final answer:
"""

    answer = ask_ollama(prompt)

    if not answer:
        return "No response from the local LLM."

    return clean_answer(answer)