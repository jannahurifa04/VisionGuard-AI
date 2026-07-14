import sqlite3
from memory import search_memory

def build_rag_context(question):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_time, object_name, person_id, shirt_color
        FROM events
        ORDER BY id DESC
        LIMIT 20
    """)

    events = cursor.fetchall()
    conn.close()

    context = "Recent Events:\n"

    for event in events:
        time, event_name, pid, color = event
        context += f"- {time}: {event_name}, Person {pid}, Shirt {color}\n"

    memory_results = search_memory(question, limit=5)

    context += "\nMemory:\n"

    if memory_results:
        docs = memory_results.get("documents", [[]])[0]
        for doc in docs:
            context += f"- {doc}\n"

    return context