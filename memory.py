import chromadb

# ==========================
# VisionGuard Memory System
# ==========================

client = chromadb.PersistentClient(
    path="visionguard_memory"
)

collection = client.get_or_create_collection(
    name="incident_memory"
)


# ==========================
# SAVE MEMORY
# ==========================

def save_incident_memory(event_id, text, metadata=None):

    if metadata is None:
        metadata = {}

    collection.add(
        ids=[str(event_id)],
        documents=[text],
        metadatas=[metadata]
    )

    print(f"[MEMORY SAVED] Event ID {event_id}")


# ==========================
# SEARCH MEMORY
# ==========================

def search_memory(question, limit=5):

    results = collection.query(
        query_texts=[question],
        n_results=limit
    )

    return results


# ==========================
# PERSON PROFILE
# ==========================

def get_person_profile(person_id):

    try:
        results = collection.get(
            where={
                "person_id": str(person_id)
            }
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if len(documents) == 0:
            return f"No memory found for Person ID {person_id}"

        visits = 0
        loitering = 0
        shirt_colors = []

        for meta in metadatas:

            visits += 1

            if meta.get("event") == "LOITERING":
                loitering += 1

            shirt = meta.get("shirt_color")

            if shirt and shirt != "None":
                shirt_colors.append(shirt)

        if len(shirt_colors) > 0:
            most_common_shirt = max(
                set(shirt_colors),
                key=shirt_colors.count
            )
        else:
            most_common_shirt = "Unknown"

        profile = f"""
Person ID {person_id}

Visits: {visits}
Loitering Incidents: {loitering}
Most Common Shirt: {most_common_shirt}
"""

        return profile

    except Exception as e:

        return f"Memory error: {str(e)}"


# ==========================
# PERSON TIMELINE
# ==========================

def get_person_timeline(person_id):

    try:

        results = collection.get(
            where={
                "person_id": str(person_id)
            }
        )

        docs = results.get("documents", [])

        if len(docs) == 0:
            return f"No timeline found for Person ID {person_id}"

        timeline = f"Timeline for Person ID {person_id}\n\n"

        for item in docs:
            timeline += f"- {item}\n"

        return timeline

    except Exception as e:

        return f"Timeline error: {str(e)}"