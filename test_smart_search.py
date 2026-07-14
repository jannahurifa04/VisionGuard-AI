from smart_search import smart_search_events, format_events_for_llm

questions = [
    "hi",
    "any loitering today?",
    "any loitering yesterday?",
    "any loitering 2 days ago?",
    "any loitering 1 month ago?",
    "what happened last week?",
    "tell me about person id 1",
    "tell me about person id 1 today",
    "when is the latest loitering?",
    "any loitering 3 months a go?"
]

for q in questions:
    print("=" * 60)
    print("QUESTION:", q)

    result = smart_search_events(q)

    print("INTENT:", result.get("intent"))
    print("DATE:", result["date_label"])
    print("EVENT TYPE:", result["event_type"])
    print("PERSON ID:", result["person_id"])
    print("FOUND:", len(result["rows"]))

    print(format_events_for_llm(result))
    print()