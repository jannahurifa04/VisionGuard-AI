from intent_parser import parse_intent

questions = [
    "hi",
    "any loitering 3 months a go?",
    "what happened 2 days ago?",
    "when is the latest loitering?",
    "tell me about person id 1",
    "who entered yesterday?"
]

for q in questions:
    print("=" * 50)
    print(q)
    print(parse_intent(q))