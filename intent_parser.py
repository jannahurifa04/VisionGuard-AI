import json
import re
import requests

MODEL_NAME = "qwen2.5:7b"


def default_intent():
    return {
        "event_type": "ANY",
        "date_mode": "today",
        "days_ago": None,
        "months_ago": None,
        "person_id": None,
        "is_chat": False
    }


def extract_json(text):
    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end == 0:
            return None

        json_text = text[start:end]
        return json.loads(json_text)

    except Exception:
        return None


def fallback_parse(question):
    q = question.lower().strip()
    intent = default_intent()

    if q in ["hi", "hello", "hey", "hai", "halo"]:
        intent["is_chat"] = True
        intent["date_mode"] = "none"
        return intent

    if "loiter" in q or "loitr" in q:
        intent["event_type"] = "LOITERING"
    elif "enter" in q or "visitor" in q:
        intent["event_type"] = "PERSON_ENTERED"
    elif "left" in q or "leave" in q:
        intent["event_type"] = "PERSON_LEFT"
    elif "present" in q:
        intent["event_type"] = "PERSON_PRESENT"

    if "latest" in q:
        intent["date_mode"] = "latest"
    elif "yesterday" in q:
        intent["date_mode"] = "yesterday"
    elif "last week" in q or "lst wek" in q:
        intent["date_mode"] = "last_week"
    elif "last month" in q:
        intent["date_mode"] = "last_month"
    elif "last year" in q:
        intent["date_mode"] = "last_year"

    day_match = re.search(r"(\d+)\s*(day|days|dys|dy)\s*(ago|a go)?", q)
    if day_match:
        intent["date_mode"] = "days_ago"
        intent["days_ago"] = int(day_match.group(1))

    month_match = re.search(r"(\d+)\s*(month|months|mnth|mnts)\s*(ago|a go)?", q)
    if month_match:
        intent["date_mode"] = "months_ago"
        intent["months_ago"] = int(month_match.group(1))

    person_match = re.search(r"person\s*id\s*(\d+)", q)
    if person_match:
        intent["person_id"] = int(person_match.group(1))
        if intent["date_mode"] == "today" and "today" not in q:
            intent["date_mode"] = "all"

    return intent


def normalize_intent(intent):
    clean = default_intent()

    if not isinstance(intent, dict):
        return clean

    valid_events = ["LOITERING", "PERSON_ENTERED", "PERSON_PRESENT", "PERSON_LEFT", "ANY"]
    valid_dates = [
        "none", "today", "yesterday", "days_ago", "months_ago",
        "last_week", "last_month", "last_year", "latest", "all"
    ]

    clean["event_type"] = intent.get("event_type", "ANY")
    if clean["event_type"] not in valid_events:
        clean["event_type"] = "ANY"

    clean["date_mode"] = intent.get("date_mode", "today")
    if clean["date_mode"] not in valid_dates:
        clean["date_mode"] = "today"

    clean["days_ago"] = intent.get("days_ago")
    clean["months_ago"] = intent.get("months_ago")
    clean["person_id"] = intent.get("person_id")
    clean["is_chat"] = bool(intent.get("is_chat", False))

    try:
        if clean["days_ago"] is not None:
            clean["days_ago"] = int(clean["days_ago"])
    except Exception:
        clean["days_ago"] = None

    try:
        if clean["months_ago"] is not None:
            clean["months_ago"] = int(clean["months_ago"])
    except Exception:
        clean["months_ago"] = None

    try:
        if clean["person_id"] is not None:
            clean["person_id"] = int(clean["person_id"])
    except Exception:
        clean["person_id"] = None

    return clean


def parse_intent(question):
    q = question.strip()

    prompt = f"""
You are the intent parser for VisionGuard AI, a CCTV security assistant.

The user may have typos.
Understand natural language like ChatGPT.

Convert the user question into JSON only.

Event types:
- LOITERING
- PERSON_ENTERED
- PERSON_PRESENT
- PERSON_LEFT
- ANY

Date modes:
- none
- today
- yesterday
- days_ago
- months_ago
- last_week
- last_month
- last_year
- latest
- all

Rules:
- If user says hi/hello only, set is_chat true and date_mode none.
- If user asks about "loitring", "loiterng", "loitering", use LOITERING.
- If user says "dys ago", "dy ago", or "days ago", use days_ago.
- If user asks "tell me about person id 1", use person_id 1 and date_mode all.
- If user asks latest, use date_mode latest.
- Return JSON only. No markdown. No explanation.

User question:
{q}

Return exactly this JSON shape:
{{
  "event_type": "ANY",
  "date_mode": "today",
  "days_ago": null,
  "months_ago": null,
  "person_id": null,
  "is_chat": false
}}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=60
        )

        raw = response.json().get("response", "").strip()
        parsed = extract_json(raw)

        if parsed:
            return normalize_intent(parsed)

        return fallback_parse(q)

    except Exception:
        return fallback_parse(q)