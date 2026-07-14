from incident_memory import add_memory, search_memory

add_memory(
    "Person ID 1 loitered near entrance on 2026-06-08 wearing gray shirt."
)

add_memory(
    "Person ID 2 entered building at 09:15."
)

print(search_memory("Who loitered?"))