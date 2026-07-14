from incident_memory import search_memory

results = search_memory("loitering")

print("\nRESULTS:")
for r in results:
    print("-" * 50)
    print(r)