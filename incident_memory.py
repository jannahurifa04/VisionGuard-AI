import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="visionguard_memory"
)

def add_memory(text):
    collection.add(
        documents=[text],
        ids=[str(collection.count() + 1)]
    )

def search_memory(query, n_results=5):
    result = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return result["documents"][0]