import chromadb

# Create a client - use PersistentClient instead of just Client
client = chromadb.PersistentClient(path="./chroma_db")

# Create a test collection
collection = client.get_or_create_collection(name="test_collection")

# Add a single test document
collection.add(
    documents=["This is a test document about Salesforce CRM"],
    ids=["test1"],
    metadatas=[{"source": "test"}]
)

# Try a test query
results = collection.query(
    query_texts=["salesforce"],
    n_results=1
)

print("Query Results:", results)