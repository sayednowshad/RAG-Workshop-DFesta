from pathlib import Path
import chromadb

# Purpose: Store and search document chunks, embeddings, and metadata.
# ChromaDB supports similarity search for RAG applications.
# Persistent storage keeps records available between program runs.

DB_PATH = Path("chroma_db")
COLLECTION_NAME = "atomic_habits"

# Create a persistent database client.
client = chromadb.PersistentClient(path=str(DB_PATH))

# Get the collection or create it if it doesn't exist.
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(embedded_chunks: list[dict]) -> int:
    """Store already-generated embeddings, text, and metadata."""

    if not embedded_chunks:
        return 0

    ids = [chunk["chunk_id"] for chunk in embedded_chunks]
    documents = [chunk["text"] for chunk in embedded_chunks]
    embeddings = [chunk["embedding"] for chunk in embedded_chunks]
    metadatas = [{"page": chunk["page"]} for chunk in embedded_chunks]

    # Upsert inserts new records or updates records with matching IDs.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(embedded_chunks)


def search_chunks(query_embedding: list[float], top_k: int = 3) -> dict:
    """Retrieve the most similar stored chunks for a query vector."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if collection.count() == 0:
        return {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }

    # Search the stored vectors without regenerating document embeddings.
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )


# Run this file to inspect the existing ChromaDB database.
# This block does NOT rerun ingestion, chunking, or embeddings.
if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("          CHROMADB VECTOR STORE")
    print("=" * 50)

    print(f"Database location: {DB_PATH.resolve()}")
    print(f"Collection name: {COLLECTION_NAME}")

    # Display how many records are currently stored.
    total_records = collection.count()
    print(f"Total stored records: {total_records}")

    if total_records > 0:

        # Retrieve one existing record for demonstration.
        sample = collection.get(
            limit=1,
            include=["documents", "metadatas"]
        )

        print("\n--- Sample stored record ---")
        print(f"Chunk ID: {sample['ids'][0]}")
        print(f"Metadata: {sample['metadatas'][0]}")
        print(f"Stored text: {sample['documents'][0][:200]}")

        print("\nStorage verification: SUCCESS")

    else:
        print("\nNo records found in this collection.")
        print("Store embedded chunks before inspecting the database.")

    print("\nVECTOR STORE INSPECTION COMPLETED")
    print("=" * 50)



# Limitations:
# 1. Local storage is tied to this machine unless the database is deployed.
# 2. Query embeddings must use the same compatible embedding model.
# 3. Changes to the source PDF require an appropriate update strategy.
#
# Why ChromaDB?
# It persists embeddings and their associated text and metadata,
# allowing the RAG system to retrieve relevant document passages.
