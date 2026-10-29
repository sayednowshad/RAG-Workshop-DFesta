
import json
from pathlib import Path

from app.vector_store import collection


# Output file for all stored embeddings.
OUTPUT_FILE = Path("exports/embeddings_metadata.json")


def export_all_embeddings():
    # Check how many records are stored in ChromaDB.
    total_records = collection.count()

    if total_records == 0:
        print("No embeddings found in ChromaDB.")
        print("Store your embeddings before exporting.")
        return

    print(f"Total records to export: {total_records}")

    # Fetch all records, including their actual embedding vectors.
    result = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    # Convert database records into JSON-compatible dictionaries.
    records = []

    for i, record_id in enumerate(result["ids"]):
        vector = result["embeddings"][i]

        records.append({
            "id": record_id,
            "text": result["documents"][i],
            "metadata": result["metadatas"][i],
            "embedding": (
                vector.tolist()
                if hasattr(vector, "tolist")
                else list(vector)
            )
        })

    # Create the exports folder if it does not exist.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Write all records into one formatted JSON file.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 45)
    print("       EMBEDDING EXPORT COMPLETED")
    print("=" * 45)
    print(f"Records exported: {len(records)}")
    print(f"File saved to: {OUTPUT_FILE.resolve()}")

    if records:
        print(f"Vector dimensions: {len(records[0]['embedding'])}")
        print(f"First record ID: {records[0]['id']}")
        print(f"Metadata: {records[0]['metadata']}")
        print(f"First 5 vector values: {records[0]['embedding'][:5]}")


if __name__ == "__main__":
    export_all_embeddings()


# Purpose:
# Export all existing ChromaDB embeddings, text, and metadata to JSON.
#
# Limitations:
# 1. JSON files can become large when exporting many vectors.
# 2. Embeddings are numerical vectors, not human-readable explanations.
# 3. This exports existing vectors; it does not generate new embeddings.
