
from sentence_transformers import SentenceTransformer
from app.chunking import chunk_text
from app.ingestion import extract_text


# Load the embedding model.
# all-MiniLM-L6-v2 converts text into 384-dimensional vectors.
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Convert text chunks into numerical embedding vectors.

    Args:
        chunks: List of dictionaries containing chunk IDs, page numbers,
                and text.

    Returns:
        A list of chunks with their corresponding embedding vectors.
    """

    # Return an empty list if there are no chunks.
    if not chunks:
        return []

    # Extract the text from each chunk for embedding generation.
    texts = [chunk["text"] for chunk in chunks]

    # Convert all chunk texts into numerical vectors.
    # normalize_embeddings makes each vector have unit length.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embedded_chunks = []

    # Attach each generated vector to its original chunk metadata.
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "embedding": embedding.tolist()
        })

    return embedded_chunks


# Run this file independently to test the embedding stage.
if __name__ == "__main__":
    try:
        # Stage 1: Extract text from the PDF.
        extracted_pages = extract_text("data/Atomic_habits.pdf")

        # Stage 2: Split the extracted text into chunks.
        chunks = chunk_text(
            extracted_pages,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Stage 3: Convert chunks into numerical vectors.
        embedded_chunks = generate_embeddings(chunks)

        # Display a summary of embedding results.
        print("\n" + "=" * 50)
        print("          EMBEDDING RESULTS")
        print("=" * 50)

        print(f"Chunks processed: {len(embedded_chunks)}")
        print(f"Embedding model: {MODEL_NAME}")

        if embedded_chunks:
            first_embedding = embedded_chunks[0]["embedding"]

            print(f"Vector dimensions: {len(first_embedding)}")
            print(f"First chunk ID: {embedded_chunks[0]['chunk_id']}")
            print(f"Source page: {embedded_chunks[0]['page']}")
            print(f"First 10 vector values: {first_embedding[:10]}")

        print("\nEMBEDDING STAGE COMPLETED")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")



# ----------- "Limitations of this embedding approach" -----------------
#
# 1. Model dependency:
#    The quality of embeddings depends on the selected model.
#
# 2. Computational cost:
#    Generating embeddings for many chunks takes time and memory.
#
# 3. Semantic limitations:
#    Embeddings represent semantic patterns, but they do not guarantee
#    that every meaning or relationship is captured correctly.
#
# 4. Model compatibility:
#    When searching later, query embeddings must use the same model
#    and compatible preprocessing as the document embeddings.

# ------------------------ Why do we need "embeddings" in RAG? ---------------------

# Ans: Embeddings convert text into numerical vectors so that the system
# can compare semantic similarity between document chunks and user queries.
# This helps retrieve relevant information even when the wording differs.
#
# Example:
# "How can I build good habits?"
# "How can I develop positive routines?"