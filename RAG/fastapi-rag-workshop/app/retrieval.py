
from app.embedding import model, MODEL_NAME
from app.vector_store import search_chunks, collection


# The number of retrieved chunks can be adjusted as needed.
def retrieve_chunks(query: str, top_k: int = 3) -> list[dict]:
    """
    Find the most relevant document chunks for a user's question.

    Args:
        query: The user's question.
        top_k: Maximum number of relevant chunks to retrieve.

    Returns:
        A list of matching chunks with text, source page, and distance.
    """

    # Validate the user's query.
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Check whether document embeddings have already been stored.
    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB is empty. Store document embeddings first."
        )

    # -------- Stage 1: Convert the user's question into a vector. --------

    # Use the same embedding model as the stored document embeddings.
    # This ensures that query and document vectors are compatible.
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # -------- Stage 2: Search ChromaDB for similar document vectors. --------

    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k
    )

    retrieved_chunks = []

    # -------- Stage 3: Format retrieved results for inspection. --------

    # ChromaDB groups search results by query.
    # We submitted one query, so we access the first result group.
    if not results["ids"] or not results["ids"][0]:
        return retrieved_chunks

    for i, chunk_id in enumerate(results["ids"][0]):
        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


# Run this file independently to demonstrate retrieval.
if __name__ == "__main__":
    question = "What is the current stock price of NVIDIA, and will its share price increase tomorrow?"

    try:
        print("\n" + "=" * 50)
        print("          RAG RETRIEVAL RESULTS")
        print("=" * 50)

        print(f"User question: {question}")
        print(f"Embedding model: {MODEL_NAME}")
        print(f"Stored records: {collection.count()}")

        # Retrieve the three most relevant chunks.
        results = retrieve_chunks(question, top_k=3)

        if not results:
            print("\nNo relevant chunks were returned.")

        # Display the retrieved chunks and their source information.
        for rank, chunk in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Source page: {chunk['metadata']['page']}")
            print(f"Cosine distance: {chunk['distance']:.4f}")
            print(f"Text:\n{chunk['text'][:500]}")

        print("\nRETRIEVAL STAGE COMPLETED")

    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")


# ------------ Limitations of this retrieval approach ---------------

# 1. Semantic matching:
#    Similarity search may return related text that does not fully
#    answer the user's question.
#
# 2. Top-k selection:
#    Retrieving too few chunks may omit useful context. Retrieving
#    too many can introduce irrelevant information.
#
# 3. Embedding dependency:
#    Query and document embeddings must use the same compatible model.
#
# 4. No answer generation:
#    This stage retrieves information but does not generate a final
#    natural-language answer using an LLM.


# -------------- Why do we need retrieval in RAG? -------------

# Ans: Retrieval finds relevant document chunks based on the meaning
# of a user's query. These chunks provide context for the LLM to use
# when generating a grounded answer.


# ------- Note --------

# Lower cosine distance generally indicates greater similarity.
# Distance is not a probability or a guaranteed relevance score.
# The top_k parameter controls the maximum number of chunks returned.
