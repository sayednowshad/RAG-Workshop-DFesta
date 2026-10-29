
from app.ingestion import extract_text


def chunk_text(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[dict]:
    """
    Split extracted PDF text into smaller chunks.

    Args:
        pages: List of dictionaries containing page numbers and text.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters shared between chunks.

    Returns:
        A list of chunks with their IDs and source page numbers.
    """

    # Validate the chunking configuration.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    # The next chunk starts here, allowing adjacent chunks to overlap.
    step = chunk_size - chunk_overlap

    # Process each extracted PDF page.
    for page in pages:
        page_number = page["page"]
        text = page["text"].strip()

        # Skip pages that contain no readable text.
        if not text:
            continue

        # Split the page text into overlapping character windows.
        for start in range(0, len(text), step):
            chunk = text[start:start + chunk_size].strip()

            # Do not store empty chunks.
            if not chunk:
                continue

            # Save the chunk along with its source information.
            chunks.append({
                "chunk_id": f"page_{page_number}_chunk_{len(chunks) + 1}",
                "page": page_number,
                "text": chunk
            })

            # Stop when the end of this page has been reached.
            if start + chunk_size >= len(text):
                break

    return chunks


# Run this file independently to test the chunking stage.
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

        # Display a summary of the chunking results.
        print("\n" + "=" * 50)
        print("          TEXT CHUNKING RESULTS")
        print("=" * 50)

        print(f"Pages processed: {len(extracted_pages)}")
        print(f"Total chunks created: {len(chunks)}")
        print("Chunk size: 1000 characters")
        print("Chunk overlap: 200 characters")

        # Display one chunk so we can inspect the result.
        if chunks:
            print("\n--- First chunk ---")
            print(f"Chunk ID: {chunks[0]['chunk_id']}")
            print(f"Source page: {chunks[0]['page']}")
            print(f"Characters: {len(chunks[0]['text'])}")
            print(f"Text:\n{chunks[0]['text']}")

        print("\nCHUNKING STAGE COMPLETED")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")

#Limitations of this chunking approach:

# Fixed chunk size: The chunk size is set to a fixed number of characters (1000). 
# This may not be optimal for all types of text, as some chunks may end in the middle of a sentence or paragraph, leading to loss of context.
# That's Why we have chunk_overlap to preserve context.
# This gives us a better chance of preserving context across chunks, but it may still not be perfect.


# Q: Why do we need overlap in RAG?

# Ans: We repeat the last 200 characters of one chunk at the beginning of the next chunk.
# This helps preserve context when a sentence or explanation spans two chunks.

#Ex:
# Imagine a sentence is split between two chunks. Without overlap, the second chunk might lose
# the beginning of the explanation. With overlap, some of the previous text is preserved, helping the
# retrieval system understand the context.