
from fastapi import FastAPI, HTTPException

# Import our pipeline stages.
from app.ingestion import extract_text
from app.chunking import chunk_text

# Create the FastAPI application.
app = FastAPI(
    title="RAG Workshop API",
    description="A step-by-step Retrieval-Augmented Generation pipeline",
    version="1.0.0"
)

# PDF path relative to the project root.
PDF_PATH = "data/Atomic_habits.pdf"


@app.get("/")
def home():
    """Check whether the API is running."""
    return {
        "message": "RAG Workshop API is running!"
    }

# --------------- Ingestion Stage ---------------

@app.post("/ingest")
def ingest_document():
    """Extract text from the PDF."""
    try:
        pages = extract_text(PDF_PATH)

        return {
            "message": "Document ingestion completed",
            "pages_with_text": len(pages),
            "characters_extracted": sum(
                len(page["text"]) for page in pages
            )
        }

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

# --------------- Chunking Stage ---------------

@app.post("/chunk")
def chunk_document():
    """Extract PDF text and split it into chunks."""
    try:
        # Stage 1: Extract text from the PDF.
        pages = extract_text(PDF_PATH)

        # Stage 2: Split the extracted text into chunks.
        chunks = chunk_text(
            pages,
            chunk_size=1000,
            chunk_overlap=200
        )

        return {
            "message": "Document chunking completed",
            "pages_processed": len(pages),
            "total_chunks": len(chunks),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "sample_chunk": chunks[0] if chunks else None
        }

    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
