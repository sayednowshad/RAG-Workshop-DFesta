
from pathlib import Path
import pymupdf


def extract_text(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF.

    Returns:
        A list of dictionaries containing page numbers and extracted text.
    """

    file_path = Path(pdf_path)

    # 1. Validate the PDF file
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path.resolve()}"
        )

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("The provided file must be a PDF.")

    extracted_pages = []
    total_characters = 0

    print("\n" + "=" * 50)
    print("       RAG PIPELINE - DOCUMENT INGESTION")
    print("=" * 50)

    print(f"PDF file: {file_path.resolve()}")

    # 2. Open the PDF
    with pymupdf.open(file_path) as document:
        total_pages = len(document)
        print(f"Total PDF pages: {total_pages}")

        # 3. Extract text page by page
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "page": page_number,
                    "text": text
                })

                total_characters += len(text)

    # 4. Display extraction results
    print(f"Pages containing text: {len(extracted_pages)}")
    print(f"Total characters extracted: {total_characters}")

    print("\n--- Sample extracted text (first page with text) ---")

    if extracted_pages:
        first_page = extracted_pages[0]
        print(f"Page number: {first_page['page']}")
        print(first_page["text"][:1000])
    else:
        print("No readable text was found in this PDF.")

    print("\n" + "=" * 50)
    print("       INGESTION STAGE COMPLETED")
    print("=" * 50 + "\n")

    return extracted_pages


# Run this file directly from the project root
if __name__ == "__main__":
    pdf_path = "data/Atomic_habits.pdf"

    try:
        pages = extract_text(pdf_path)
        print(f"Successfully extracted {len(pages)} pages.")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")

# ---------------- What this File Do? ---------------

# 1. Opens the PDF:  (Reads data/Atomic_habits.pdf using PyMuPDF).

# 2. Reads each page: (Goes through the PDF page by page and extracts readable text).

# 3. Collects the extracted text: (Stores each page's number and text in a Python dictionary).

# 4. Displays the results: Prints the "total pages", "pages containing text", "character count",
# and a "sample of the extracted text".