"""
Run once (or re-run when books change) to populate ChromaDB.
Usage: python ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env", override=True)

BOOKS_DIR = Path(os.getenv("BOOKS_DIR", "../books"))
CHROMA_DIR = Path("./chroma_db")
COLLECTION_NAME = "wisdom_books"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 64

print(f"Loading embedding model: {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Model loaded.")


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"id": f"{source}_{idx}", "text": chunk, "source": source})
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks


def ingest():
    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {BOOKS_DIR.resolve()}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  {f.name}")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for pdf_path in pdf_files:
        source = pdf_path.stem
        print(f"\nProcessing: {pdf_path.name}")
        text = extract_text(pdf_path)
        if not text.strip():
            print("  Warning: no text extracted (scanned PDF?), skipping.")
            continue

        chunks = chunk_text(text, source)
        print(f"  {len(chunks)} chunks")

        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            texts = [c["text"] for c in batch]
            embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

            collection.add(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{"source": c["source"]} for c in batch],
            )
            print(f"  Embedded chunks {i}–{i + len(batch) - 1}", end="\r")

        total_chunks += len(chunks)
        print(f"  Done — {len(chunks)} chunks stored.        ")

    print(f"\nIngestion complete. Total chunks in DB: {total_chunks}")


if __name__ == "__main__":
    ingest()
