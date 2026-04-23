"""
Run once (or re-run when books change) to populate Pinecone.
Usage: python ingest.py
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pypdf import PdfReader

load_dotenv(Path(__file__).parent / ".env", override=True)

BOOKS_DIR = Path(os.getenv("BOOKS_DIR", str(Path(__file__).parent.parent / "books")))
INDEX_NAME = "wisdom-books"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "jina-embeddings-v2-base-en"
EMBED_BATCH = 64
UPSERT_BATCH = 100

JINA_API_KEY = os.getenv("JINA_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)


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


def embed_batch(texts: list[str]) -> list[list[float]]:
    import time
    for attempt in range(8):
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
            json={"model": EMBED_MODEL, "input": texts},
        )
        if response.status_code == 429:
            wait = 2 ** attempt
            print(f"\n  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    raise RuntimeError("Max retries exceeded on Jina AI rate limit")


def ingest():
    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {BOOKS_DIR.resolve()}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s):")
    for f in pdf_files:
        print(f"  {f.name}")

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
            batch = chunks[i: i + EMBED_BATCH]
            embeddings = embed_batch([c["text"] for c in batch])

            vectors = [
                {
                    "id": c["id"],
                    "values": emb,
                    "metadata": {"source": c["source"], "text": c["text"]},
                }
                for c, emb in zip(batch, embeddings)
            ]

            for j in range(0, len(vectors), UPSERT_BATCH):
                index.upsert(vectors=vectors[j: j + UPSERT_BATCH])

            print(f"  Embedded chunks {i}–{i + len(batch) - 1}", end="\r")

        total_chunks += len(chunks)
        print(f"  Done — {len(chunks)} chunks stored.        ")

    stats = index.describe_index_stats()
    print(f"\nIngestion complete. Total vectors in Pinecone: {stats['total_vector_count']}")


if __name__ == "__main__":
    ingest()
