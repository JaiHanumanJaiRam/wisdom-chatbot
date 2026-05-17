"""
Run once (or re-run when books change) to populate Pinecone with
structured metadata: book, author, tradition, text_type, citation,
concepts, and neighbour chunk IDs for context stitching.

Usage: python ingest.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pypdf import PdfReader
import anthropic

load_dotenv(Path(__file__).parent / ".env", override=True)

BOOKS_DIR   = Path(os.getenv("BOOKS_DIR", str(Path(__file__).parent.parent / "books")))
INDEX_NAME  = "wisdom-books"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 100
EMBED_MODEL = "jina-embeddings-v2-base-en"
EMBED_BATCH = 64
UPSERT_BATCH = 100
CONCEPT_BATCH = 5   # chunks per Claude Haiku call for concept extraction

JINA_API_KEY = os.getenv("JINA_API_KEY")
pc           = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index        = pc.Index(INDEX_NAME)
claude       = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Book catalog ──────────────────────────────────────────────────────────────
# Maps PDF filename stem → structured fields stored in every chunk's metadata.
# Add a new entry here whenever a new PDF is added to the books/ directory.

BOOK_CATALOG = {
    "8283_text": {
        "book":        "Brihadaranyaka Upanishad",
        "author":      "Traditional (Vedic)",
        "translator":  "Swami Madhvananda",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"BrU",
    },
    "Ashtavakra Gita by Swami Nityaswarupananda": {
        "book":        "Ashtavakra Gita",
        "author":      "Traditional (attributed to Ashtavakra)",
        "translator":  "Swami Nityaswarupananda",
        "tradition":   "Advaita Vedanta",
        "text_type":   "verse",
        "abbreviation":"AG",
    },
    "Chandogya_Upanishad-1942": {
        "book":        "Chandogya Upanishad",
        "author":      "Traditional (Vedic)",
        "translator":  "Unknown (1942 edition)",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"ChU",
    },
    "Gitapress_Gita_Roman": {
        "book":        "Bhagavad Gita",
        "author":      "Vyasa",
        "translator":  "Gita Press, Gorakhpur",
        "tradition":   "Advaita Vedanta",
        "text_type":   "verse",
        "abbreviation":"BG",
    },
    "Mandukya Upanishad - Gita Press Gorakhpur": {
        "book":        "Mandukya Upanishad",
        "author":      "Traditional (Vedic)",
        "translator":  "Gita Press, Gorakhpur",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"MaU",
    },
    "Mandukya Upanishad with Anandagiri tika_text": {
        "book":        "Mandukya Upanishad with Anandagiri Tika",
        "author":      "Traditional (Vedic); commentary by Anandagiri",
        "translator":  "Unknown",
        "tradition":   "Advaita Vedanta",
        "text_type":   "commentary",
        "abbreviation":"MaU-AG",
    },
    "Mandukya Upanishad1": {
        "book":        "Mandukya Upanishad",
        "author":      "Traditional (Vedic)",
        "translator":  "Unknown",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"MaU",
    },
    "Sacred Books of the Hindus Vol 01 - Upanishads - Mundaka to Katha with Madhwa Bhashya_text": {
        "book":        "Mundaka and Katha Upanishads",
        "author":      "Traditional (Vedic)",
        "translator":  "Sacred Books of the Hindus series",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"MuKaU",
    },
    "Sacred Books of the Hindus Vol 03 - Chandogya Upanishad 86-94_text": {
        "book":        "Chandogya Upanishad",
        "author":      "Traditional (Vedic)",
        "translator":  "Sacred Books of the Hindus series",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"ChU",
    },
    "Sacred Books of the Hindus Vol 22 - Studies in Vedanta Sutras and the Upanishads_text": {
        "book":        "Studies in Vedanta Sutras and the Upanishads",
        "author":      "Various",
        "translator":  "Sacred Books of the Hindus series",
        "tradition":   "Advaita Vedanta",
        "text_type":   "commentary",
        "abbreviation":"SBH22",
    },
    "princialupanisad00radh": {
        "book":        "The Principal Upanishads",
        "author":      "Traditional (Vedic)",
        "translator":  "S. Radhakrishnan",
        "tradition":   "Advaita Vedanta",
        "text_type":   "prose",
        "abbreviation":"PU",
    },
}

_FALLBACK_META = {
    "book":        "Unknown",
    "author":      "Unknown",
    "translator":  "Unknown",
    "tradition":   "Hindu Philosophy",
    "text_type":   "prose",
    "abbreviation":"UNK",
}


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ── Chunking with positional metadata ────────────────────────────────────────

def chunk_text(text, source_id, book_meta):
    """Split text into overlapping chunks, recording neighbour IDs for stitching."""
    raw_chunks = []
    start = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            raw_chunks.append(chunk)
        start = end - CHUNK_OVERLAP

    total = len(raw_chunks)
    chunks = []
    for idx, text_chunk in enumerate(raw_chunks):
        chunk_id = f"{source_id}_{idx}"
        chunks.append({
            "id":            chunk_id,
            "text":          text_chunk,
            # bibliographic
            "source":        book_meta["book"],
            "book":          book_meta["book"],
            "author":        book_meta["author"],
            "translator":    book_meta["translator"],
            "tradition":     book_meta["tradition"],
            "text_type":     book_meta["text_type"],
            "abbreviation":  book_meta["abbreviation"],
            # citation placeholder (will be overwritten if verse refs are found)
            "citation":      book_meta["abbreviation"],
            # positional
            "chunk_index":   idx,
            "total_chunks":  total,
            "prev_chunk_id": f"{source_id}_{idx - 1}" if idx > 0          else "",
            "next_chunk_id": f"{source_id}_{idx + 1}" if idx < total - 1  else "",
        })
    return chunks


# ── Concept extraction via Claude Haiku ──────────────────────────────────────

def _clean_json(raw):
    """Strip markdown code fences and whitespace before parsing JSON."""
    raw = raw.strip()
    # remove ```json ... ``` or ``` ... ``` wrapping
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_concepts_batch(texts):
    """
    Extract 4-6 philosophical/Sanskrit concepts from each chunk.
    Sends CONCEPT_BATCH chunks per Claude Haiku call to keep costs low.
    Returns a list of concept lists, one per input text.
    Falls back to empty list per chunk if the text is too garbled to parse.
    """
    # Skip chunks that are mostly garbage characters (e.g. bad OCR)
    def is_readable(t):
        printable = sum(c.isalpha() or c.isspace() for c in t)
        return len(t) > 0 and (printable / len(t)) > 0.6

    if not any(is_readable(t) for t in texts):
        return [[] for _ in texts]

    numbered = "\n\n".join(f"CHUNK {i+1}:\n{t[:600]}" for i, t in enumerate(texts))
    prompt = (
        f"Below are {len(texts)} text chunk(s) from Hindu sacred texts. "
        "For each chunk, extract 4-6 key philosophical concepts or Sanskrit terms "
        "(e.g. atman, brahman, maya, karma, dharma, moksha, consciousness, self). "
        "Respond with ONLY a valid JSON array of arrays — one inner array per chunk — "
        "with lowercase English strings. No markdown, no explanation.\n"
        "Example for 2 chunks: [[\"atman\",\"self\",\"eternal\"],[\"maya\",\"illusion\",\"world\"]]\n\n"
        + numbered
    )
    for attempt in range(3):
        try:
            response = claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            raw    = _clean_json(response.content[0].text)
            result = json.loads(raw)
            if isinstance(result, list) and len(result) == len(texts):
                return result
            # if lengths mismatch, pad or truncate to match
            result.extend([[] for _ in range(len(texts) - len(result))])
            return result[:len(texts)]
        except Exception as e:
            print(f"  Concept extraction attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return [[] for _ in texts]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(texts):
    for attempt in range(8):
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization":  f"Bearer {JINA_API_KEY}",
                "Content-Type":   "application/json",
            },
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


# ── Main ingestion loop ───────────────────────────────────────────────────────

def ingest():
    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {BOOKS_DIR.resolve()}")
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s) in {BOOKS_DIR.resolve()}:")
    for f in pdf_files:
        known = "✓" if f.stem in BOOK_CATALOG else "? (using fallback metadata)"
        print(f"  {f.name}  {known}")

    total_chunks = 0

    for pdf_path in pdf_files:
        source_id = pdf_path.stem
        book_meta = BOOK_CATALOG.get(source_id, _FALLBACK_META)

        print(f"\n── {book_meta['book']} ──")
        print(f"   File: {pdf_path.name}")
        print(f"   Tradition: {book_meta['tradition']}  |  Type: {book_meta['text_type']}")

        text = extract_text(pdf_path)
        if not text.strip():
            print("   Warning: no text extracted (scanned PDF?), skipping.")
            continue

        chunks = chunk_text(text, source_id, book_meta)
        print(f"   {len(chunks)} chunks")

        # ── concept extraction in batches ──
        print(f"   Extracting concepts via Claude Haiku...", end=" ", flush=True)
        all_concepts = []
        for i in range(0, len(chunks), CONCEPT_BATCH):
            batch_texts = [c["text"] for c in chunks[i: i + CONCEPT_BATCH]]
            concepts    = extract_concepts_batch(batch_texts)
            all_concepts.extend(concepts)
            print(".", end="", flush=True)
        print(f" done ({len(all_concepts)} concept lists)")

        # attach concepts to chunks — ensure flat list of strings only
        for chunk, concepts in zip(chunks, all_concepts):
            if isinstance(concepts, list):
                chunk["concepts"] = [c for c in concepts if isinstance(c, str)]
            else:
                chunk["concepts"] = []

        # ── embed + upsert ──
        for i in range(0, len(chunks), EMBED_BATCH):
            batch      = chunks[i: i + EMBED_BATCH]
            embeddings = embed_batch([c["text"] for c in batch])

            vectors = []
            for c, emb in zip(batch, embeddings):
                vectors.append({
                    "id":     c["id"],
                    "values": emb,
                    "metadata": {
                        "text":          c["text"],
                        "source":        c["source"],
                        "book":          c["book"],
                        "author":        c["author"],
                        "translator":    c["translator"],
                        "tradition":     c["tradition"],
                        "text_type":     c["text_type"],
                        "abbreviation":  c["abbreviation"],
                        "citation":      c["citation"],
                        "chunk_index":   c["chunk_index"],
                        "total_chunks":  c["total_chunks"],
                        "prev_chunk_id": c["prev_chunk_id"],
                        "next_chunk_id": c["next_chunk_id"],
                        "concepts":      c["concepts"],
                    },
                })

            for j in range(0, len(vectors), UPSERT_BATCH):
                batch_v = vectors[j: j + UPSERT_BATCH]
                for attempt in range(3):
                    try:
                        index.upsert(vectors=batch_v)
                        break
                    except Exception as ue:
                        print(f"\n  Upsert error (attempt {attempt+1}): {ue}")
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)

            print(f"   Embedded & upserted chunks {i}–{i + len(batch) - 1}", end="\r")

        total_chunks += len(chunks)
        print(f"   Done — {len(chunks)} chunks stored.                    ")

    stats = index.describe_index_stats()
    print(f"\nIngestion complete.")
    print(f"Total vectors in Pinecone: {stats['total_vector_count']}")


if __name__ == "__main__":
    ingest()
