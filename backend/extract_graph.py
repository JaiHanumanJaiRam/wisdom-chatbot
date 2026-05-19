"""
Build a knowledge graph from the books/ directory.

Reads each PDF, chunks it with the same parameters as ingest.py so chunk IDs
match Pinecone exactly, then calls Claude Haiku to extract philosophical
entities (Concept, Person, Book, Tradition) and directed relations between them.

The graph is saved to backend/knowledge_graph.json and loaded by main.py at
startup to enrich retrieval with relationship-aware context.

Usage:  python extract_graph.py
Re-run: safe to re-run — merges into existing graph file if present.
"""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
import anthropic
import networkx as nx
from networkx.readwrite import json_graph

load_dotenv(Path(__file__).parent / ".env", override=True)

BOOKS_DIR   = Path(os.getenv("BOOKS_DIR", str(Path(__file__).parent.parent / "books")))
GRAPH_FILE  = Path(__file__).parent / "knowledge_graph.json"

# Must match ingest.py so chunk IDs align with Pinecone
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
BATCH_SIZE    = 4    # chunks per Claude Haiku call

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Same catalog as ingest.py — book identity for each PDF stem
BOOK_CATALOG = {
    "8283_text": {"book": "Brihadaranyaka Upanishad", "tradition": "Advaita Vedanta", "abbreviation": "BrU"},
    "Ashtavakra Gita by Swami Nityaswarupananda": {"book": "Ashtavakra Gita", "tradition": "Advaita Vedanta", "abbreviation": "AG"},
    "Chandogya_Upanishad-1942": {"book": "Chandogya Upanishad", "tradition": "Advaita Vedanta", "abbreviation": "ChU"},
    "Gitapress_Gita_Roman": {"book": "Bhagavad Gita", "tradition": "Advaita Vedanta", "abbreviation": "BG"},
    "Mandukya Upanishad - Gita Press Gorakhpur": {"book": "Mandukya Upanishad", "tradition": "Advaita Vedanta", "abbreviation": "MaU"},
    "Mandukya Upanishad with Anandagiri tika_text": {"book": "Mandukya Upanishad with Anandagiri Tika", "tradition": "Advaita Vedanta", "abbreviation": "MaU-AG"},
    "Mandukya Upanishad1": {"book": "Mandukya Upanishad", "tradition": "Advaita Vedanta", "abbreviation": "MaU"},
    "Sacred Books of the Hindus Vol 01 - Upanishads - Mundaka to Katha with Madhwa Bhashya_text": {"book": "Mundaka and Katha Upanishads", "tradition": "Advaita Vedanta", "abbreviation": "MuKaU"},
    "Sacred Books of the Hindus Vol 03 - Chandogya Upanishad 86-94_text": {"book": "Chandogya Upanishad", "tradition": "Advaita Vedanta", "abbreviation": "ChU"},
    "Sacred Books of the Hindus Vol 22 - Studies in Vedanta Sutras and the Upanishads_text": {"book": "Studies in Vedanta Sutras and the Upanishads", "tradition": "Advaita Vedanta", "abbreviation": "SBH22"},
    "princialupanisad00radh": {"book": "The Principal Upanishads", "tradition": "Advaita Vedanta", "abbreviation": "PU"},
}
_FALLBACK = {"book": "Unknown", "tradition": "Hindu Philosophy", "abbreviation": "UNK"}


# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)


def make_chunks(text, source_id):
    """Same chunking as ingest.py — chunk IDs will match Pinecone."""
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"id": f"{source_id}_{idx}", "text": chunk})
            idx += 1
        start = end - CHUNK_OVERLAP
    return chunks


def is_readable(text):
    if not text:
        return False
    printable = sum(c.isalpha() or c.isspace() for c in text)
    return (printable / len(text)) > 0.6


# ── Graph extraction via Claude Haiku ────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are building a knowledge graph of Hindu and Kashmir Shaivism philosophy.

From the text chunks below, extract:
1. ENTITIES — philosophical concepts, people, books, or traditions.
   Normalise to lowercase English (e.g. "Self" → "atman", "the Absolute" → "brahman").
   Types: Concept | Person | Book | Tradition

2. RELATIONS — directed relationships between entities found in the text.
   Use precise predicates such as:
   is_identical_to, is_aspect_of, is_manifestation_of, causes, negates,
   transcends, reveals, is_realized_through, authored, belongs_to,
   is_commentary_on, is_contrasted_with, leads_to, is_obstacle_to

Return ONLY valid JSON — no markdown, no explanation:
{
  "entities": [{"name": "atman", "type": "Concept"}],
  "relations": [{"subject": "atman", "predicate": "is_identical_to", "object": "brahman"}]
}

TEXT CHUNKS:
"""


def extract_graph_from_batch(chunks):
    """
    Send a batch of chunks to Claude Haiku and return extracted
    entities and relations, each tagged with the originating chunk IDs.
    """
    numbered = "\n\n".join(
        f"[chunk_id: {c['id']}]\n{c['text'][:600]}" for c in chunks
    )
    prompt = EXTRACTION_PROMPT + numbered

    for attempt in range(3):
        try:
            r = claude.messages.create(
                model="claude-haiku-4-5",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = r.content[0].text.strip()
            # strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())

            entities  = data.get("entities",  [])
            relations = data.get("relations", [])

            # tag each item with all chunk IDs in this batch
            batch_ids = [c["id"] for c in chunks]
            for e in entities:
                e["chunk_ids"] = batch_ids
            for rel in relations:
                rel["chunk_ids"] = batch_ids

            return entities, relations

        except Exception as ex:
            print(f"  attempt {attempt+1} failed: {ex}")
            time.sleep(2 ** attempt)

    return [], []


# ── Graph build / merge ───────────────────────────────────────────────────────

def load_graph():
    """Load existing graph from disk, or return an empty DiGraph."""
    if GRAPH_FILE.exists():
        data = json.loads(GRAPH_FILE.read_text())
        return json_graph.node_link_graph(data)
    return nx.DiGraph()


def save_graph(G):
    data = json_graph.node_link_data(G)
    GRAPH_FILE.write_text(json.dumps(data, indent=2))


def add_entity(G, name, etype, chunk_ids):
    name = name.lower().strip()
    if not name:
        return
    if G.has_node(name):
        existing = set(G.nodes[name].get("chunk_ids", []))
        G.nodes[name]["chunk_ids"] = list(existing | set(chunk_ids))
    else:
        G.add_node(name, type=etype, chunk_ids=list(chunk_ids))


def add_relation(G, subject, predicate, obj, chunk_ids):
    s = subject.lower().strip()
    o = obj.lower().strip()
    if not s or not o or s == o:
        return
    # ensure both nodes exist
    for n in (s, o):
        if not G.has_node(n):
            G.add_node(n, type="Concept", chunk_ids=list(chunk_ids))
    if G.has_edge(s, o):
        existing = set(G[s][o].get("chunk_ids", []))
        G[s][o]["chunk_ids"] = list(existing | set(chunk_ids))
        # keep the most specific predicate (first seen)
    else:
        G.add_edge(s, o, predicate=predicate, chunk_ids=list(chunk_ids))


# ── Main ──────────────────────────────────────────────────────────────────────

PROGRESS_FILE = Path(__file__).parent / "graph_progress.json"


def load_progress():
    """Track which books are fully processed so re-runs skip them."""
    if PROGRESS_FILE.exists():
        return set(json.loads(PROGRESS_FILE.read_text()))
    return set()


def save_progress(done_set):
    PROGRESS_FILE.write_text(json.dumps(list(done_set)))


def build():
    G        = load_graph()
    done     = load_progress()
    print(f"Loaded existing graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Already completed: {done or 'none'}")

    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {BOOKS_DIR}")
        sys.exit(1)

    total_relations = 0

    for pdf_path in pdf_files:
        source_id = pdf_path.stem
        meta      = BOOK_CATALOG.get(source_id, _FALLBACK)

        if source_id in done:
            print(f"\n── {meta['book']} — already done, skipping.")
            continue

        print(f"\n── {meta['book']} ({pdf_path.name})")

        text = extract_text(pdf_path)
        if not text.strip():
            print("   Scanned PDF — skipping.")
            done.add(source_id)
            save_progress(done)
            continue

        chunks = make_chunks(text, source_id)
        chunks = [c for c in chunks if is_readable(c["text"])]
        print(f"   {len(chunks)} readable chunks")

        book_relations = 0
        print(f"   Extracting graph...", end=" ", flush=True)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch            = chunks[i: i + BATCH_SIZE]
            entities, relations = extract_graph_from_batch(batch)

            for e in entities:
                add_entity(G, e.get("name",""), e.get("type","Concept"), e.get("chunk_ids",[]))
            for rel in relations:
                add_relation(G, rel.get("subject",""), rel.get("predicate","related_to"),
                             rel.get("object",""), rel.get("chunk_ids",[]))
                book_relations += 1

            print(".", end="", flush=True)

        print(f" done — {book_relations} relations extracted")
        total_relations += book_relations

        # save after every book so progress survives a credit outage
        save_graph(G)
        done.add(source_id)
        save_progress(done)
        print(f"   Graph saved ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    print(f"\nAll books processed.")
    print(f"Final graph — Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    build()
