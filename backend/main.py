import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import anthropic
import networkx as nx
from networkx.readwrite import json_graph

load_dotenv(Path(__file__).parent / ".env", override=True)

INDEX_NAME  = "wisdom-books"
EMBED_MODEL = "jina-embeddings-v2-base-en"
TOP_K       = 6
GRAPH_FILE  = Path(__file__).parent / "knowledge_graph.json"

JINA_API_KEY     = os.getenv("JINA_API_KEY")
pc               = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index            = pc.Index(INDEX_NAME)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://suneelkaw.com",
        "https://www.suneelkaw.com",
        "http://localhost:5173",
        "http://localhost:5200",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

SYSTEM_PROMPT = """You are a wisdom guide specialising in Vedantic and Hindu philosophical texts.
You answer ONLY using the context passages provided below from the sacred books.
If the answer cannot be found in the provided context, respond with:
"I cannot find an answer to that in the texts I have been given."

Do not use any outside knowledge. Do not speculate beyond the texts.
When a concept map is provided, use it to explain how ideas connect — but
still ground every claim in the text passages that follow it.
Cite the source book and citation reference for key points when possible."""


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: list = []


class ChatResponse(BaseModel):
    answer: str
    sources: list


# ── Knowledge graph ───────────────────────────────────────────────────────────

def load_knowledge_graph():
    """Load graph from disk. Returns empty DiGraph if file not found."""
    if GRAPH_FILE.exists():
        try:
            data = json.loads(GRAPH_FILE.read_text())
            G    = json_graph.node_link_graph(data)
            print(f"Knowledge graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
        except Exception as e:
            print(f"Warning: could not load knowledge graph: {e}")
    print("Knowledge graph not found — running without it. Run extract_graph.py to build it.")
    return nx.DiGraph()


# Load once at startup
knowledge_graph = load_knowledge_graph()


_COMMON_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "not", "but", "have", "what", "who", "how", "when", "which", "can",
    "man", "men", "one", "two", "all", "any", "its", "his", "her", "our",
    "you", "him", "has", "had", "did", "may", "say", "said", "also", "than",
    "then", "them", "they", "will", "been", "does", "into", "more", "some",
}

def find_entities_in_question(question):
    """
    Match question words/phrases against known graph nodes.
    Filters out short tokens and common English words to avoid noise.
    Returns a list of matched node names.
    """
    if knowledge_graph.number_of_nodes() == 0:
        return []

    q = question.lower()
    tokens  = q.split()
    bigrams = {f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)}
    search_terms = set(tokens) | bigrams

    matched = []
    for node in knowledge_graph.nodes():
        # skip very short nodes and common English words
        if len(node) < 4 or node in _COMMON_WORDS:
            continue
        if node in search_terms or node in q:
            matched.append(node)
    return matched


def graph_traverse(entities, max_hops=1, max_chunk_ids=3):
    """
    Starting from matched entities, traverse the graph up to max_hops.
    Returns:
      - relationship_lines: human-readable list of discovered relations
      - extra_chunk_ids:    Pinecone chunk IDs linked to traversed nodes/edges
    """
    if not entities or knowledge_graph.number_of_nodes() == 0:
        return [], []

    visited      = set()
    rel_lines    = []
    chunk_id_set = set()

    frontier = [e for e in entities if knowledge_graph.has_node(e)]

    for _ in range(max_hops + 1):
        next_frontier = []
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)

            # collect chunk IDs from this node
            for cid in knowledge_graph.nodes[node].get("chunk_ids", []):
                chunk_id_set.add(cid)

            # traverse outgoing edges
            for _, neighbour, data in knowledge_graph.out_edges(node, data=True):
                predicate = data.get("predicate", "related_to")
                rel_lines.append(f"{node} → {predicate} → {neighbour}")
                for cid in data.get("chunk_ids", []):
                    chunk_id_set.add(cid)
                if neighbour not in visited:
                    next_frontier.append(neighbour)

            # traverse incoming edges
            for source, _, data in knowledge_graph.in_edges(node, data=True):
                predicate = data.get("predicate", "related_to")
                rel_lines.append(f"{source} → {predicate} → {node}")
                for cid in data.get("chunk_ids", []):
                    chunk_id_set.add(cid)
                if source not in visited:
                    next_frontier.append(source)

        frontier = next_frontier

    # cap extra chunk IDs to avoid bloating context
    extra_ids = list(chunk_id_set)[:max_chunk_ids * 4]
    return rel_lines, extra_ids


def fetch_chunks_by_ids(chunk_ids, already_seen_ids):
    """Fetch specific chunks from Pinecone, skipping ones already retrieved."""
    new_ids = [cid for cid in chunk_ids if cid not in already_seen_ids]
    if not new_ids:
        return []
    try:
        result = index.fetch(ids=new_ids[:10])   # cap at 10 fetches
        chunks = []
        for vid, vec in result.vectors.items():
            meta = vec.metadata
            citation = meta.get("citation") or meta.get("book") or "Unknown"
            chunks.append({
                "id":       vid,
                "citation": citation,
                "book":     meta.get("book") or meta.get("source", "Unknown"),
                "text":     meta.get("text", ""),
            })
        return chunks
    except Exception:
        return []


# ── Metadata pre-filter ───────────────────────────────────────────────────────

_TRADITION_KEYWORDS = {
    "Kashmir Shaivism": [
        "kashmir shaivism", "kashmir shaiva", "trika", "shaivism",
        "siva sutras", "shiva sutras", "spanda", "pratyabhijna",
        "abhinavagupta", "lal ded", "lalla", "vijnana bhairava",
        "tantrasara", "paratrishika", "spanda karikas",
    ],
    "Advaita Vedanta": [
        "vedanta", "advaita", "upanishad", "bhagavad gita", "gita",
        "ashtavakra", "brahman", "vedantic", "shankaracharya",
    ],
}

_BOOK_KEYWORDS = {
    "Bhagavad Gita":              ["bhagavad gita", "gita", "krishna said", "arjuna"],
    "Ashtavakra Gita":            ["ashtavakra"],
    "Mandukya Upanishad":         ["mandukya"],
    "Chandogya Upanishad":        ["chandogya"],
    "Brihadaranyaka Upanishad":   ["brihadaranyaka"],
    "The Principal Upanishads":   ["radhakrishnan"],
}


def detect_filter(question):
    q = question.lower()
    for book, keywords in _BOOK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return {"book": {"$eq": book}}
    for tradition, keywords in _TRADITION_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return {"tradition": {"$eq": tradition}}
    return None


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(text):
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": [text]},
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


# ── Neighbour stitching ───────────────────────────────────────────────────────

def fetch_neighbours(matches):
    neighbour_ids = []
    for m in matches:
        meta = m["metadata"]
        if meta.get("prev_chunk_id"):
            neighbour_ids.append(meta["prev_chunk_id"])
        if meta.get("next_chunk_id"):
            neighbour_ids.append(meta["next_chunk_id"])
    if not neighbour_ids:
        return {}
    try:
        fetched = index.fetch(ids=neighbour_ids)
        return {vid: v.metadata.get("text", "") for vid, v in fetched.vectors.items()}
    except Exception:
        return {}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question):
    """
    1. Metadata pre-filter by tradition / book
    2. Vector similarity search
    3. Neighbour stitching for coherent passages
    4. Knowledge graph traversal — adds relationship map + extra chunks
    5. Returns assembled context and sources
    """
    # ── vector search ──
    embedding   = embed(question)
    meta_filter = detect_filter(question)

    query_kwargs = {"vector": embedding, "top_k": TOP_K, "include_metadata": True}
    if meta_filter:
        query_kwargs["filter"] = meta_filter

    result  = index.query(**query_kwargs)
    matches = result["matches"]

    if not matches:
        return "No relevant passages found.", []

    neighbour_texts = fetch_neighbours(matches)
    seen_ids        = {m["id"] for m in matches}

    # build vector-retrieved context blocks
    context_blocks = []
    sources        = []

    for m in matches:
        meta     = m["metadata"]
        citation = meta.get("citation") or meta.get("book") or meta.get("source", "Unknown")
        book     = meta.get("book") or meta.get("source", "Unknown")

        prev_text = neighbour_texts.get(meta.get("prev_chunk_id", ""), "")
        next_text = neighbour_texts.get(meta.get("next_chunk_id", ""), "")
        stitched  = "\n".join(filter(None, [prev_text, meta["text"], next_text]))

        context_blocks.append(f"[{citation}]\n{stitched}")
        if book not in sources:
            sources.append(book)

    # ── knowledge graph enrichment ──
    graph_section = ""
    entities = find_entities_in_question(question)

    if entities:
        rel_lines, extra_ids = graph_traverse(entities, max_hops=1, max_chunk_ids=3)

        # relationship map for Claude
        if rel_lines:
            unique_rels = list(dict.fromkeys(rel_lines))[:12]   # deduplicate, cap at 12
            graph_section = (
                "CONCEPT RELATIONSHIPS (from knowledge graph):\n"
                + "\n".join(f"  • {r}" for r in unique_rels)
                + "\n\n"
            )

        # extra chunks from graph traversal not already in vector results
        graph_chunks = fetch_chunks_by_ids(extra_ids, seen_ids)
        for gc in graph_chunks[:3]:                            # max 3 extra chunks
            if gc["text"].strip():
                context_blocks.append(f"[{gc['citation']} — via knowledge graph]\n{gc['text']}")
                if gc["book"] not in sources:
                    sources.append(gc["book"])

    context = graph_section + "\n\n---\n\n".join(context_blocks)
    return context, sources


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    import traceback

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        context, sources = retrieve(req.question)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    messages = []
    for turn in req.history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role":    "user",
        "content": f"Context from the sacred texts:\n\n{context}\n\nQuestion: {req.question}",
    })

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ChatResponse(
        answer=response.content[0].text,
        sources=sources,
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    stats = index.describe_index_stats()
    return {
        "status":        "ok",
        "vectors":       stats["total_vector_count"],
        "graph_nodes":   knowledge_graph.number_of_nodes(),
        "graph_edges":   knowledge_graph.number_of_edges(),
    }
