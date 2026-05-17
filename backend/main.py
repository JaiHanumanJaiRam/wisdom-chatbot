import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
import anthropic

load_dotenv(Path(__file__).parent / ".env", override=True)

INDEX_NAME  = "wisdom-books"
EMBED_MODEL = "jina-embeddings-v2-base-en"
TOP_K       = 6

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
Cite the source book and citation reference for key points when possible."""


# ── Request / response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: list = []


class ChatResponse(BaseModel):
    answer: str
    sources: list


# ── Tradition / book detection for metadata pre-filtering ─────────────────────

# Keyword sets that signal the user is asking about a specific scope.
# The filter is passed to Pinecone BEFORE vector similarity search,
# so we only rank chunks that match the scope.

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
    "Bhagavad Gita":     ["bhagavad gita", "gita", "krishna said", "arjuna"],
    "Ashtavakra Gita":   ["ashtavakra"],
    "Mandukya Upanishad":["mandukya"],
    "Chandogya Upanishad":["chandogya"],
    "Brihadaranyaka Upanishad": ["brihadaranyaka"],
    "The Principal Upanishads": ["radhakrishnan"],
}


def detect_filter(question):
    """
    Inspect the question for tradition or book keywords.
    Returns a Pinecone metadata filter dict, or None if no scope is detected.
    """
    q = question.lower()

    # Check for specific book first (more precise)
    for book, keywords in _BOOK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return {"book": {"$eq": book}}

    # Fall back to tradition-level filter
    for tradition, keywords in _TRADITION_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return {"tradition": {"$eq": tradition}}

    return None


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed(text):
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization":  f"Bearer {JINA_API_KEY}",
            "Content-Type":   "application/json",
        },
        json={"model": EMBED_MODEL, "input": [text]},
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


# ── Neighbour stitching ───────────────────────────────────────────────────────

def fetch_neighbours(matches):
    """
    For each retrieved chunk, fetch its prev and next chunks from Pinecone
    so Claude receives a complete passage rather than a mid-sentence fragment.
    Returns a dict of {chunk_id: text}.
    """
    neighbour_ids = []
    for m in matches:
        meta = m["metadata"]
        prev_id = meta.get("prev_chunk_id", "")
        next_id = meta.get("next_chunk_id", "")
        if prev_id:
            neighbour_ids.append(prev_id)
        if next_id:
            neighbour_ids.append(next_id)

    if not neighbour_ids:
        return {}

    try:
        fetched = index.fetch(ids=neighbour_ids)
        return {
            vid: v.metadata.get("text", "")
            for vid, v in fetched.vectors.items()
        }
    except Exception:
        return {}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(question):
    """
    1. Detect tradition/book scope → Pinecone metadata pre-filter
    2. Embed question → vector similarity search within that scope
    3. Fetch neighbouring chunks for context stitching
    4. Return assembled context string and source citations
    """
    embedding    = embed(question)
    meta_filter  = detect_filter(question)

    query_kwargs = {
        "vector":          embedding,
        "top_k":           TOP_K,
        "include_metadata": True,
    }
    if meta_filter:
        query_kwargs["filter"] = meta_filter

    result  = index.query(**query_kwargs)
    matches = result["matches"]

    if not matches:
        return "No relevant passages found.", []

    # fetch neighbours for context stitching
    neighbour_texts = fetch_neighbours(matches)

    context_blocks = []
    sources        = []

    for m in matches:
        meta      = m["metadata"]
        chunk_id  = m["id"]

        # build citation: prefer stored citation field, fall back to book name
        citation = meta.get("citation") or meta.get("book") or meta.get("source", "Unknown")
        book     = meta.get("book") or meta.get("source", "Unknown")

        # stitch prev + current + next for a coherent passage
        prev_id   = meta.get("prev_chunk_id", "")
        next_id   = meta.get("next_chunk_id", "")
        prev_text = neighbour_texts.get(prev_id, "")
        next_text = neighbour_texts.get(next_id, "")

        stitched = "\n".join(filter(None, [prev_text, meta["text"], next_text]))

        context_blocks.append(f"[{citation}]\n{stitched}")

        # deduplicate sources list
        if book not in sources:
            sources.append(book)

    context = "\n\n---\n\n".join(context_blocks)
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
        "status":  "ok",
        "vectors": stats["total_vector_count"],
    }
