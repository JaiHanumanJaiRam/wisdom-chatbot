import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv(Path(__file__).parent / ".env", override=True)

INDEX_NAME = "wisdom-books"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6

embedder = SentenceTransformer(EMBED_MODEL)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://suneelkaw.com", "https://www.suneelkaw.com", "http://localhost:5173", "http://localhost:5200"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

SYSTEM_PROMPT = """You are a wisdom guide specialising in Vedantic and Hindu philosophical texts.
You answer ONLY using the context passages provided below from the sacred books.
If the answer cannot be found in the provided context, respond with:
"I cannot find an answer to that in the texts I have been given."

Do not use any outside knowledge. Do not speculate beyond the texts.
Cite the source book for key points when possible."""


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


def retrieve(question: str) -> tuple[str, list[str]]:
    embedding = embedder.encode([question])[0].tolist()
    results = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
    )
    matches = results["matches"]
    sources = list({m["metadata"]["source"] for m in matches})
    context = "\n\n---\n\n".join(
        f"[{m['metadata']['source']}]\n{m['metadata']['text']}"
        for m in matches
    )
    return context, sources


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    context, sources = retrieve(req.question)

    messages = []
    for turn in req.history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from the sacred texts:\n\n{context}\n\nQuestion: {req.question}",
    })

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    return ChatResponse(
        answer=response.content[0].text,
        sources=sources,
    )


@app.get("/health")
async def health():
    stats = index.describe_index_stats()
    return {"status": "ok", "vectors": stats["total_vector_count"]}
