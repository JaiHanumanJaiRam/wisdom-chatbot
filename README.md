# Wisdom Chatbot + Instagram Automation

A RAG-powered chatbot grounded in 9 Hindu philosophical texts, with a daily Instagram post pipeline.

## Project Structure

```
wisdom-chatbot/
├── backend/          # FastAPI + ChromaDB + Claude (deploy to Railway)
├── widget/           # React component (drop into suneelkaw-portfolio)
└── instagram/        # Daily quote automation (run via cron)
```

---

## Setup: Backend

### 1. Install dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Create .env

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 3. Add your books

```bash
mkdir books
# Copy all 9 PDFs into backend/books/
```

### 4. Ingest the books (run once)

```bash
python ingest.py
```

This creates `chroma_db/` with all embeddings. Takes a few minutes for large PDFs.

### 5. Run locally

```bash
uvicorn main:app --reload
# API available at http://localhost:8000
# Health check: http://localhost:8000/health
```

---

## Deploy Backend to Railway

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Point to this `backend/` folder
3. Set environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
4. **Important:** You must run `python ingest.py` once on Railway or upload a pre-built `chroma_db/` volume
5. Railway will use the `Procfile` to start the server

Note: Upload the `chroma_db/` folder as a Railway volume after ingestion so the DB persists across deploys.

---

## Setup: React Widget

### 1. Copy files into your portfolio repo

```bash
cp widget/ChatbotWidget.jsx /path/to/suneelkaw-portfolio/src/components/
cp widget/ChatbotWidget.css /path/to/suneelkaw-portfolio/src/components/
```

### 2. Add environment variable to Vercel

In Vercel dashboard → Settings → Environment Variables:
```
VITE_WISDOM_API_URL = https://your-railway-app.up.railway.app
```

### 3. Add widget to App.jsx

```jsx
import ChatbotWidget from "./components/ChatbotWidget";

// Inside your App return:
<ChatbotWidget />
```

It renders as a floating button in the bottom-right corner.

---

## Setup: Instagram Automation

### 1. Prerequisites

- Instagram Business or Creator account
- Connected to a Meta Developer App
- Access token with `instagram_basic`, `instagram_content_publish` permissions

### 2. Install dependencies

```bash
cd instagram
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Create .env

```bash
cp .env.example .env
# Fill in all values
```

`IMAGE_HOST_URL` must be a publicly accessible URL where Railway can serve the generated image.
The simplest approach: upload to a Railway static volume or an S3 bucket, then set the URL.

### 4. Run manually

```bash
python daily_post.py
```

### 5. Schedule daily (Railway cron or system cron)

```cron
0 8 * * * cd /path/to/instagram && python daily_post.py
```

---

## API Reference

### POST /chat

```json
{
  "question": "What is the nature of the self?",
  "history": []
}
```

Response:
```json
{
  "answer": "According to the Mandukya Upanishad...",
  "sources": ["Mandukya_Upanishad1", "Gitapress_Gita_Roman"]
}
```

### GET /health

Returns chunk count and status.
