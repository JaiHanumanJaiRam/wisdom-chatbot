"""Quick end-to-end test. Run from backend/: python test_chat.py"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from main import retrieve, SYSTEM_PROMPT
import os, anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

question = "What is the nature of the self according to the Upanishads?"
context, sources = retrieve(question)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=400,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"}],
)

print("SOURCES:", sources)
print("\nANSWER:", response.content[0].text)
