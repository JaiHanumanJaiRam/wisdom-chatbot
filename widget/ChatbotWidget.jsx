import { useState, useRef, useEffect } from "react";
import "./ChatbotWidget.css";

const API_URL = import.meta.env.VITE_WISDOM_API_URL || "https://your-railway-app.up.railway.app";

export default function ChatbotWidget() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Namaste. Ask me anything from the sacred texts — the Bhagavad Gita, Upanishads, and Ashtavakra Gita." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  async function send() {
    const question = input.trim();
    if (!question || loading) return;

    const history = messages
      .slice(1)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    setSources([]);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });

      if (!res.ok) throw new Error("API error");
      const data = await res.json();

      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
      setSources(data.sources || []);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <>
      <button
        className="cwb-toggle"
        onClick={() => setOpen((o) => !o)}
        aria-label="Open wisdom chatbot"
      >
        {open ? "✕" : "🕉"}
      </button>

      {open && (
        <div className="cwb-panel" role="dialog" aria-label="Wisdom Chatbot">
          <div className="cwb-header">
            <span>Wisdom from the Texts</span>
          </div>

          <div className="cwb-messages">
            {messages.map((m, i) => (
              <div key={i} className={`cwb-msg cwb-msg--${m.role}`}>
                {m.content}
              </div>
            ))}
            {loading && (
              <div className="cwb-msg cwb-msg--assistant cwb-msg--typing">
                <span /><span /><span />
              </div>
            )}
            {sources.length > 0 && (
              <div className="cwb-sources">
                Sources: {sources.join(", ")}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="cwb-input-row">
            <textarea
              className="cwb-input"
              rows={2}
              placeholder="Ask a question…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              disabled={loading}
            />
            <button className="cwb-send" onClick={send} disabled={loading || !input.trim()}>
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
