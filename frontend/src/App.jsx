import { useEffect, useRef, useState } from "react";
import { ask, getStats, removeDoc, upload } from "./api.js";

// Turn "[1]" markers in the answer into clickable chips that highlight the source.
function AnswerText({ text, onCite }) {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    return m ? (
      <button key={i} className="cite" onClick={() => onCite(Number(m[1]))}>
        {m[1]}
      </button>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

export default function App() {
  const [stats, setStats] = useState(null);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [active, setActive] = useState(null);
  const fileRef = useRef();

  const refresh = () => getStats().then(setStats).catch((e) => setError(e.message));
  useEffect(() => { refresh(); }, []);

  async function submit(e) {
    e?.preventDefault();
    if (!question.trim()) return;
    setLoading(true); setError(""); setActive(null);
    try { setResult(await ask(question, k, mode)); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function onUpload(e) {
    const files = [...e.target.files];
    if (!files.length) return;
    setError("");
    try { setStats(await upload(files)); }
    catch (err) { setError(err.message); }
    e.target.value = "";
  }

  async function onRemove(name) {
    try { setStats(await removeDoc(name)); } catch (err) { setError(err.message); }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>ragpipe</h1>
        <p className="muted">Hybrid-search RAG with cited answers</p>

        <h2>Knowledge base</h2>
        {stats ? (
          <>
            <p className="muted small">{stats.chunks} chunks · {stats.embedder} · {stats.llm === "ClaudeGenerator" ? "Claude" : "extractive (no API key)"}</p>
            <ul className="docs">
              {stats.documents.map((d) => (
                <li key={d.name}>
                  <span title={d.name}>{d.name}</span>
                  <em>{d.chunks}</em>
                  <button className="x" title="Remove (uploaded files only)" onClick={() => onRemove(d.name)}>×</button>
                </li>
              ))}
            </ul>
          </>
        ) : <p className="muted small">Connecting to backend…</p>}
        <input ref={fileRef} type="file" multiple accept=".pdf,.md,.txt" hidden onChange={onUpload} />
        <button className="secondary" onClick={() => fileRef.current.click()}>Upload PDF / MD / TXT</button>

        <h2>Retrieval</h2>
        <label>Mode
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="hybrid">Hybrid (BM25 + dense, RRF)</option>
            <option value="dense">Dense only</option>
            <option value="bm25">BM25 only</option>
          </select>
        </label>
        <label>Top-k: {k}
          <input type="range" min="1" max="10" value={k} onChange={(e) => setK(Number(e.target.value))} />
        </label>
      </aside>

      <main>
        <form onSubmit={submit} className="ask">
          <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about your documents…" autoFocus />
          <button disabled={loading}>{loading ? "Thinking…" : "Ask"}</button>
        </form>
        {error && <div className="error">{error}</div>}

        {result && (
          <>
            <section className="answer">
              <div className="badge-row">
                <span className={result.grounded ? "badge ok" : "badge warn"}>{result.grounded ? "Grounded" : "Not in documents"}</span>
              </div>
              <p><AnswerText text={result.answer} onCite={setActive} /></p>
            </section>
            <h3>Sources</h3>
            {result.citations.map((c) => (
              <article key={c.n} className={`source ${active === c.n ? "active" : ""}`}>
                <header>
                  <strong>[{c.n}] {c.source}</strong>
                  <span className="muted small">
                    score {c.score}
                    {c.dense_rank ? ` · dense #${c.dense_rank}` : ""}
                    {c.bm25_rank ? ` · bm25 #${c.bm25_rank}` : ""}
                  </span>
                </header>
                <p>{c.text}</p>
              </article>
            ))}
          </>
        )}
        {!result && !error && <p className="muted empty">Try: “What does Reciprocal Rank Fusion compute?”</p>}
      </main>
    </div>
  );
}
