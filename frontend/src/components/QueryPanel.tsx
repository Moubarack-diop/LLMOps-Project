import { FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "../api/client";
import type { QueryResponse } from "../types";

const SUGGESTIONS = [
  "Quel était le motif d'admission de ce patient ?",
  "Quels traitements ont été administrés ?",
  "Quelle a été l'issue clinique et le suivi prévu ?",
];

export default function QueryPanel({ noteId }: { noteId: string }) {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);

  const ask = async (text: string) => {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.query(text.trim(), noteId, topK);
      setResult(response);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoading(false);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(question);
  };

  return (
    <div className="card query-panel">
      <div className="card-title">
        <h2>Interroger ce dossier</h2>
        <span className="ai-badge">Assistant IA</span>
      </div>
      <p className="query-hint">
        L'assistant répond uniquement à partir du contenu de ce dossier et cite
        ses sources. Il signale explicitement les informations absentes.
      </p>
      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            type="button"
            className="suggestion-chip"
            disabled={loading}
            onClick={() => {
              setQuestion(s);
              void ask(s);
            }}
          >
            {s}
          </button>
        ))}
      </div>
      <form onSubmit={onSubmit} className="query-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Posez votre question sur ce patient…"
          rows={3}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              void ask(question);
            }
          }}
        />
        <div className="query-actions">
          <label className="topk-label">
            Extraits analysés
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              disabled={loading}
            >
              {[3, 5, 8, 10, 15, 20].map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || !question.trim()}
          >
            {loading ? (
              <>
                <span className="spinner light" /> Analyse en cours…
              </>
            ) : (
              "Envoyer la question"
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="answer-block error-card">
          <strong>Erreur :</strong> {error}
        </div>
      )}

      {result && (
        <div className="answer-block">
          <div className="answer-question">{result.question}</div>
          <div className="answer-markdown">
            <ReactMarkdown>{result.answer}</ReactMarkdown>
          </div>
          {result.sources.length > 0 && (
            <div className="answer-sources">
              <span>Sources citées :</span>
              {result.sources.map((s) => (
                <span key={s} className="source-badge">
                  {s}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
