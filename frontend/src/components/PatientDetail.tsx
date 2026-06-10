import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { NoteDetail } from "../types";
import { patientLabel } from "./Sidebar";
import QueryPanel from "./QueryPanel";

interface Section {
  heading: string | null;
  body: string;
}

/** Découpe la note en sections à partir des titres « Heading: » du texte. */
export function splitSections(content: string): Section[] {
  const lines = content.split(/\r?\n/);
  const sections: Section[] = [];
  let current: Section = { heading: null, body: "" };
  const headingRe = /^[A-Z][A-Za-z0-9 ()/&'-]{1,60}:\s*$/;
  for (const line of lines) {
    if (headingRe.test(line.trim())) {
      if (current.body.trim() || current.heading) sections.push(current);
      current = { heading: line.trim().replace(/:\s*$/, ""), body: "" };
    } else {
      current.body += line + "\n";
    }
  }
  if (current.body.trim() || current.heading) sections.push(current);
  return sections;
}

export default function PatientDetail({ noteId }: { noteId: string }) {
  const [note, setNote] = useState<NoteDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showReference, setShowReference] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setNote(null);
    setError(null);
    api
      .getNote(noteId)
      .then((data) => {
        if (!cancelled) setNote(data);
      })
      .catch((exc) => {
        if (!cancelled)
          setError(exc instanceof Error ? exc.message : String(exc));
      });
    return () => {
      cancelled = true;
    };
  }, [noteId]);

  if (error) {
    return (
      <div className="detail">
        <div className="card error-card">
          <strong>Impossible de charger le dossier.</strong>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!note) {
    return (
      <div className="detail">
        <div className="card detail-loading">
          <span className="spinner" /> Ouverture du dossier {noteId}…
        </div>
      </div>
    );
  }

  const sections = splitSections(note.content);

  return (
    <div className="detail">
      <div className="card record-head">
        <div className="record-identity">
          <span className="record-avatar">
            {noteId.slice(-2).toUpperCase()}
          </span>
          <div>
            <h1>{patientLabel(noteId)}</h1>
            <div className="record-tags">
              <span className="tag">{note.note_id}</span>
              <span className="tag">{note.n_chunks} segments indexés</span>
              {note.source && <span className="tag">{note.source}</span>}
            </div>
          </div>
        </div>
      </div>

      <div className="card record-body">
        <div className="card-title">
          <h2>Note clinique</h2>
        </div>
        {sections.map((section, i) => (
          <section key={i} className="record-section">
            {section.heading && <h3>{section.heading}</h3>}
            <p>{section.body.trim()}</p>
          </section>
        ))}
      </div>

      {(note.reference_question || note.reference_answer) && (
        <div className="card">
          <button
            className="collapse-toggle"
            onClick={() => setShowReference((v) => !v)}
            aria-expanded={showReference}
          >
            <h2>Question / réponse de référence (dataset)</h2>
            <span className={`chevron ${showReference ? "open" : ""}`}>▾</span>
          </button>
          {showReference && (
            <div className="reference-block">
              {note.reference_question && (
                <p>
                  <strong>Question :</strong> {note.reference_question}
                </p>
              )}
              {note.reference_answer && (
                <p>
                  <strong>Réponse attendue :</strong> {note.reference_answer}
                </p>
              )}
            </div>
          )}
        </div>
      )}

      <QueryPanel noteId={noteId} />
    </div>
  );
}
