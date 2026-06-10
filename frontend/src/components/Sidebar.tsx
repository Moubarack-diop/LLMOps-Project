import { useMemo, useState } from "react";

interface SidebarProps {
  notes: string[];
  loading: boolean;
  error: string | null;
  selected: string | null;
  onSelect: (noteId: string) => void;
  onRetry: () => void;
}

export function patientLabel(noteId: string): string {
  const num = noteId.replace(/^note_0*/, "");
  return `Patient n° ${num || "0"}`;
}

export default function Sidebar({
  notes,
  loading,
  error,
  selected,
  onSelect,
  onRetry,
}: SidebarProps) {
  const [search, setSearch] = useState("");

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    if (!term) return notes;
    return notes.filter(
      (n) =>
        n.toLowerCase().includes(term) ||
        patientLabel(n).toLowerCase().includes(term),
    );
  }, [notes, search]);

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <h2>Dossiers patients</h2>
        <span className="count-badge">{notes.length}</span>
      </div>
      <div className="search-box">
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden>
          <circle
            cx="11"
            cy="11"
            r="7"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
          <path
            d="m20 20-3.5-3.5"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
          />
        </svg>
        <input
          type="search"
          placeholder="Rechercher un dossier…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          aria-label="Rechercher un dossier"
        />
      </div>
      <div className="patient-list">
        {loading && (
          <div className="sidebar-info">
            <span className="spinner" /> Chargement des dossiers…
          </div>
        )}
        {error && (
          <div className="sidebar-info error">
            <p>{error}</p>
            <button className="btn btn-ghost" onClick={onRetry}>
              Réessayer
            </button>
          </div>
        )}
        {!loading && !error && filtered.length === 0 && (
          <div className="sidebar-info">
            {notes.length === 0
              ? "Aucun dossier. Lancez une ingestion via l'API."
              : "Aucun résultat pour cette recherche."}
          </div>
        )}
        {!loading &&
          !error &&
          filtered.map((noteId) => (
            <button
              key={noteId}
              className={`patient-item ${selected === noteId ? "active" : ""}`}
              onClick={() => onSelect(noteId)}
            >
              <span className="patient-avatar">
                {noteId.slice(-2).toUpperCase()}
              </span>
              <span className="patient-meta">
                <span className="patient-name">{patientLabel(noteId)}</span>
                <span className="patient-id">{noteId}</span>
              </span>
            </button>
          ))}
      </div>
    </aside>
  );
}
