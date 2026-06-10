import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import PatientDetail from "./components/PatientDetail";
import EmptyState from "./components/EmptyState";

export default function App() {
  const [notes, setNotes] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const loadNotes = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listNotes();
      setNotes(data.notes);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadNotes();
  }, [loadNotes]);

  return (
    <div className="app">
      <Header />
      <div className="layout">
        <Sidebar
          notes={notes}
          loading={loading}
          error={error}
          selected={selected}
          onSelect={setSelected}
          onRetry={loadNotes}
        />
        <main className="main">
          {selected ? (
            <PatientDetail key={selected} noteId={selected} />
          ) : (
            <EmptyState count={notes.length} />
          )}
        </main>
      </div>
    </div>
  );
}
