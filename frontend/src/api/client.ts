import type {
  HealthStatus,
  NoteDetail,
  NotesList,
  QueryResponse,
} from "../types";

// URLs relatives : proxy Vite en dev, même origine que FastAPI en prod.
const API_KEY = import.meta.env.VITE_API_KEY as string | undefined;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (API_KEY) headers["X-API-Key"] = API_KEY;
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    let detail = `Erreur ${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* corps non JSON */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthStatus>("/health"),
  listNotes: () => request<NotesList>("/notes"),
  getNote: (noteId: string) =>
    request<NoteDetail>(`/notes/${encodeURIComponent(noteId)}`),
  query: (question: string, noteId: string, topK: number) =>
    request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify({ question, note_id: noteId, top_k: topK }),
    }),
};
