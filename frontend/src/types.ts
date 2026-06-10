export interface HealthStatus {
  status: string;
  qdrant: string;
  mlflow: string;
  llm: string;
}

export interface NotesList {
  count: number;
  notes: string[];
}

export interface NoteDetail {
  note_id: string;
  content: string;
  n_chunks: number;
  reference_question: string;
  reference_answer: string;
  source: string;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
  question: string;
}
