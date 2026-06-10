import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HealthStatus } from "../types";

function StatusDot({ label, state }: { label: string; state?: string }) {
  const cls = state === "ok" ? "ok" : state ? "ko" : "unknown";
  return (
    <span className="status-item" title={`${label} : ${state ?? "inconnu"}`}>
      <span className={`status-dot ${cls}`} />
      {label}
    </span>
  );
}

export default function Header() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    const refresh = () => {
      api
        .health()
        .then((h) => {
          if (!cancelled) setHealth(h);
        })
        .catch(() => {
          if (!cancelled)
            setHealth({
              status: "down",
              qdrant: "error",
              mlflow: "error",
              llm: "error",
            });
        });
    };
    refresh();
    const timer = setInterval(refresh, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <header className="header">
      <div className="brand">
        <div className="brand-icon">
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden>
            <path
              d="M12 5v14M5 12h14"
              stroke="currentColor"
              strokeWidth="3.2"
              strokeLinecap="round"
            />
          </svg>
        </div>
        <div>
          <div className="brand-name">MedAssist</div>
          <div className="brand-sub">Gestion des dossiers médicaux</div>
        </div>
      </div>
      <div className="header-status">
        <StatusDot label="Base vectorielle" state={health?.qdrant} />
        <StatusDot label="Monitoring" state={health?.mlflow} />
        <StatusDot label="Assistant IA" state={health?.llm} />
      </div>
    </header>
  );
}
