export default function EmptyState({ count }: { count: number }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <svg viewBox="0 0 24 24" width="40" height="40" aria-hidden>
          <path
            d="M9 3h6a1 1 0 0 1 1 1v1h3a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h3V4a1 1 0 0 1 1-1Z"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
          />
          <path
            d="M12 10v6M9 13h6"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
      </div>
      <h2>Sélectionnez un dossier patient</h2>
      <p>
        {count > 0
          ? `${count} dossiers cliniques sont disponibles. Choisissez-en un dans la liste pour consulter la note complète et interroger l'assistant IA.`
          : "Aucun dossier n'est encore indexé. Lancez une ingestion du dataset Asclepius pour commencer."}
      </p>
    </div>
  );
}
