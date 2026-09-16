/** Badge discret : durée totale, détail des étapes au clic. */

const STEP_LABELS = [
  ["llm", "Modèle"],
  ["search_knowledge", "Recherche RAG"],
  ["rag_retrieve", "Embed + Qdrant"],
  ["rag_format", "Formatage"],
  ["rag_catalog", "Catalog"],
];

export function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "";
  const s = Number(seconds);
  if (s < 0.001) return "<1 ms";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  const minutes = Math.floor(s / 60);
  const rest = s - minutes * 60;
  return `${minutes} min ${rest.toFixed(1)} s`;
}

export default function ChatTiming({ timing, open, onToggle }) {
  if (!timing || timing.total_seconds == null) return null;

  const steps = timing.steps || {};
  const details = STEP_LABELS.filter(([key]) => (steps[key] || 0) > 0).map(
    ([key, label]) => [label, steps[key]]
  );
  const extra = Object.entries(steps).filter(
    ([key]) => !STEP_LABELS.some(([known]) => known === key)
  );

  return (
    <div style={{ marginTop: 6 }}>
      <button
        type="button"
        onClick={onToggle}
        title={open ? "Masquer le détail" : "Voir le détail des étapes"}
        style={{
          border: "none",
          background: "none",
          padding: 0,
          fontSize: 10,
          color: "#94a3b8",
          cursor: "pointer",
          fontFamily: "inherit",
        }}
      >
        {formatDuration(timing.total_seconds)}
        {details.length > 0 ? (open ? " ▴" : " ▾") : null}
      </button>
      {open && details.length + extra.length > 0 && (
        <div
          style={{
            marginTop: 4,
            paddingLeft: 8,
            borderLeft: "1px solid #e2e8f0",
            fontSize: 10,
            color: "#94a3b8",
            lineHeight: 1.6,
          }}
        >
          {details.map(([label, seconds]) => (
            <div key={label}>
              {label} · {formatDuration(seconds)}
            </div>
          ))}
          {extra.map(([key, seconds]) => (
            <div key={key}>
              {key} · {formatDuration(seconds)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
