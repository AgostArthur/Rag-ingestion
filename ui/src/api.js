/** Client HTTP vers rag-chat (proxy Vite / nginx → API). */

function encodeId(id) {
  return encodeURIComponent(id);
}

export async function fetchSites() {
  const res = await fetch("/sites");
  if (!res.ok) {
    throw new Error(`GET /sites ${res.status}`);
  }
  const data = await res.json();
  return Array.isArray(data.sites) ? data.sites : [];
}

export async function fetchTimeline(siteId) {
  const res = await fetch(`/sites/${encodeId(siteId)}/timeline`);
  if (!res.ok) {
    throw new Error(`GET /timeline ${res.status}`);
  }
  const data = await res.json();
  return Array.isArray(data.events) ? data.events : [];
}

export async function postChat({ message, threadId, siteId, documentId }) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId || null,
      site_id: siteId || null,
      document_id: documentId || null,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `POST /chat ${res.status}`);
  }
  return res.json();
}

export function fileLabel(event) {
  const path = event && event.source_path;
  if (path) {
    const parts = String(path).split(/[/\\]/);
    return parts[parts.length - 1] || path;
  }
  return (
    (event && (event.title || event.project_id || event.firm)) ||
    "rapport"
  );
}
