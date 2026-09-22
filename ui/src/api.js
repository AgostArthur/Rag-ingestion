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
  return {
    documents: Array.isArray(data.documents) ? data.documents : [],
    events: Array.isArray(data.events) ? data.events : [],
  };
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
    const raw = await res.text();
    let message = raw;
    try {
      const data = JSON.parse(raw);
      if (typeof data.detail === "string") message = data.detail;
    } catch {
      /* keep raw */
    }
    throw new Error(message || `POST /chat ${res.status}`);
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

const ROLE_LABELS = {
  contract: "Mandat",
  site_visit: "Visite du site",
  fieldwork: "Travaux de chantier",
  sampling: "Prélèvement",
  information_request: "Demande d'accès à l'information",
  information_response: "Réponse d'accès à l'information",
  lab_request: "Demande d'analyse",
  lab_receipt: "Réception au laboratoire",
  lab_analysis: "Analyse laboratoire",
  lab_certificate: "Certificat d'analyses",
  report: "Rapport",
};

const DOC_TYPE_LABELS = {
  ees_phase_1: "ÉES phase I",
  ees_phase_2: "ÉES phase II",
  rapport: "Rapport",
  contrat: "Contrat",
  facture: "Facture",
  note: "Note",
  guide: "Guide",
  autre: "Autre",
};

export function roleLabel(role) {
  if (!role) return "";
  const key = String(role).trim().toLowerCase();
  return ROLE_LABELS[key] || role;
}

export function docTypeLabel(docType) {
  if (!docType) return "";
  return DOC_TYPE_LABELS[docType] || docType;
}

export function eventHeading(event) {
  const type = docTypeLabel(event && event.doc_type);
  const title = (event && event.title) || "";
  if (type && title && title.toLowerCase() !== type.toLowerCase()) {
    return title;
  }
  return title || type || (event && event.firm) || "Rapport";
}

export function siteLine(event) {
  if (!event) return "";
  const lot = event.lot_cadastral ? `lot ${event.lot_cadastral}` : "";
  return [event.address, event.city, lot].filter(Boolean).join(" · ");
}

export function parseQualityLabel(quality) {
  if (quality === "ocr_heavy") return "OCR lourd";
  if (quality === "ocr_partial") return "OCR partiel";
  return "";
}

export function groupTimeline(documents, events) {
  const eventRows = Array.isArray(events) ? events : [];
  const docRows = Array.isArray(documents) ? documents : [];
  if (docRows.length === 0) {
    return eventRows.map((event) => ({ document: event, events: [event] }));
  }
  const byDoc = new Map();
  for (const event of eventRows) {
    const id = event && event.document_id;
    if (!id) continue;
    if (!byDoc.has(id)) byDoc.set(id, []);
    byDoc.get(id).push(event);
  }
  return docRows.map((document) => ({
    document,
    events: byDoc.get(document.document_id) || [],
  }));
}

/** Une entrée par jour : les events du site (une adresse) regroupés par `iso_date`. */
export function groupTimelineByDate(documents, events) {
  const eventRows = Array.isArray(events) ? events : [];
  const docRows = Array.isArray(documents) ? documents : [];
  const docsById = new Map(docRows.map((doc) => [doc.document_id, doc]));
  const merged = [];
  const seen = new Set();
  for (const event of eventRows) {
    if (!event || !event.iso_date) continue;
    const key = `${event.iso_date}|${event.role || ""}|${event.document_id || ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const doc = (event.document_id && docsById.get(event.document_id)) || {};
    merged.push({ ...doc, ...event });
  }
  const withEvents = new Set(merged.map((row) => row.document_id).filter(Boolean));
  for (const doc of docRows) {
    if (!doc || withEvents.has(doc.document_id) || !doc.report_date) continue;
    merged.push({ ...doc, iso_date: doc.report_date, role: "report" });
  }
  merged.sort((a, b) => String(b.iso_date).localeCompare(String(a.iso_date)));
  const groups = [];
  const index = new Map();
  for (const event of merged) {
    let group = index.get(event.iso_date);
    if (!group) {
      group = { iso_date: event.iso_date, events: [] };
      index.set(event.iso_date, group);
      groups.push(group);
    }
    group.events.push(event);
  }
  return groups;
}
