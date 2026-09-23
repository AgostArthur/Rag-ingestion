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

export async function postChat({ message, threadId, siteId, documentId, context }) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      thread_id: threadId || null,
      site_id: siteId || null,
      document_id: documentId || null,
      context: Array.isArray(context) ? context : [],
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

const PHASE_RULES = [
  {
    id: "ees_phase_2",
    patterns: [
      /phase[\s_-]*(2|ii)(?!i)/i,
      /ees[\s_-]*(2|ii)(?!i)/i,
      /esa[\s_-]*(2|ii)(?!i)/i,
    ],
  },
  {
    id: "ees_phase_1",
    patterns: [
      /phase[\s_-]*(1|i)(?!i)/i,
      /ees[\s_-]*(1|i)(?!i)/i,
      /esa[\s_-]*(1|i)(?!i)/i,
    ],
  },
];

function foldAscii(text) {
  return String(text || "")
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLowerCase();
}

/** Phase I/II : `doc_type` catalog, sinon les mêmes motifs que le profil LangExtract (nom, puis titre). */
export function phaseIdForDocument(doc) {
  if (!doc) return "";
  const stored = String(doc.doc_type || "")
    .trim()
    .toLowerCase();
  if (stored === "ees_phase_1" || stored === "ees_phase_2") return stored;
  const name = fileLabel(doc);
  const title = doc.title || "";
  for (const source of [name, title]) {
    const folded = foldAscii(source);
    for (const rule of PHASE_RULES) {
      if (rule.patterns.some((pattern) => pattern.test(folded))) return rule.id;
    }
  }
  return "";
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

const DATE_ONLY_LABEL =
  /^\d{4}-\d{2}-\d{2}$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$|^\d{1,2}\s+\p{L}+\s+\d{4}$/u;

/** Texte d'événement utile en chronologie : pas la date, ni le rôle, ni le titre du PDF. */
export function eventNote(event) {
  if (!event) return "";
  const label = String(event.label || "").trim();
  if (!label) return "";
  const folded = label.toLowerCase();
  const role = roleLabel(event.role).trim().toLowerCase();
  if (folded === role || folded === "rapport") return "";
  if (folded === String(event.iso_date || "").toLowerCase()) return "";
  if (DATE_ONLY_LABEL.test(label)) return "";
  const heading = eventHeading(event).trim().toLowerCase();
  if (heading && folded === heading) return "";
  return label;
}

/** Client, fichiers et contaminants du site, une fois, dédupliqués. */
export function siteHeaderFacts(documents) {
  const docs = Array.isArray(documents) ? documents : [];
  const files = [];
  const seenFiles = new Set();
  const clients = [];
  const seenClients = new Set();
  const contaminants = [];
  const seenContaminants = new Set();
  for (const doc of docs) {
    if (!doc) continue;
    const name = fileLabel(doc);
    const key = doc.document_id || name;
    if (key && !seenFiles.has(key)) {
      seenFiles.add(key);
      files.push({
        document_id: doc.document_id || key,
        name,
        project_id: doc.project_id || "",
        firm: doc.firm || "",
        doc_type: docTypeLabel(doc.doc_type),
        quality: parseQualityLabel(doc.parse_quality),
      });
    }
    const client = String(doc.client || "").trim();
    const clientKey = client.toLowerCase();
    if (client && !seenClients.has(clientKey)) {
      seenClients.add(clientKey);
      clients.push(client);
    }
    const list = Array.isArray(doc.contaminants) ? doc.contaminants : [];
    for (const raw of list) {
      const item = String(raw || "").trim();
      const itemKey = item.toLowerCase();
      if (!item || seenContaminants.has(itemKey)) continue;
      seenContaminants.add(itemKey);
      contaminants.push(item);
    }
  }
  const phaseIds = new Set();
  for (const doc of docs) {
    const phase = phaseIdForDocument(doc);
    if (phase) phaseIds.add(phase);
  }
  const phases = ["ees_phase_1", "ees_phase_2"]
    .filter((id) => phaseIds.has(id))
    .map((id) => docTypeLabel(id));
  return { files, clients, contaminants, phases };
}

/** Fichiers du site affichés dans la barre de contexte (un chip par PDF). */
export function contextFilesFromDocuments(documents) {
  return siteHeaderFacts(documents).files.map((file) => ({
    document_id: file.document_id,
    name: file.name,
  }));
}

/** Libellé d'une journée de chronologie : date, rôles, note utile. */
export function sectionContextLabel(events, isoDate) {
  const roles = [];
  const seenRoles = new Set();
  const notes = [];
  const seenNotes = new Set();
  for (const event of events || []) {
    const role = roleLabel(event && event.role);
    if (role && !seenRoles.has(role)) {
      seenRoles.add(role);
      roles.push(role);
    }
    const note = eventNote(event);
    if (note && !seenNotes.has(note)) {
      seenNotes.add(note);
      notes.push(note);
    }
  }
  const meta = [...roles, ...notes].join(" · ");
  const date = isoDate || "sans date";
  return meta ? `${date} ${meta}` : date;
}

/** Une ligne par fichier, pages uniques triées. Le nom n'est pas répété à chaque page. */
export function groupCitations(citations) {
  const groups = [];
  const index = new Map();
  for (const citation of Array.isArray(citations) ? citations : []) {
    if (!citation) continue;
    const name = String(citation.source || citation.document_id || "source");
    const key = citation.document_id || name;
    let group = index.get(key);
    if (!group) {
      group = { key, name, pages: [] };
      index.set(key, group);
      groups.push(group);
    }
    const page = citation.page;
    if (page != null && page !== "" && !group.pages.includes(page)) {
      group.pages.push(page);
    }
  }
  for (const group of groups) {
    group.pages.sort((a, b) => Number(a) - Number(b));
  }
  return groups;
}

/** `document_id` envoyé au chat : la section si son PDF est encore chargé, sinon l'unique fichier. */
export function contextDocumentId(files, section) {
  const list = Array.isArray(files) ? files : [];
  if (
    section &&
    section.document_id &&
    list.some((file) => file.document_id === section.document_id)
  ) {
    return section.document_id;
  }
  if (list.length === 1) return list[0].document_id || null;
  return null;
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
