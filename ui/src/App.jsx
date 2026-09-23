// Client shell: map + chronologie + chat. Data comes from rag-chat catalog, not mock SITES.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleMarker, GeoJSON, MapContainer, TileLayer, useMap } from "react-leaflet";
import L from "leaflet";
import { ChevronDown, ChevronUp, Clock, Send, ShieldCheck } from "lucide-react";
import {
  contextDocumentId,
  contextFilesFromDocuments,
  docTypeLabel,
  eventNote,
  fetchSites,
  fetchTimeline,
  groupCitations,
  groupTimelineByDate,
  postChat,
  roleLabel,
  sectionContextLabel,
  siteHeaderFacts,
} from "./api";
import ChatMarkdown from "./ChatMarkdown";
import ChatTiming from "./ChatTiming";

// Sud du Québec, Montréal dans le cadre (zoom assez large pour la plaine du Saint-Laurent).
const MAP_HOME = { lat: 45.55, lng: -73.4 };
const MAP_HOME_ZOOM = 8;

function SiteLayer({ site, selected, onSelect }) {
  const pos = pinPosition(site);
  const feature =
    site.lot_geometry && site.lot_geometry.type
      ? {
          type: "Feature",
          properties: { site_id: site.site_id },
          geometry: site.lot_geometry,
        }
      : null;
  if (!pos && !feature) return null;
  return (
    <>
      {feature ? (
        <GeoJSON
          key={`${site.site_id}-lot-${selected ? "on" : "off"}`}
          data={feature}
          style={{
            color: selected ? "#3DA821" : "#475569",
            weight: selected ? 3 : 2,
            fillColor: selected ? "#50C32A" : "#94a3b8",
            fillOpacity: selected ? 0.4 : 0.22,
          }}
          eventHandlers={{ click: () => onSelect(site.site_id) }}
        />
      ) : null}
      {pos ? (
        <CircleMarker
          center={[pos.lat, pos.lng]}
          radius={selected ? 8 : 6}
          pathOptions={{
            color: "#fff",
            weight: 2,
            fillColor: selected ? "#50C32A" : "#64748b",
            fillOpacity: 1,
          }}
          eventHandlers={{ click: () => onSelect(site.site_id) }}
        />
      ) : null}
    </>
  );
}

function FocusSite({ geometry, pos }) {
  const map = useMap();
  useEffect(() => {
    if (geometry && geometry.type) {
      const layer = L.geoJSON(geometry);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [48, 48], maxZoom: 18 });
        return;
      }
    }
    if (pos && Number.isFinite(pos.lat) && Number.isFinite(pos.lng)) {
      map.setView([pos.lat, pos.lng], Math.max(map.getZoom() || 8, 16));
    }
  }, [map, geometry, pos]);
  return null;
}

function pinPosition(site) {
  if (site == null || site.lat == null || site.lon == null) return null;
  const lat = Number(site.lat);
  const lng = Number(site.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
  return { lat, lng };
}

function App() {
  const [sites, setSites] = useState([]);
  const [sitesError, setSitesError] = useState("");
  const [activeId, setActiveId] = useState(null);
  const [events, setEvents] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [contextFiles, setContextFiles] = useState([]);
  const [contextSection, setContextSection] = useState(null);
  const [timelineOpen, setTimelineOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [chatError, setChatError] = useState("");
  const [openTimingIdx, setOpenTimingIdx] = useState(null);
  const chatEndRef = useRef(null);
  const pinContextRef = useRef(null);

  const activeSite = useMemo(
    () => sites.find((s) => s.site_id === activeId) || null,
    [sites, activeId]
  );
  const activePos = pinPosition(activeSite);

  useEffect(() => {
    let cancelled = false;
    fetchSites()
      .then((rows) => {
        if (cancelled) return;
        setSites(rows);
        setSitesError("");
        setActiveId((current) =>
          current && rows.some((s) => s.site_id === current) ? current : null
        );
      })
      .catch((err) => {
        if (!cancelled) setSitesError(err.message || "Catalog indisponible");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!activeId) {
      setEvents([]);
      setDocuments([]);
      setContextFiles([]);
      setContextSection(null);
      return;
    }
    let cancelled = false;
    const fromPin = pinContextRef.current === activeId;
    fetchTimeline(activeId)
      .then((payload) => {
        if (cancelled) return;
        setDocuments(payload.documents);
        setEvents(payload.events);
        if (fromPin) {
          pinContextRef.current = null;
          setContextFiles(contextFilesFromDocuments(payload.documents));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setDocuments([]);
          setEvents([]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, busy]);

  const selectSite = useCallback(
    (siteId) => {
      setActiveId(siteId);
      setContextSection(null);
      if (siteId === activeId) {
        setContextFiles(contextFilesFromDocuments(documents));
        return;
      }
      pinContextRef.current = siteId;
      setContextFiles([]);
    },
    [activeId, documents]
  );

  const selectSection = useCallback((group) => {
    const dated = (group && group.events) || [];
    const primary = dated.find((event) => event.document_id) || dated[0];
    if (!primary || !primary.document_id) return;
    setContextSection({
      document_id: primary.document_id,
      iso_date: group.iso_date || null,
      label: sectionContextLabel(dated, group.iso_date),
    });
  }, []);

  const send = useCallback(async () => {
    const text = draft.trim();
    if (!text || busy) return;
    setDraft("");
    setChatError("");
    setBusy(true);
    const userMsg = { role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    const context = [
      ...contextFiles.map((file) => `@${file.name}`),
      contextSection ? contextSection.label : null,
    ].filter(Boolean);
    try {
      const data = await postChat({
        message: text,
        threadId,
        siteId: activeId,
        documentId: contextDocumentId(contextFiles, contextSection),
        context,
      });
      setThreadId(data.thread_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.reply || "",
          citations: data.citations || [],
          timing: data.timing || null,
        },
      ]);
      setOpenTimingIdx(null);
      const nextSite =
        data.focus && data.focus.site_ids && data.focus.site_ids[0];
      if (nextSite && nextSite !== activeId) {
        setActiveId(nextSite);
        setContextFiles([]);
        setContextSection(null);
      }
    } catch (err) {
      setChatError(err.message || "Chat indisponible");
    } finally {
      setBusy(false);
    }
  }, [draft, busy, threadId, activeId, contextFiles, contextSection]);

  const timelineItems = useMemo(
    () => groupTimelineByDate(documents, events),
    [documents, events]
  );
  const siteFacts = useMemo(() => siteHeaderFacts(documents), [documents]);
  const severalFiles = siteFacts.files.length > 1;
  const severalTypes =
    new Set(siteFacts.files.map((file) => file.doc_type).filter(Boolean)).size > 1;
  const nDocs =
    (activeSite && activeSite.document_ids
      ? activeSite.document_ids.length
      : documents.length) || 0;

  const mapInner = (
    <>
      <MapContainer
        center={[MAP_HOME.lat, MAP_HOME.lng]}
        zoom={MAP_HOME_ZOOM}
        style={{ width: "100%", height: "100%" }}
        scrollWheelZoom
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <FocusSite
          geometry={activeSite && activeSite.lot_geometry}
          pos={activePos}
        />
        {sites.map((s) => (
          <SiteLayer
            key={`${s.site_id}-${s.lot_geometry ? "poly" : "pt"}-${s.lat}-${s.lon}`}
            site={s}
            selected={activeId === s.site_id}
            onSelect={selectSite}
          />
        ))}
      </MapContainer>

      <div
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          zIndex: 1100,
          width: 340,
          maxWidth: "calc(100% - 32px)",
          maxHeight: "calc(100% - 32px)",
          display: "flex",
          flexDirection: "column",
          background: "rgba(255,255,255,0.95)",
          backdropFilter: "blur(8px)",
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "12px 16px",
            flexShrink: 0,
            maxHeight: timelineOpen ? "46%" : "70%",
            overflowY: "auto",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 8,
              marginBottom: 4,
            }}
          >
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: "#94a3b8",
                textTransform: "uppercase",
              }}
            >
              Site sélectionné
            </div>
            <button
              type="button"
              onClick={() => setTimelineOpen((open) => !open)}
              aria-expanded={timelineOpen}
              style={{
                border: "none",
                background: "none",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontSize: 10,
                fontWeight: 700,
                color: "#64748b",
                textTransform: "uppercase",
                letterSpacing: 0.4,
                padding: 0,
              }}
            >
              <Clock size={12} color="#64748b" />
              Chronologie
              {timelineOpen ? (
                <ChevronUp size={14} color="#64748b" />
              ) : (
                <ChevronDown size={14} color="#64748b" />
              )}
            </button>
          </div>
          {activeSite ? (
            <>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#3DA821" }}>
                {activeSite.lot_cadastral
                  ? `Lot ${activeSite.lot_cadastral}`
                  : activeSite.address || activeSite.site_id}
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                {[activeSite.address, activeSite.city]
                  .filter(Boolean)
                  .join(" · ") || activeSite.site_id}
              </div>
              {siteFacts.phases.length > 0 && (
                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    color: "#3DA821",
                    marginBottom: 8,
                  }}
                >
                  {siteFacts.phases.join(" · ")}
                </div>
              )}
              <div
                style={{
                  display: "inline-block",
                  fontSize: 10,
                  fontWeight: 700,
                  padding: "2px 10px",
                  borderRadius: 999,
                  background: "#50C32A22",
                  color: "#3DA821",
                  border: "1px solid #50C32A44",
                }}
              >
                ● {nDocs} rapport{nDocs === 1 ? "" : "s"}
              </div>
              {siteFacts.clients.length > 0 && (
                <div style={{ marginTop: 8, fontSize: 11, color: "#64748b" }}>
                  Client : {siteFacts.clients.join(" · ")}
                </div>
              )}
              {siteFacts.files.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div
                    style={{
                      fontSize: 10,
                      fontWeight: 700,
                      color: "#94a3b8",
                      textTransform: "uppercase",
                      marginBottom: 2,
                    }}
                  >
                    {siteFacts.files.length > 1 ? "Fichiers" : "Fichier"}
                  </div>
                  {siteFacts.files.map((file) => (
                    <div
                      key={file.document_id}
                      style={{ fontSize: 11, color: "#64748b", lineHeight: 1.45 }}
                    >
                      {[
                        file.name,
                        file.project_id && `n° ${file.project_id}`,
                        file.firm,
                        file.quality,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </div>
                  ))}
                </div>
              )}
              {siteFacts.contaminants.length > 0 && (
                <div style={{ marginTop: 8, fontSize: 11, color: "#64748b" }}>
                  Contaminants : {siteFacts.contaminants.join(", ")}
                </div>
              )}
              {activeSite.lot_cadastral && !activeSite.lot_geometry && (
                <div style={{ marginTop: 8, fontSize: 10, color: "#b45309" }}>
                  Polygone de lot absent — ré-ingérez le PDF (cadastre QC).
                </div>
              )}
              {activeSite.geocode_status && activeSite.geocode_status !== "ok" && (
                <div style={{ marginTop: 8, fontSize: 10, color: "#b45309" }}>
                  Pas de coordonnées ({activeSite.geocode_status})
                </div>
              )}
            </>
          ) : (
            <div style={{ fontSize: 12, color: "#64748b" }}>
              {sitesError ||
                (sites.length === 0
                  ? "Aucun site dans le catalog. Ingérez des PDF."
                  : "Sélectionnez une épingle.")}
            </div>
          )}
        </div>
        {timelineOpen && (
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              padding: 16,
              borderTop: "1px solid #f1f5f9",
            }}
          >
            {timelineItems.length === 0 && (
              <div style={{ fontSize: 12, color: "#94a3b8" }}>
                {activeId
                  ? "Aucun rapport dans le catalog pour ce site."
                  : "Sélectionnez un site sur la carte."}
              </div>
            )}
            {timelineItems.map((group, i) => {
              const dated = group.events || [];
              const primary =
                dated.find((ev) => ev.role === "report") || dated[0] || {};
              const selected =
                contextSection &&
                contextSection.iso_date === group.iso_date &&
                dated.some(
                  (ev) => ev.document_id === contextSection.document_id
                );
              const typeLabel = docTypeLabel(primary.doc_type);
              return (
                <button
                  key={group.iso_date || `date-${i}`}
                  type="button"
                  onClick={() => selectSection(group)}
                  style={{
                    display: "block",
                    width: "100%",
                    textAlign: "left",
                    background: selected ? "#E8F7DD" : "transparent",
                    border: "none",
                    paddingLeft: 20,
                    paddingBottom: 16,
                    borderLeft: selected
                      ? "2px solid #50C32A"
                      : "2px solid #e2e8f0",
                    position: "relative",
                    cursor: primary.document_id ? "pointer" : "default",
                  }}
                >
                  <div
                    style={{
                      position: "absolute",
                      left: -9,
                      top: 2,
                      width: 16,
                      height: 16,
                      borderRadius: "50%",
                      background: "white",
                      border: selected
                        ? "3px solid #50C32A"
                        : "3px solid #cbd5e1",
                    }}
                  />
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      marginBottom: 6,
                      gap: 8,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        fontFamily: "monospace",
                        color: "#1e293b",
                      }}
                    >
                      {group.iso_date || "sans date"}
                    </span>
                    {severalTypes && typeLabel ? (
                      <span
                        style={{
                          fontSize: 9,
                          fontWeight: 700,
                          padding: "1px 6px",
                          background: "#E8F7DD",
                          color: "#3DA821",
                          borderRadius: 4,
                        }}
                      >
                        {typeLabel}
                      </span>
                    ) : null}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#334155",
                      lineHeight: 1.55,
                    }}
                  >
                    {dated.map((ev) => {
                      const note = eventNote(ev);
                      return (
                        <div
                          key={`${ev.iso_date}-${ev.role}-${ev.document_id || ""}`}
                        >
                          {roleLabel(ev.role)}
                          {severalFiles && ev.project_id
                            ? ` · n° ${ev.project_id}`
                            : ""}
                          {note ? ` — ${note}` : ""}
                        </div>
                      );
                    })}
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </>
  );

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        fontFamily: "sans-serif",
        background: "#f1f5f9",
      }}
    >
      <nav
        style={{
          height: 56,
          background: "#0f172a",
          color: "white",
          display: "flex",
          alignItems: "center",
          padding: "0 24px",
          justifyContent: "space-between",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <ShieldCheck color="#7AD455" size={22} />
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>
              Enviro-Expert · Portefeuille Environnemental
            </div>
            <div style={{ fontSize: 10, color: "#94a3b8" }}>
              Catalog RAG · carte, chronologie, chat
            </div>
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#7AD455" }}>
          ● {sites.length} site{sites.length === 1 ? "" : "s"}
        </div>
      </nav>

      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "row",
          padding: 12,
          gap: 12,
          overflow: "hidden",
        }}
      >
          <div
            style={{
              flex: "65 1 0",
              minWidth: 0,
              borderRadius: 16,
              overflow: "hidden",
              border: "1px solid #e2e8f0",
              boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
              position: "relative",
              background: "#e2e8f0",
            }}
          >
            {mapInner}
          </div>

        <div
          style={{
            flex: "35 1 0",
            minWidth: 0,
            background: "white",
            borderRadius: 16,
            border: "1px solid #e2e8f0",
            boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
          }}
        >
          <div
            style={{
              padding: "10px 16px",
              borderBottom: "1px solid #f1f5f9",
              background: "#f8fafc",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: "#50C32A",
              }}
            />
            <span
              style={{
                fontSize: 11,
                fontWeight: 700,
                color: "#475569",
                textTransform: "uppercase",
                letterSpacing: 1,
              }}
            >
              Enviro-Expert Assistant
            </span>
            <span
              style={{ marginLeft: "auto", fontSize: 10, color: "#94a3b8" }}
            >
              {activeSite
                ? activeSite.address || activeSite.site_id
                : "Aucun site"}
            </span>
          </div>
          <div style={{ flex: 1, padding: 16, overflowY: "auto" }}>
            {messages.length === 0 && (
              <div style={{ display: "flex", gap: 12 }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    background: "#2F7A19",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 11,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  AI
                </div>
                <div
                  style={{
                    background: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    borderRadius: 16,
                    padding: "10px 14px",
                    maxWidth: "80%",
                  }}
                >
                  <p style={{ fontSize: 13, color: "#334155", margin: 0 }}>
                    Posez une question sur les rapports ingérés. Une épingle
                    charge le fichier dans la barre de contexte
                    {activeSite ? (
                      <>
                        {" "}
                        (
                        <strong style={{ color: "#3DA821" }}>
                          {activeSite.address || activeSite.site_id}
                        </strong>
                        )
                      </>
                    ) : null}
                    . Une date de la chronologie ajoute cette journée. Chaque
                    croix retire un élément, sans toucher au texte.
                  </p>
                </div>
              </div>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  gap: 12,
                  marginBottom: 12,
                  justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                {m.role === "assistant" && (
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: 8,
                      background: "#2F7A19",
                      color: "white",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 11,
                      fontWeight: 700,
                      flexShrink: 0,
                    }}
                  >
                    AI
                  </div>
                )}
                <div
                  style={{
                    background: m.role === "user" ? "#E8F7DD" : "#ffffff",
                    border:
                      m.role === "user"
                        ? "1px solid #B5E294"
                        : "1px solid #e8eef4",
                    borderRadius: m.role === "user" ? "16px 16px 4px 16px" : "4px 16px 16px 16px",
                    boxShadow:
                      m.role === "user"
                        ? "none"
                        : "0 1px 2px rgba(15, 23, 42, 0.04)",
                    padding: "12px 14px",
                    maxWidth: m.role === "user" ? "78%" : "min(720px, 88%)",
                  }}
                >
                  {m.role === "assistant" ? (
                    <ChatMarkdown text={m.text} />
                  ) : (
                    <p
                      style={{
                        fontSize: 13.5,
                        lineHeight: 1.55,
                        color: "#1e293b",
                        margin: 0,
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {m.text}
                    </p>
                  )}
                  {m.citations && m.citations.length > 0 && (
                    <div className="chat-sources">
                      {groupCitations(m.citations).map((group) => (
                        <div key={group.key} className="chat-source">
                          <span className="chat-source-name" title={group.name}>
                            {group.name}
                          </span>
                          {group.pages.length > 0 && (
                            <span className="chat-source-pages">
                              {group.pages.map((page) => (
                                <span key={page}>p.{page}</span>
                              ))}
                            </span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                  {m.role === "assistant" && m.timing && (
                    <ChatTiming
                      timing={m.timing}
                      open={openTimingIdx === i}
                      onToggle={() =>
                        setOpenTimingIdx((current) =>
                          current === i ? null : i
                        )
                      }
                    />
                  )}
                </div>
              </div>
            ))}
            {busy && (
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    background: "#2F7A19",
                    color: "white",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 11,
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  AI
                </div>
                <div
                  style={{
                    background: "#ffffff",
                    border: "1px solid #e8eef4",
                    borderRadius: "4px 16px 16px 16px",
                    padding: "12px 14px",
                  }}
                  aria-label="Recherche"
                >
                  <div className="chat-typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
            {chatError && (
              <div style={{ fontSize: 12, color: "#b91c1c" }}>{chatError}</div>
            )}
            <div ref={chatEndRef} />
          </div>
          <div
            style={{
              padding: "12px 16px",
              borderTop: "1px solid #f1f5f9",
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {(contextFiles.length > 0 || contextSection) && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {contextFiles.map((file) => (
                  <span
                    key={file.document_id}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      background: "#E8F7DD",
                      color: "#3DA821",
                      padding: "4px 8px 4px 10px",
                      borderRadius: 999,
                      border: "1px solid #B5E294",
                      maxWidth: "100%",
                    }}
                  >
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                      @{file.name}
                    </span>
                    <button
                      type="button"
                      aria-label={`Retirer ${file.name}`}
                      onClick={() =>
                        setContextFiles((current) =>
                          current.filter(
                            (item) => item.document_id !== file.document_id
                          )
                        )
                      }
                      style={{
                        border: "none",
                        background: "none",
                        color: "#3DA821",
                        cursor: "pointer",
                        fontSize: 14,
                        lineHeight: 1,
                        padding: 0,
                      }}
                    >
                      ×
                    </button>
                  </span>
                ))}
                {contextSection && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      fontSize: 11,
                      background: "#f8fafc",
                      color: "#334155",
                      padding: "4px 8px 4px 10px",
                      borderRadius: 999,
                      border: "1px solid #e2e8f0",
                    }}
                  >
                    <span>{contextSection.label}</span>
                    <button
                      type="button"
                      aria-label={`Retirer ${contextSection.label}`}
                      onClick={() => setContextSection(null)}
                      style={{
                        border: "none",
                        background: "none",
                        color: "#64748b",
                        cursor: "pointer",
                        fontSize: 14,
                        lineHeight: 1,
                        padding: 0,
                      }}
                    >
                      ×
                    </button>
                  </span>
                )}
              </div>
            )}
            <div style={{ display: "flex", gap: 8 }}>
              <input
                type="text"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") send();
                }}
                placeholder="Demandez les résultats de forage, les dépassements, un n° de projet…"
                style={{
                  flex: 1,
                  border: "1px solid #B5E294",
                  borderRadius: 999,
                  padding: "10px 20px",
                  fontSize: 13,
                  outline: "none",
                  background: "#f8fafc",
                }}
              />
              <button
                type="button"
                onClick={send}
                disabled={busy}
                style={{
                  background: "#50C32A",
                  color: "white",
                  border: "none",
                  borderRadius: 12,
                  padding: "0 16px",
                  cursor: busy ? "wait" : "pointer",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <Send size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
