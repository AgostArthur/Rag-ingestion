// Client shell: map + chronologie + chat. Data comes from rag-chat catalog, not mock SITES.
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  APIProvider,
  Map,
  AdvancedMarker,
  Pin,
  useMap,
} from "@vis.gl/react-google-maps";
import { Clock, Send, ShieldCheck } from "lucide-react";
import {
  docTypeLabel,
  eventHeading,
  fetchSites,
  fetchTimeline,
  fileLabel,
  groupTimeline,
  parseQualityLabel,
  postChat,
  roleLabel,
  siteLine,
} from "./api";

const MAPS_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";
const MAP_ID = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || "DEMO_MAP_ID";
const QC_CENTER = { lat: 46.8139, lng: -71.208 };

function PanTo({ pos }) {
  const map = useMap();
  useEffect(() => {
    if (map && pos && Number.isFinite(pos.lat) && Number.isFinite(pos.lng)) {
      map.panTo(pos);
    }
  }, [map, pos]);
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
  const [focusedDoc, setFocusedDoc] = useState(null);
  const [draft, setDraft] = useState("");
  const [messages, setMessages] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [busy, setBusy] = useState(false);
  const [chatError, setChatError] = useState("");
  const chatEndRef = useRef(null);

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
        setActiveId((current) => {
          if (current && rows.some((s) => s.site_id === current)) return current;
          const withPin = rows.find((s) => pinPosition(s));
          return (withPin || rows[0] || {}).site_id || null;
        });
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
      return;
    }
    let cancelled = false;
    fetchTimeline(activeId)
      .then((payload) => {
        if (cancelled) return;
        setDocuments(payload.documents);
        setEvents(payload.events);
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

  const selectSite = useCallback((siteId) => {
    setActiveId(siteId);
    setFocusedDoc(null);
  }, []);

  const selectReport = useCallback((event) => {
    if (!event || !event.document_id) return;
    const name = fileLabel(event);
    setFocusedDoc({
      document_id: event.document_id,
      name,
      project_id: event.project_id,
    });
    setDraft((current) => {
      const mention = `@${name} `;
      if (!current.trim() || current.trim().startsWith("@")) return mention;
      if (current.includes(mention.trim())) return current;
      return `${mention}${current}`;
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
    try {
      const data = await postChat({
        message: text,
        threadId,
        siteId: activeId,
        documentId: focusedDoc ? focusedDoc.document_id : null,
      });
      setThreadId(data.thread_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.reply || "",
          citations: data.citations || [],
        },
      ]);
      const nextSite =
        data.focus && data.focus.site_ids && data.focus.site_ids[0];
      if (nextSite) {
        setActiveId(nextSite);
      }
    } catch (err) {
      setChatError(err.message || "Chat indisponible");
    } finally {
      setBusy(false);
    }
  }, [draft, busy, threadId, activeId, focusedDoc]);

  const mapCenter = activePos || QC_CENTER;
  const timelineItems = useMemo(
    () => groupTimeline(documents, events),
    [documents, events]
  );
  const nDocs =
    (activeSite && activeSite.document_ids
      ? activeSite.document_ids.length
      : documents.length) || 0;

  const mapInner = (
    <>
      {MAPS_KEY ? (
        <APIProvider apiKey={MAPS_KEY} libraries={["marker"]}>
          <Map
            mapId={MAP_ID}
            defaultCenter={QC_CENTER}
            defaultZoom={8}
            style={{ width: "100%", height: "100%" }}
            gestureHandling="greedy"
            disableDefaultUI={false}
          >
            <PanTo pos={mapCenter} />
            {sites.map((s) => {
              const pos = pinPosition(s);
              if (!pos) return null;
              const selected = activeId === s.site_id;
              return (
                <AdvancedMarker
                  key={s.site_id}
                  position={pos}
                  onClick={() => selectSite(s.site_id)}
                  zIndex={selected ? 10 : 1}
                >
                  <Pin
                    background={selected ? "#50C32A" : "#64748b"}
                    borderColor="#fff"
                    glyphColor="#fff"
                    scale={selected ? 1.4 : 1.0}
                  />
                </AdvancedMarker>
              );
            })}
          </Map>
        </APIProvider>
      ) : (
        <div
          style={{
            height: "100%",
            background: "#e2e8f0",
            padding: 24,
            overflow: "auto",
          }}
        >
          <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12 }}>
            Définissez VITE_GOOGLE_MAPS_API_KEY dans ui/.env.local pour la
            carte. Sites du catalog :
          </div>
          {sites.map((s) => (
            <button
              key={s.site_id}
              type="button"
              onClick={() => selectSite(s.site_id)}
              style={{
                display: "block",
                width: "100%",
                textAlign: "left",
                marginBottom: 8,
                padding: 10,
                borderRadius: 8,
                border:
                  activeId === s.site_id
                    ? "2px solid #50C32A"
                    : "1px solid #cbd5e1",
                background: "white",
                cursor: "pointer",
              }}
            >
              <strong>{s.address || s.site_id}</strong>
              <div style={{ fontSize: 11, color: "#64748b" }}>{s.site_id}</div>
            </button>
          ))}
        </div>
      )}

      <div
        style={{
          position: "absolute",
          top: 16,
          left: 16,
          background: "rgba(255,255,255,0.95)",
          backdropFilter: "blur(8px)",
          borderRadius: 12,
          padding: "12px 16px",
          border: "1px solid #e2e8f0",
          boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
          minWidth: 220,
          maxWidth: 320,
        }}
      >
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            color: "#94a3b8",
            textTransform: "uppercase",
            marginBottom: 4,
          }}
        >
          Site sélectionné
        </div>
        {activeSite ? (
          <>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#3DA821" }}>
              {activeSite.address || activeSite.site_id}
            </div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8 }}>
              {[activeSite.city, activeSite.lot_cadastral && `lot ${activeSite.lot_cadastral}`]
                .filter(Boolean)
                .join(" · ") || activeSite.site_id}
            </div>
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
            {activeSite.geocode_status && activeSite.geocode_status !== "ok" && (
              <div style={{ marginTop: 8, fontSize: 10, color: "#b45309" }}>
                Pas de coordonnées ({activeSite.geocode_status})
              </div>
            )}
          </>
        ) : (
          <div style={{ fontSize: 12, color: "#64748b" }}>
            {sitesError || "Aucun site dans le catalog. Ingérez des PDF."}
          </div>
        )}
        <div
          style={{
            marginTop: 8,
            paddingTop: 8,
            borderTop: "1px solid #f1f5f9",
            fontSize: 10,
            color: "#64748b",
            lineHeight: 1.4,
          }}
        >
          Une épingle = un site. Plusieurs rapports (ex. 4405 et 2259) se
          regroupent ici.
        </div>
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
          flexDirection: "column",
          padding: 12,
          gap: 12,
          overflow: "hidden",
        }}
      >
        <div style={{ flex: 3, display: "flex", gap: 12, minHeight: 0 }}>
          <div
            style={{
              flex: 2,
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
              width: 420,
              background: "white",
              borderRadius: 16,
              border: "1px solid #e2e8f0",
              boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                padding: "12px 16px",
                borderBottom: "1px solid #f1f5f9",
                background: "#f8fafc",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <Clock size={14} color="#64748b" />
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    color: "#475569",
                    textTransform: "uppercase",
                    letterSpacing: 1,
                  }}
                >
                  Chronologie
                </span>
              </div>
              {activeSite && (
                <div
                  style={{
                    marginTop: 8,
                    fontSize: 11,
                    color: "#64748b",
                    lineHeight: 1.45,
                  }}
                >
                  <div style={{ fontWeight: 600, color: "#334155" }}>
                    {activeSite.address || activeSite.site_id}
                  </div>
                  {(activeSite.city || activeSite.lot_cadastral) && (
                    <div>
                      {[
                        activeSite.city,
                        activeSite.lot_cadastral &&
                          `lot ${activeSite.lot_cadastral}`,
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                    </div>
                  )}
                </div>
              )}
            </div>
            <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>
              {timelineItems.length === 0 && (
                <div style={{ fontSize: 12, color: "#94a3b8" }}>
                  {activeId
                    ? "Aucun rapport dans le catalog pour ce site."
                    : "Sélectionnez un site sur la carte."}
                </div>
              )}
              {timelineItems.map((item, i) => {
                const r = item.document;
                const selected =
                  focusedDoc && focusedDoc.document_id === r.document_id;
                const typeLabel = docTypeLabel(r.doc_type);
                const heading = eventHeading(r);
                const place = siteLine(r);
                const fileName = fileLabel(r);
                const quality = parseQualityLabel(r.parse_quality);
                const contaminants = Array.isArray(r.contaminants)
                  ? r.contaminants.filter(Boolean)
                  : [];
                const dated = (item.events || []).filter(
                  (ev, idx, all) =>
                    all.findIndex(
                      (other) =>
                        other.iso_date === ev.iso_date && other.role === ev.role
                    ) === idx
                );
                const mainDate = r.report_date || (dated[0] && dated[0].iso_date);
                return (
                  <button
                    key={r.document_id || `${mainDate}-${i}`}
                    type="button"
                    onClick={() => selectReport(r)}
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      background: selected ? "#E8F7DD" : "transparent",
                      border: "none",
                      paddingLeft: 20,
                      paddingBottom: 20,
                      borderLeft: selected
                        ? "2px solid #50C32A"
                        : "2px solid #e2e8f0",
                      position: "relative",
                      cursor: r.document_id ? "pointer" : "default",
                    }}
                  >
                    <div
                      style={{
                        position: "absolute",
                        left: -9,
                        top: 0,
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
                        marginBottom: 4,
                        gap: 8,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 10,
                          fontFamily: "monospace",
                          color: "#94a3b8",
                        }}
                      >
                        {mainDate || "sans date"}
                      </span>
                      <span
                        style={{
                          display: "flex",
                          gap: 4,
                          flexWrap: "wrap",
                          justifyContent: "flex-end",
                        }}
                      >
                        {typeLabel ? (
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
                        {quality ? (
                          <span
                            style={{
                              fontSize: 9,
                              fontWeight: 700,
                              padding: "1px 6px",
                              background: "#f8fafc",
                              color: "#94a3b8",
                              borderRadius: 4,
                            }}
                          >
                            {quality}
                          </span>
                        ) : null}
                      </span>
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: "#1e293b",
                      }}
                    >
                      {heading}
                      {r.project_id ? ` · n° ${r.project_id}` : ""}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: "#64748b",
                        marginTop: 4,
                        lineHeight: 1.5,
                      }}
                    >
                      {r.firm ? <div>{r.firm}</div> : null}
                      {r.client ? <div>Client : {r.client}</div> : null}
                      {place ? <div>{place}</div> : null}
                      {contaminants.length > 0 ? (
                        <div>Contaminants : {contaminants.join(", ")}</div>
                      ) : null}
                      {fileName && fileName !== heading ? (
                        <div>{fileName}</div>
                      ) : null}
                      {dated.length > 0 ? (
                        <div style={{ marginTop: 6 }}>
                          {dated.map((ev) => (
                            <div key={`${ev.iso_date}-${ev.role}`}>
                              {ev.iso_date} · {roleLabel(ev.role)}
                            </div>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div
          style={{
            flex: 2,
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
                    Posez une question sur les rapports ingérés. Cliquez une
                    épingle pour limiter au site
                    {activeSite ? (
                      <>
                        {" "}
                        <strong style={{ color: "#3DA821" }}>
                          {activeSite.address || activeSite.site_id}
                        </strong>
                      </>
                    ) : null}
                    , ou un événement de la chronologie pour attacher{" "}
                    <code>@fichier</code> au prochain message.
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
                    background: m.role === "user" ? "#E8F7DD" : "#f8fafc",
                    border: "1px solid #e2e8f0",
                    borderRadius: 16,
                    padding: "10px 14px",
                    maxWidth: "80%",
                  }}
                >
                  <p
                    style={{
                      fontSize: 13,
                      color: "#334155",
                      margin: 0,
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {m.text}
                  </p>
                  {m.citations && m.citations.length > 0 && (
                    <div
                      style={{
                        marginTop: 8,
                        display: "flex",
                        gap: 6,
                        flexWrap: "wrap",
                      }}
                    >
                      {m.citations.map((c, ci) => (
                        <span
                          key={`${c.chunk_id || c.document_id}-${ci}`}
                          style={{
                            fontSize: 10,
                            background: "#E8F7DD",
                            color: "#3DA821",
                            padding: "2px 8px",
                            borderRadius: 999,
                            border: "1px solid #B5E294",
                          }}
                        >
                          📎 {c.source || c.document_id}
                          {c.page != null ? ` p.${c.page}` : ""}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {busy && (
              <div style={{ fontSize: 12, color: "#94a3b8" }}>Recherche…</div>
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
            {focusedDoc && (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    fontSize: 11,
                    background: "#E8F7DD",
                    color: "#3DA821",
                    padding: "4px 10px",
                    borderRadius: 999,
                    border: "1px solid #B5E294",
                  }}
                >
                  @{focusedDoc.name}
                  {focusedDoc.project_id ? ` · ${focusedDoc.project_id}` : ""}
                </span>
                <button
                  type="button"
                  onClick={() => setFocusedDoc(null)}
                  style={{
                    border: "none",
                    background: "none",
                    color: "#94a3b8",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                >
                  retirer
                </button>
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
