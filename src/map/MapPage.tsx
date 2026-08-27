// src/map/MapPage.tsx

import { useMemo, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Popup,
  useMap,
} from "react-leaflet";
import type { LatLngExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

import type { PHC } from "../services/project44";

type Severity = "critical" | "high" | "medium" | "low";

function severityColor(sev: Severity): string {
  if (sev === "critical") return "#ef5b68";
  if (sev === "high") return "#f5b942";
  if (sev === "medium") return "#e8935a";
  return "#4fd9b0";
}

function buildGraticule(): LatLngExpression[][] {
  const lines: LatLngExpression[][] = [];

  for (let lng = -180; lng <= 180; lng += 30) {
    lines.push([
      [-85, lng],
      [85, lng],
    ]);
  }

  for (let lat = -60; lat <= 60; lat += 30) {
    lines.push([
      [lat, -180],
      [lat, 180],
    ]);
  }

  return lines;
}

/**
 * Forces Leaflet to recalculate its container size after mount (and on
 * window resize). Needed because react-leaflet reads the container's
 * dimensions at mount time — if the parent's layout hasn't settled yet
 * (e.g. a smaller dashboard panel vs. the full-page Live Map), the map
 * can render at the wrong size or blank until manually invalidated.
 */
function InvalidateMapSize() {
  const map = useMap();

  useEffect(() => {
    const invalidate = () => map.invalidateSize();

    invalidate();
    const settleTimer = setTimeout(invalidate, 100);

    window.addEventListener("resize", invalidate);
    return () => {
      clearTimeout(settleTimer);
      window.removeEventListener("resize", invalidate);
    };
  }, [map]);

  return null;
}

/**
 * The actual Leaflet map: tiles, graticule, and PHC markers colored/
 * sized by stock-out risk score. Reusable — fills whatever parent
 * container it's placed inside (width/height 100%, minHeight 0, so it
 * works in both the full-page Live Map and the smaller Dashboard map
 * panel).
 */
export function LiveMapCanvas({ phcs }: { phcs: PHC[] }) {
  const graticule = useMemo(buildGraticule, []);

  const counts = useMemo(() => {
    const c: Record<Severity, number> = { critical: 0, high: 0, medium: 0, low: 0 };
    for (const p of phcs) c[p.risk]++;
    return c;
  }, [phcs]);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        minHeight: 0,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Map title */}
      <div
        className="glass-card"
        style={{
          position: "absolute",
          zIndex: 1000,
          top: 14,
          left: 14,
          padding: "7px 14px",
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: "-0.005em",
          color: "var(--text)",
        }}
      >
        PHC Network · India
      </div>

      {/* Live status */}
      <div
        style={{
          position: "absolute",
          zIndex: 1000,
          top: 14,
          right: 14,
          display: "flex",
          gap: 8,
          alignItems: "center",
        }}
      >
        <div
          className="glass-card"
          style={{
            padding: "6px 12px",
            display: "flex",
            alignItems: "center",
            gap: 7,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "#4fd9b0",
              boxShadow: "0 0 0 3px rgba(79,217,176,0.18)",
              display: "inline-block",
            }}
          />

          <span
            style={{
              fontSize: 11,
              fontWeight: 500,
              color: "#4fd9b0",
            }}
          >
            Live
          </span>
        </div>

        {phcs.length === 0 && (
          <div
            className="glass-card"
            style={{
              padding: "6px 12px",
              fontSize: 11,
              color: "var(--text-2)",
            }}
          >
            Syncing data…
          </div>
        )}
      </div>

      {/* Legend */}
      <div
        className="glass-card"
        style={{
          position: "absolute",
          zIndex: 1000,
          bottom: 16,
          left: 16,
          padding: "10px 14px",
          display: "flex",
          alignItems: "center",
          gap: 14,
          flexWrap: "wrap",
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: "var(--text-2)",
            fontWeight: 600,
          }}
        >
          Risk
        </span>

        {(["critical", "high", "medium", "low"] as Severity[]).map((severity) => (
          <span
            key={severity}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: severityColor(severity),
              }}
            />

            <span
              style={{
                fontSize: 11,
                color: "var(--text-2)",
                textTransform: "capitalize",
              }}
            >
              {severity} ({counts[severity]})
            </span>
          </span>
        ))}

        <span
          style={{
            width: 1,
            height: 14,
            background: "var(--border-light)",
          }}
        />

        <span
          className="mono"
          style={{
            fontSize: 11,
            color: "var(--primary)",
            fontWeight: 500,
          }}
        >
          ● {phcs.length} PHCs monitored
        </span>
      </div>

      <MapContainer
        center={[22.9, 79.0]}
        zoom={5}
        minZoom={4}
        maxZoom={12}
        worldCopyJump={false}
        scrollWheelZoom={true}
        style={{
          width: "100%",
          height: "100%",
          minHeight: 0,
          background: "#080b12",
        }}
      >
        <InvalidateMapSize />

        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
          className="map-tiles-dark"
        />

        {/* Graticule */}
        {graticule.map((line, i) => (
          <Polyline
            key={`grid-${i}`}
            positions={line}
            pathOptions={{
              color: "#334155",
              weight: 1,
              opacity: 0.22,
            }}
          />
        ))}

        {/* PHC markers, sized and colored by stock-out risk score */}
        {phcs.map((phc) => (
          <CircleMarker
            key={`phc-${phc.id}`}
            center={[phc.lat, phc.lng]}
            radius={6 + phc.score / 12}
            pathOptions={{
              color: severityColor(phc.risk),
              weight: 1.5,
              fillColor: severityColor(phc.risk),
              fillOpacity: 0.85,
            }}
          >
            <Popup>
              <div style={{ minWidth: 160 }}>
                <strong>{phc.name}</strong>

                <div>
                  {phc.district}, {phc.state}
                </div>

                <div>Type: {phc.type}{phc.isRemote ? " (remote)" : ""}</div>

                <div>
                  Risk score: {phc.score} ({phc.risk.toUpperCase()})
                </div>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}

/**
 * Full Live Map page: LiveMapCanvas plus a right-side panel breaking
 * down PHCs by state — supports the multi-state "reach across India"
 * view alongside the map itself.
 */
export default function MapPage({ phcs }: { phcs: PHC[] }) {
  const byState = useMemo(() => {
    const map = new Map<string, PHC[]>();
    for (const p of phcs) {
      const list = map.get(p.state) ?? [];
      list.push(p);
      map.set(p.state, list);
    }
    return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [phcs]);

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        minWidth: 0,
        minHeight: 0,
        height: "100%",
        overflow: "hidden",
        background: "var(--bg)",
      }}
    >
      {/* MAP */}
      <div
        style={{
          flex: 1,
          minWidth: 0,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <LiveMapCanvas phcs={phcs} />
      </div>

      {/* RIGHT PANEL: PHCs by state */}
      <div
        style={{
          width: 260,
          borderLeft: "1px solid var(--border)",
          background: "var(--panel)",
          display: "flex",
          flexDirection: "column",
          flexShrink: 0,
          overflowY: "auto",
        }}
      >
        <div
          style={{
            padding: "10px 14px",
            borderBottom: "1px solid var(--border)",
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
          }}
        >
          PHCs by State
        </div>

        {byState.map(([state, list]) => {
          const avgScore = Math.round(list.reduce((sum, p) => sum + p.score, 0) / list.length);
          return (
            <div
              key={state}
              style={{
                padding: "10px 14px",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  marginBottom: 3,
                }}
              >
                {state}
              </div>

              <div
                className="mono"
                style={{
                  fontSize: 8,
                  color: "var(--text-3)",
                }}
              >
                {list.length} PHC{list.length !== 1 ? "s" : ""} · avg risk {avgScore}
              </div>
            </div>
          );
        })}

        {byState.length === 0 && (
          <div style={{ padding: 20, textAlign: "center", fontSize: 11, color: "var(--text-3)" }}>
            No PHC data loaded.
          </div>
        )}
      </div>
    </div>
  );
}
