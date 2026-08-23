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
  if (sev === "critical") return "#ef4444";
  if (sev === "high") return "#f59e0b";
  if (sev === "medium") return "#f97316";
  return "#22c55e";
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
        style={{
          position: "absolute",
          zIndex: 1000,
          top: 12,
          left: 12,
          padding: "5px 10px",
          background: "rgba(7,10,17,0.92)",
          border: "1px solid var(--border)",
          borderRadius: 4,
          fontSize: 9,
          fontWeight: 700,
          letterSpacing: "0.1em",
          color: "var(--text-3)",
        }}
      >
        PHC NETWORK MAP · INDIA
      </div>

      {/* Live status */}
      <div
        style={{
          position: "absolute",
          zIndex: 1000,
          top: 12,
          right: 12,
          display: "flex",
          gap: 8,
          alignItems: "center",
        }}
      >
        <div
          style={{
            padding: "4px 8px",
            background: "rgba(7,10,17,0.9)",
            border: "1px solid var(--border)",
            borderRadius: 4,
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "#22c55e",
              display: "inline-block",
            }}
          />

          <span
            className="mono"
            style={{
              fontSize: 8,
              color: "#22c55e",
            }}
          >
            LIVE
          </span>
        </div>

        {phcs.length === 0 && (
          <div
            className="mono"
            style={{
              padding: "4px 8px",
              background: "rgba(7,10,17,0.9)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              fontSize: 8,
              color: "var(--text-3)",
            }}
          >
            SYNCING DATA…
          </div>
        )}
      </div>

      {/* Legend */}
      <div
        style={{
          position: "absolute",
          zIndex: 1000,
          bottom: 14,
          left: 14,
          padding: "8px 10px",
          background: "rgba(7,10,17,0.92)",
          border: "1px solid var(--border)",
          borderRadius: 5,
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 8,
            color: "var(--text-3)",
            fontWeight: 700,
          }}
        >
          RISK
        </span>

        {(["critical", "high", "medium", "low"] as Severity[]).map((severity) => (
          <span
            key={severity}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
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
              className="mono"
              style={{
                fontSize: 8,
                color: "var(--text-3)",
                textTransform: "uppercase",
              }}
            >
              {severity} ({counts[severity]})
            </span>
          </span>
        ))}

        <span
          style={{
            width: 1,
            height: 12,
            background: "var(--border)",
          }}
        />

        <span
          className="mono"
          style={{
            fontSize: 8,
            color: "#38bdf8",
          }}
        >
          ● {phcs.length} PHCs MONITORED
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
          background: "#07111e",
        }}
      >
        <InvalidateMapSize />

        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap contributors &copy; CARTO'
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
