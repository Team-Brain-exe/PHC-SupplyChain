import { useState, useEffect, useRef } from "react"
import LiveMapPage, { LiveMapCanvas } from "./map/MapPage"
import {
  fetchDashboardData,
  dismissAlertApi,
  setPHCWatched,
  applyRedistributionApi,
  dismissRedistributionApi,
  generateRedistributionsApi,
  adaptRedistribution,
  notifyTeamApi,
  BACKEND_URL,
} from "./services/project44"
import type { ResourceStockView } from "./services/project44"
// ─── Types ────────────────────────────────────────────────────────────────────

type Severity = "critical" | "high" | "medium" | "low"
type NavPage = "dashboard" | "map" | "stock" | "planner" | "alerts" | "analytics" | "settings"
type DataSourceEntry = { name: string; status: string; detail: string }

type AlertEvent = {
  id: number
  time: string
  type: string
  location: string
  route: string
  severity: Severity
  summary: string
  ageMin: number
  dismissed?: boolean
}

type PHCEntry = {
  id: string
  name: string
  state: string
  district: string
  lat: number
  lng: number
  type: string
  isRemote: boolean
  risk: Severity
  score: number
  watched: boolean
}

type RedistributionEntry = {
  id: number
  originPhcId: number
  resourceType: string
  fromPhc: string
  toPhc: string
  quantity: number
  extraHours: number
  extraCost: string
  confidence: number
  reason: string
  applied?: boolean
  dismissed?: boolean
}

type MLScore = {
  score: number
  confidence: number
  features: { severity: number; age: number; overlap: number; chokepoint: number; freight: number }
}

// ─── Data ─────────────────────────────────────────────────────────────────────



const SEV_COLOR: Record<Severity, string> = {
  critical: "#ef5b68",
  high: "#f5b942",
  medium: "#e8935a",
  low: "#4fd9b0",
}
const SEV_BG: Record<Severity, string> = {
  critical: "rgba(239,91,104,0.10)",
  high: "rgba(245,185,66,0.10)",
  medium: "rgba(232,147,90,0.10)",
  low: "rgba(79,217,176,0.08)",
}

function fmtAge(min: number) {
  if (min < 60) return `${min}m ago`
  return `${Math.floor(min / 60)}h ${min % 60}m ago`
}

// ─── ML Engine ────────────────────────────────────────────────────────────────

const SEV_WEIGHT: Record<Severity, number> = { critical: 1.0, high: 0.75, medium: 0.5, low: 0.25 }

// Remoteness stands in for the old shipping-chokepoint lookup: a remote
// PHC is inherently harder/slower to resupply, so it carries higher
// baseline risk the same way a chokepoint route did.
const REMOTENESS_RISK = { remote: 0.75, standard: 0.30 }

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x))
}

function computeRiskScore(alert: AlertEvent, phcs: PHCEntry[]): MLScore {
  const sev = SEV_WEIGHT[alert.severity]
  // Recency feature: events < 2h score highest, decay over 10h
  const ageFactor = Math.max(0, 1 - alert.ageMin / 600)
  // District overlap: how many monitored PHCs share this alert's district
  const overlapCount = phcs.filter(p => p.district === alert.route || p.name === alert.location).length
  const overlap = overlapCount / Math.max(phcs.length, 1)
  const matchedPhc = phcs.find(p => p.name === alert.location)
  const choke = matchedPhc ? (matchedPhc.isRemote ? REMOTENESS_RISK.remote : REMOTENESS_RISK.standard) : REMOTENESS_RISK.standard
  const freight = sev * (choke * 0.9 + 0.1)

  // Weighted sum → sigmoid squash to 0–100
  const raw = 0.35 * sev + 0.20 * ageFactor + 0.18 * overlap + 0.17 * choke + 0.10 * freight
  const score = Math.round(sigmoid(raw * 5.2 - 2.1) * 100)
  const confidence = Math.min(Math.round(62 + sev * 24 + ageFactor * 9 + choke * 5), 97)

  return {
    score,
    confidence,
    features: {
      severity: Math.round(sev * 100) / 100,
      age: Math.round(ageFactor * 100) / 100,
      overlap: Math.round(overlap * 100) / 100,
      chokepoint: Math.round(choke * 100) / 100,
      freight: Math.round(freight * 100) / 100,
    },
  }
}

function zScoreAnomalies(series: number[]): number[] {
  const n = series.length
  if (n < 2) return series.map(() => 0)
  const mean = series.reduce((a, b) => a + b, 0) / n
  const std = Math.sqrt(series.map(x => (x - mean) ** 2).reduce((a, b) => a + b, 0) / n)
  return series.map(x => Math.abs((x - mean) / (std || 1)))
}

function linearRegression(xs: number[], ys: number[]) {
  const n = xs.length
  const sx = xs.reduce((a, b) => a + b, 0)
  const sy = ys.reduce((a, b) => a + b, 0)
  const sxy = xs.reduce((a, xi, i) => a + xi * ys[i], 0)
  const sx2 = xs.reduce((a, xi) => a + xi * xi, 0)
  const denom = n * sx2 - sx * sx
  const slope = denom !== 0 ? (n * sxy - sx * sy) / denom : 0
  const intercept = (sy - slope * sx) / n
  const yMean = sy / n
  const ssTot = ys.reduce((a, yi) => a + (yi - yMean) ** 2, 0)
  const ssRes = ys.reduce((a, yi, i) => a + (yi - (slope * xs[i] + intercept)) ** 2, 0)
  return { slope, intercept, r2: ssTot > 0 ? 1 - ssRes / ssTot : 1 }
}

// Calls the backend's /ai/risk-analysis endpoint, which runs the Gemini
// call server-side. The API key never touches the browser bundle.
async function callRiskAnalysis(payload: {
  alerts: { severity: string; type: string; location: string; summary: string }[]
  top_location?: string
  top_score?: number
  top_confidence?: number
}): Promise<string> {
  const res = await fetch(`${BACKEND_URL}/ai/risk-analysis`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as any)?.detail ?? `HTTP ${res.status}`)
  }
  const data = await res.json()
  return (data.text as string) ?? "No response."
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function LiveDot() {
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center", width: 10, height: 10 }}>
      <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#4fd9b0", animation: "pulse-ring 1.8s ease-out infinite" }} />
      <span style={{ position: "relative", width: 7, height: 7, borderRadius: "50%", background: "#4fd9b0", display: "block" }} />
    </span>
  )
}

function SeverityBadge({ sev }: { sev: Severity }) {
  return (
    <span
      className="mono"
      style={{
        fontSize: 9,
        fontWeight: 500,
        letterSpacing: "0.08em",
        padding: "2px 6px",
        borderRadius: 3,
        background: SEV_BG[sev],
        color: SEV_COLOR[sev],
        border: `1px solid ${SEV_COLOR[sev]}30`,
        whiteSpace: "nowrap",
      }}
    >
      {sev.toUpperCase()}
    </span>
  )
}

function RiskBar({ score, sev }: { score: number; sev: Severity }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{ flex: 1, height: 3, background: "#1c2530", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ width: `${score}%`, height: "100%", background: SEV_COLOR[sev], borderRadius: 2, transition: "width 0.4s ease" }} />
      </div>
      <span className="mono" style={{ fontSize: 10, color: SEV_COLOR[sev], minWidth: 24, textAlign: "right" }}>
        {score}
      </span>
    </div>
  )
}

function ShimmerBar() {
  return (
    <div style={{ height: 3, borderRadius: 2, overflow: "hidden", background: "#1c2530" }}>
      <div
        style={{
          width: "55%",
          height: "100%",
          background: "linear-gradient(90deg, #1c2530 0%, #242e3b 50%, #1c2530 100%)",
          backgroundSize: "200% 100%",
          animation: "shimmer 1.4s ease infinite",
          borderRadius: 2,
        }}
      />
    </div>
  )
}

function Btn({
  children,
  onClick,
  small,
  danger,
}: {
  children: React.ReactNode
  onClick?: () => void
  small?: boolean
  danger?: boolean
}) {
  const [hover, setHover] = useState(false)
  const bg = danger ? (hover ? "#ef5b68" : "rgba(239,91,104,0.12)") : hover ? "var(--primary)" : "var(--primary-dim)"
  const col = danger ? (hover ? "#fff" : "#ef5b68") : hover ? "#000" : "var(--primary)"
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        padding: small ? "4px 10px" : "6px 14px",
        fontSize: small ? 11 : 12,
        fontFamily: "var(--font-ui)",
        fontWeight: 600,
        background: bg,
        color: col,
        border: `1px solid ${danger ? "rgba(239,91,104,0.3)" : "rgba(54,199,165,0.3)"}`,
        borderRadius: 6,
        cursor: "pointer",
        transition: "all 0.15s",
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </button>
  )
}

function FeatureChart({ features }: { features: MLScore["features"] }) {
  const items = [
    { key: "severity", label: "SEV", val: features.severity },
    { key: "chokepoint", label: "CHOKE", val: features.chokepoint },
    { key: "overlap", label: "OVERLAP", val: features.overlap },
    { key: "age", label: "RECENCY", val: features.age },
    { key: "freight", label: "FREIGHT", val: features.freight },
  ]
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 6 }}>
      {items.map(item => (
        <div key={item.key} style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span className="mono" style={{ fontSize: 7, color: "var(--text-3)", width: 42, textAlign: "right" }}>
            {item.label}
          </span>
          <div style={{ flex: 1, height: 2, background: "#1c2530", borderRadius: 1, overflow: "hidden" }}>
            <div
              style={{
                width: `${item.val * 100}%`,
                height: "100%",
                background: "var(--primary)",
                opacity: 0.65,
                borderRadius: 1,
                transition: "width 0.5s ease",
              }}
            />
          </div>
          <span className="mono" style={{ fontSize: 7, color: "var(--text-3)", width: 26 }}>
            {(item.val * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  )
}
// ─── Analytics Page ───────────────────────────────────────────────────────────

// Replaces the old shipping/freight AnalyticsPage. Everything here is
// derived from real props (alerts, stocks, phcs, reroutes) — nothing hardcoded.
function AnalyticsPage({
  alerts,
  stocks,
  phcs,
  reroutes,
}: {
  alerts: AlertEvent[]
  stocks: ResourceStockView[]
  phcs: PHCEntry[]
  reroutes: RedistributionEntry[]
}) {
  const typeCounts: Record<string, number> = {}
  alerts.forEach(a => {
    typeCounts[a.type] = (typeCounts[a.type] || 0) + 1
  })
  const typeBars = Object.entries(typeCounts).sort((a, b) => b[1] - a[1])
  const maxTypeBar = Math.max(1, ...typeBars.map(([, c]) => c))

  const byCategory: Record<string, number[]> = {}
  stocks.forEach(s => {
    if (!byCategory[s.category]) byCategory[s.category] = []
    byCategory[s.category].push(s.daysRemaining)
  })
  const categoryAverages = Object.entries(byCategory)
    .map(([cat, vals]) => ({ cat, avg: vals.reduce((a, b) => a + b, 0) / vals.length }))
    .sort((a, b) => a.avg - b.avg)

  const byState: Record<string, PHCEntry[]> = {}
  phcs.forEach(p => {
    if (!byState[p.state]) byState[p.state] = []
    byState[p.state].push(p)
  })
  const stateRows = Object.entries(byState)
    .map(([state, list]) => ({
      state,
      count: list.length,
      avgScore: Math.round(list.reduce((a, p) => a + p.score, 0) / list.length),
      critical: list.filter(p => p.risk === "critical").length,
    }))
    .sort((a, b) => b.avgScore - a.avgScore)

  const totalSuggestions = reroutes.length
  const applied = reroutes.filter(r => r.applied).length
  const avgConfidence = totalSuggestions
    ? Math.round(reroutes.reduce((a, r) => a + r.confidence, 0) / totalSuggestions)
    : 0

  return (
    <div style={{ padding: 24, overflowY: "auto", height: "100%" }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20 }}>Analytics</h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 6, padding: 16 }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-3)", textTransform: "uppercase", marginBottom: 14 }}>
            Active Alerts by Type
          </div>
          {typeBars.length === 0 ? (
            <div style={{ fontSize: 11, color: "var(--text-3)" }}>No alerts recorded.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {typeBars.map(([type, count]) => (
                <div key={type} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 9, color: "var(--text-3)", width: 100 }}>{type}</span>
                  <div style={{ flex: 1, height: 8, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${(count / maxTypeBar) * 100}%`, height: "100%", background: "var(--primary)", opacity: 0.75 }} />
                  </div>
                  <span className="mono" style={{ fontSize: 9, color: "var(--text-2)", width: 20, textAlign: "right" }}>{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 6, padding: 16 }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-3)", textTransform: "uppercase", marginBottom: 14 }}>
            Avg Days of Stock Remaining
          </div>
          {categoryAverages.length === 0 ? (
            <div style={{ fontSize: 11, color: "var(--text-3)" }}>No stock records available.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {categoryAverages.map(({ cat, avg }) => {
                const sev = stockSeverity(avg)
                return (
                  <div key={cat} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 9, color: "var(--text-3)", width: 100 }}>{cat}</span>
                    <div style={{ flex: 1, height: 8, background: "var(--panel-2)", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ width: `${Math.min(100, (avg / 14) * 100)}%`, height: "100%", background: SEV_COLOR[sev] }} />
                    </div>
                    <span className="mono" style={{ fontSize: 9, color: SEV_COLOR[sev], width: 34, textAlign: "right" }}>
                      {avg.toFixed(1)}d
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      <div
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 6,
          padding: "10px 16px",
          marginBottom: 16,
          display: "flex",
          alignItems: "center",
          gap: 20,
          flexWrap: "wrap",
        }}
      >
        <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-3)", textTransform: "uppercase" }}>
          Redistribution Activity
        </span>
        <span className="mono" style={{ fontSize: 11, color: "var(--text)" }}>{totalSuggestions} suggested</span>
        <span className="mono" style={{ fontSize: 11, color: "#4fd9b0" }}>{applied} applied</span>
        <span className="mono" style={{ fontSize: 11, color: "var(--primary)" }}>{avgConfidence}% avg confidence</span>
      </div>

      <div style={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 6, overflow: "hidden" }}>
        <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", color: "var(--text-3)", textTransform: "uppercase" }}>
          Risk by State
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["State", "Facilities", "Avg Risk Score", "Critical"].map(h => (
                <th key={h} style={{ padding: "8px 16px", fontSize: 11, color: "var(--text-2)", fontFamily: "var(--font-ui)", letterSpacing: "0.02em", textAlign: "left", fontWeight: 600 }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {stateRows.map((row, i) => (
              <tr key={row.state} style={{ borderBottom: "1px solid var(--border)", background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                <td style={{ padding: "8px 16px", fontSize: 11, color: "var(--text)", fontWeight: 500 }}>{row.state}</td>
                <td className="mono" style={{ padding: "8px 16px", fontSize: 11, color: "var(--text-2)" }}>{row.count}</td>
                <td className="mono" style={{ padding: "8px 16px", fontSize: 11, color: row.avgScore >= 65 ? "#ef5b68" : "var(--text-2)" }}>{row.avgScore}</td>
                <td className="mono" style={{ padding: "8px 16px", fontSize: 11, color: row.critical > 0 ? "#ef5b68" : "var(--text-3)" }}>{row.critical}</td>
              </tr>
            ))}
            {stateRows.length === 0 && (
              <tr><td colSpan={4} style={{ padding: 24, textAlign: "center", color: "var(--text-3)", fontSize: 11 }}>No PHC data available.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}


function RoutePlannerPage({
  phcs,
  reroutes,
  onGenerateForPhc,
  generatingPhcId,
  onApplyReroute,
  onDismissReroute,
}: {
  phcs: PHCEntry[]
  reroutes: RedistributionEntry[]
  onGenerateForPhc: (phcId: string) => Promise<void>
  generatingPhcId: string | null
  onApplyReroute: (id: number) => void
  onDismissReroute: (id: number) => void
}) {
  const [selectedPhcId, setSelectedPhcId] = useState<string>("")

  const atRiskPhcs = phcs
    .filter(p => p.risk === "critical" || p.risk === "high" || p.risk === "medium")
    .sort((a, b) => b.score - a.score)

  const selectedPhc = phcs.find(p => p.id === selectedPhcId) ?? null

  const suggestionsForSelected = selectedPhc
    ? reroutes.filter(r => r.toPhc === selectedPhc.name && !r.dismissed)
    : []

  const exportPlanPdf = () => {
    if (!selectedPhc) return
    const w = window.open("", "_blank")
    if (!w) return
    const rows = suggestionsForSelected
      .map(
        r => `<tr>
          <td>${r.fromPhc}</td>
          <td>${r.resourceType}</td>
          <td>${r.quantity}</td>
          <td>+${r.extraHours}h</td>
          <td>${r.extraCost}</td>
          <td>${r.confidence}%</td>
        </tr>`
      )
      .join("")
    w.document.write(`
      <html>
        <head>
          <title>Redistribution Plan - ${selectedPhc.name}</title>
          <style>
            body { font-family: -apple-system, sans-serif; padding: 40px; color: #111; }
            h1 { font-size: 20px; margin-bottom: 4px; }
            .sub { color: #666; margin-bottom: 24px; font-size: 13px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; text-align: left; }
            th { color: #666; font-weight: 600; }
          </style>
        </head>
        <body>
          <h1>PHC-Nexus Redistribution Plan</h1>
          <div class="sub">${selectedPhc.name} · ${selectedPhc.district}, ${selectedPhc.state} · risk ${selectedPhc.score}/100</div>
          <table>
            <thead><tr><th>From Facility</th><th>Resource</th><th>Qty</th><th>Extra Time</th><th>Extra Cost</th><th>Confidence</th></tr></thead>
            <tbody>${rows || `<tr><td colspan="6">No suggestions generated yet.</td></tr>`}</tbody>
          </table>
        </body>
      </html>
    `)
    w.document.close()
    w.focus()
    w.print()
  }

  return (
    <div style={{ padding: 24, overflowY: "auto", height: "100%" }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 20 }}>Redistribution Planner</h2>
      <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div>
            <div
              style={{
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: "0.1em",
                color: "var(--text-3)",
                textTransform: "uppercase",
                marginBottom: 6,
              }}
            >
              Facility in Need
            </div>
            <select
              value={selectedPhcId}
              onChange={e => setSelectedPhcId(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 10px",
                background: "var(--panel)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                color: "var(--text)",
                fontSize: 12,
                fontFamily: "var(--font-ui)",
                cursor: "pointer",
              }}
            >
              <option value="">Select a PHC/CHC…</option>
              {atRiskPhcs.map(p => (
                <option key={p.id} value={p.id}>
                  {p.name} — {p.district} ({p.risk.toUpperCase()}, {p.score})
                </option>
              ))}
            </select>
            {atRiskPhcs.length === 0 && (
              <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 6 }}>
                No facilities currently at medium/high/critical risk.
              </div>
            )}
          </div>

          <button
            disabled={!selectedPhcId || generatingPhcId === selectedPhcId}
            onClick={() => selectedPhcId && onGenerateForPhc(selectedPhcId)}
            style={{
              marginTop: 4,
              padding: "10px",
              background: "var(--primary)",
              color: "#000",
              fontWeight: 700,
              fontSize: 11,
              fontFamily: "DM Mono, monospace",
              letterSpacing: "0.08em",
              border: "none",
              borderRadius: 4,
              cursor: !selectedPhcId ? "not-allowed" : "pointer",
              opacity: !selectedPhcId ? 0.5 : generatingPhcId === selectedPhcId ? 0.6 : 1,
            }}
          >
            {generatingPhcId === selectedPhcId ? "Finding options…" : "Find redistribution options"}
          </button>

          {selectedPhc && (
            <div
              style={{
                background: "var(--panel)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: 12,
                fontSize: 11,
                color: "var(--text-2)",
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 4 }}>{selectedPhc.name}</div>
              <div style={{ color: "var(--text-3)", marginBottom: 6 }}>
                {selectedPhc.district}, {selectedPhc.state} · {selectedPhc.type} · {selectedPhc.isRemote ? "remote" : "accessible"}
              </div>
              <RiskBar score={selectedPhc.score} sev={selectedPhc.risk} />
            </div>
          )}
        </div>

        {selectedPhc ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div
              style={{
                fontSize: 10,
                color: "var(--text-3)",
                fontWeight: 600,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                display: "flex",
                alignItems: "center",
              }}
            >
              <span>{suggestionsForSelected.length} option(s) for {selectedPhc.name}</span>
              {suggestionsForSelected.length > 0 && (
                <button
                  onClick={exportPlanPdf}
                  style={{
                    marginLeft: "auto",
                    padding: "4px 10px",
                    fontSize: 9,
                    fontFamily: "DM Mono, monospace",
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    background: "transparent",
                    color: "var(--text-3)",
                    border: "1px solid var(--border)",
                    borderRadius: 4,
                    cursor: "pointer",
                  }}
                >
                  EXPORT PDF
                </button>
              )}
            </div>

            {suggestionsForSelected.map(rr => (
              <div
                key={rr.id}
                style={{
                  background: "var(--panel)",
                  border: `1px solid ${rr.applied ? "#4fd9b060" : "var(--border)"}`,
                  borderLeft: "2px solid #4fd9b0",
                  borderRadius: 6,
                  padding: 16,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                  <span style={{ fontSize: 12, fontWeight: 600 }}>
                    {rr.fromPhc} → {rr.toPhc}
                  </span>
                  {rr.applied && (
                    <span
                      style={{
                        fontSize: 8,
                        fontWeight: 700,
                        color: "#4fd9b0",
                        padding: "2px 6px",
                        background: "rgba(79,217,176,0.1)",
                        borderRadius: 3,
                        letterSpacing: "0.08em",
                      }}
                    >
                      APPLIED
                    </span>
                  )}
                  <span style={{ marginLeft: "auto", fontSize: 9, color: "var(--text-3)" }}>
                    {rr.resourceType} · {rr.quantity} units
                  </span>
                </div>
                <div style={{ display: "flex", gap: 20, marginBottom: 10 }}>
                  {[
                    ["Extra Time", `+${rr.extraHours}h`],
                    ["Extra Cost", rr.extraCost],
                    ["Confidence", `${rr.confidence}%`],
                  ].map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 9, color: "var(--text-3)", marginBottom: 2 }}>{k}</div>
                      <div className="mono" style={{ fontSize: 14, fontWeight: 500, color: "var(--text)" }}>
                        {v}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-2)", lineHeight: 1.5, marginBottom: rr.applied ? 0 : 10 }}>
                  {rr.reason}
                </div>
                {!rr.applied && (
                  <div style={{ display: "flex", gap: 8 }}>
                    <Btn onClick={() => onApplyReroute(rr.id)}>APPLY</Btn>
                    <Btn danger onClick={() => onDismissReroute(rr.id)}>
                      DISMISS
                    </Btn>
                  </div>
                )}
              </div>
            ))}

            {suggestionsForSelected.length === 0 && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "var(--panel)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  color: "var(--text-3)",
                  fontSize: 12,
                  padding: 40,
                }}
              >
                No suggestions yet — click "Find Redistribution Options" above.
              </div>
            )}
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "var(--panel)",
              border: "1px solid var(--border)",
              borderRadius: 6,
              color: "var(--text-3)",
              fontSize: 12,
            }}
          >
            Select a facility to plan a redistribution
          </div>
        )}
      </div>
    </div>
  )
}


function AlertsPage({ alerts, onDismiss, onNotify }: { alerts: AlertEvent[]; onDismiss: (id: number) => void; onNotify: (id: number) => void }) {
  const [filter, setFilter] = useState<Severity | "all">("all")
  const filtered = filter === "all" ? alerts : alerts.filter(a => a.severity === filter)

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginRight: 8,
          }}
        >
          Filter:
        </span>
        {(["all", "critical", "high", "medium", "low"] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: "3px 10px",
              fontSize: 9,
              fontFamily: "DM Mono, monospace",
              fontWeight: 600,
              letterSpacing: "0.08em",
              borderRadius: 3,
              border: `1px solid ${filter === f && f !== "all" ? SEV_COLOR[f as Severity] + "60" : "var(--border)"}`,
              background: filter === f ? (f === "all" ? "var(--primary-dim)" : SEV_BG[f as Severity]) : "transparent",
              color: filter === f ? (f === "all" ? "var(--primary)" : SEV_COLOR[f as Severity]) : "var(--text-3)",
              cursor: "pointer",
              textTransform: "uppercase",
            }}
          >
            {f}
          </button>
        ))}
        <span className="mono" style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-3)" }}>
          {filtered.filter(a => !a.dismissed).length} active
        </span>
      </div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {filtered
          .filter(a => !a.dismissed)
          .map(a => (
            <div
              key={a.id}
              style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", borderLeft: `2px solid ${SEV_COLOR[a.severity]}` }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span className="mono" style={{ fontSize: 9, color: "var(--text-3)" }}>
                  {a.time}
                </span>
                <span
                  className="mono"
                  style={{
                    fontSize: 8,
                    letterSpacing: "0.1em",
                    fontWeight: 600,
                    padding: "1px 5px",
                    background: SEV_BG[a.severity],
                    color: SEV_COLOR[a.severity],
                    borderRadius: 2,
                  }}
                >
                  {a.type}
                </span>
                <SeverityBadge sev={a.severity} />
                <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                  <Btn small onClick={() => onNotify(a.id)}>NOTIFY TEAM</Btn>
                  <Btn small danger onClick={() => onDismiss(a.id)}>
                    DISMISS
                  </Btn>
                </span>
              </div>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 3 }}>{a.location}</div>
              <div className="mono" style={{ fontSize: 9, color: "var(--text-3)", marginBottom: 4 }}>
                {a.route}
              </div>
              <div style={{ fontSize: 11, color: "var(--text-2)", lineHeight: 1.6 }}>{a.summary}</div>
            </div>
          ))}
        {filtered.filter(a => !a.dismissed).length === 0 && (
          <div style={{ padding: 40, textAlign: "center", color: "var(--text-3)", fontSize: 12 }}>No active alerts for this filter.</div>
        )}
      </div>
    </div>
  )
}

function stockSeverity(daysRemaining: number): Severity {
  if (daysRemaining < 2) return "critical"
  if (daysRemaining < 5) return "high"
  if (daysRemaining < 10) return "medium"
  return "low"
}

function StockPage({ stocks, phcs }: { stocks: ResourceStockView[]; phcs: PHCEntry[] }) {
  const [stateFilter, setStateFilter] = useState<string>("all")
  const phcById = new Map(phcs.map(p => [Number(p.id), p]))
  const states = ["all", ...Array.from(new Set(phcs.map(p => p.state))).sort()]

  const filtered = stocks.filter(s => {
    if (stateFilter === "all") return true
    const phc = phcById.get(s.phcId)
    return phc?.state === stateFilter
  })

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginRight: 8,
          }}
        >
          State:
        </span>
        {states.map(s => (
          <button
            key={s}
            onClick={() => setStateFilter(s)}
            style={{
              padding: "3px 10px",
              fontSize: 9,
              fontFamily: "DM Mono, monospace",
              fontWeight: 600,
              letterSpacing: "0.08em",
              borderRadius: 3,
              border: `1px solid ${stateFilter === s ? "var(--primary)60" : "var(--border)"}`,
              background: stateFilter === s ? "var(--primary-dim)" : "transparent",
              color: stateFilter === s ? "var(--primary)" : "var(--text-3)",
              cursor: "pointer",
              textTransform: "uppercase",
            }}
          >
            {s}
          </button>
        ))}
        <span className="mono" style={{ marginLeft: "auto", fontSize: 10, color: "var(--text-3)" }}>
          {filtered.length} records
        </span>
      </div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        {filtered.map(s => {
          const phc = phcById.get(s.phcId)
          const sev = stockSeverity(s.daysRemaining)
          const stockPct = s.capacity ? Math.round((s.units / s.capacity) * 100) : 0
          const bedPct = s.bedTotal ? Math.round((s.bedOccupied / s.bedTotal) * 100) : 0
          const staffPct = s.staffTotal ? Math.round((s.staffPresent / s.staffTotal) * 100) : 0
          return (
            <div
              key={s.id}
              style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", borderLeft: `2px solid ${SEV_COLOR[sev]}` }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600 }}>{phc?.name ?? `PHC #${s.phcId}`}</span>
                <span className="mono" style={{ fontSize: 9, color: "var(--text-3)" }}>
                  {phc ? `${phc.district}, ${phc.state}` : ""}
                </span>
                <span
                  className="mono"
                  style={{
                    fontSize: 8,
                    letterSpacing: "0.1em",
                    fontWeight: 600,
                    padding: "1px 5px",
                    background: SEV_BG[sev],
                    color: SEV_COLOR[sev],
                    borderRadius: 2,
                  }}
                >
                  {s.category.toUpperCase()}
                </span>
                <span className="mono" style={{ marginLeft: "auto", fontSize: 9, color: SEV_COLOR[sev] }}>
                  {s.daysRemaining.toFixed(1)}d of stock left
                </span>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                <div>
                  <div className="mono" style={{ fontSize: 8, color: "var(--text-3)", marginBottom: 2 }}>
                    STOCK {s.units}/{s.capacity} ({stockPct}%)
                  </div>
                  <div style={{ height: 4, background: "var(--panel-2)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${stockPct}%`, height: "100%", background: SEV_COLOR[sev] }} />
                  </div>
                </div>
                <div>
                  <div className="mono" style={{ fontSize: 8, color: "var(--text-3)", marginBottom: 2 }}>
                    BEDS {s.bedOccupied}/{s.bedTotal} ({bedPct}%)
                  </div>
                  <div style={{ height: 4, background: "var(--panel-2)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${bedPct}%`, height: "100%", background: "#5ab8c9" }} />
                  </div>
                </div>
                <div>
                  <div className="mono" style={{ fontSize: 8, color: "var(--text-3)", marginBottom: 2 }}>
                    STAFF {s.staffPresent}/{s.staffTotal} ({staffPct}%)
                  </div>
                  <div style={{ height: 4, background: "var(--panel-2)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${staffPct}%`, height: "100%", background: "#8b7fd4" }} />
                  </div>
                </div>
              </div>
              <div className="mono" style={{ fontSize: 9, color: "var(--text-3)", marginTop: 6 }}>
                Footfall: {s.footfallDaily}/day · Updated {s.updatedAt}
              </div>
            </div>
          )
        })}
        {filtered.length === 0 && (
          <div style={{ padding: 40, textAlign: "center", color: "var(--text-3)", fontSize: 12 }}>No stock records for this filter.</div>
        )}
      </div>
    </div>
  )
}


function SettingsPage() {
  const [notif, setNotif] = useState({ email: true, whatsapp: false, sms: true })
  const [thresh, setThresh] = useState<Severity>("high")
  const [contact, setContact] = useState({ name: "", email: "", phone: "" })
  const [saved, setSaved] = useState(false)
  const [loading, setLoading] = useState(true)
  const [sources, setSources] = useState<DataSourceEntry[]>([])

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [settingsRes, statusRes] = await Promise.all([
          fetch(`${BACKEND_URL}/settings`),
          fetch(`${BACKEND_URL}/settings/system-status`),
        ])
        const settingsData = await settingsRes.json()
        const statusData = await statusRes.json()
        if (cancelled) return

        setNotif({
          email: settingsData.notif_email,
          whatsapp: settingsData.notif_whatsapp,
          sms: settingsData.notif_sms,
        })
        setThresh(settingsData.alert_threshold as Severity)
        setContact({
          name: settingsData.contact_name || "",
          email: settingsData.contact_email || "",
          phone: settingsData.contact_phone || "",
        })
        setSources(statusData.sources || [])
      } catch (e) {
        console.error("Failed to load settings", e)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  async function save() {
    try {
      await fetch(`${BACKEND_URL}/settings`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          notif_email: notif.email,
          notif_whatsapp: notif.whatsapp,
          notif_sms: notif.sms,
          alert_threshold: thresh,
          contact_name: contact.name,
          contact_email: contact.email,
          contact_phone: contact.phone,
        }),
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error("Failed to save settings", e)
    }
  }

  if (loading) {
    return (
      <div style={{ padding: 24, fontSize: 12, color: "var(--text-3)" }}>Loading settings…</div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 560, overflowY: "auto", height: "100%" }}>
      <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 24 }}>Settings</h2>

      <section style={{ marginBottom: 28 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginBottom: 12,
          }}
        >
          Contact Info
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 16 }}>
          <input
            value={contact.name}
            onChange={e => setContact(c => ({ ...c, name: e.target.value }))}
            placeholder="Your name"
            style={{
              padding: "8px 10px",
              fontSize: 11,
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-1)",
            }}
          />
          <input
            value={contact.email}
            onChange={e => setContact(c => ({ ...c, email: e.target.value }))}
            placeholder="Email address"
            style={{
              padding: "8px 10px",
              fontSize: 11,
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-1)",
            }}
          />
          <input
            value={contact.phone}
            onChange={e => setContact(c => ({ ...c, phone: e.target.value }))}
            placeholder="Phone number"
            style={{
              padding: "8px 10px",
              fontSize: 11,
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-1)",
            }}
          />
        </div>
      </section>

      <section style={{ marginBottom: 28 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginBottom: 12,
          }}
        >
          Alert Notifications
        </div>
        {(["email", "whatsapp", "sms"] as const).map(ch => (
          <div
            key={ch}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "10px 0",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <div>
              <div style={{ fontSize: 12, fontWeight: 500, textTransform: "capitalize", marginBottom: 2 }}>
                {ch === "sms" ? "SMS" : ch.charAt(0).toUpperCase() + ch.slice(1)}
              </div>
              <div style={{ fontSize: 10, color: "var(--text-3)" }}>
                {ch === "email" ? (contact.email || "No email set") : (contact.phone || "No phone set")}
              </div>
            </div>
            <div
              onClick={() => setNotif(n => ({ ...n, [ch]: !n[ch] }))}
              style={{
                width: 40,
                height: 22,
                borderRadius: 11,
                background: notif[ch] ? "var(--primary)" : "var(--border)",
                position: "relative",
                cursor: "pointer",
                transition: "background 0.2s",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 3,
                  left: notif[ch] ? 20 : 3,
                  width: 16,
                  height: 16,
                  borderRadius: "50%",
                  background: "#fff",
                  transition: "left 0.2s",
                }}
              />
            </div>
          </div>
        ))}
      </section>

      <section style={{ marginBottom: 28 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginBottom: 12,
          }}
        >
          Alert Threshold
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {(["critical", "high", "medium", "low"] as const).map(s => (
            <button
              key={s}
              onClick={() => setThresh(s)}
              style={{
                flex: 1,
                padding: "8px 4px",
                borderRadius: 4,
                border: `1px solid ${thresh === s ? SEV_COLOR[s] + "60" : "var(--border)"}`,
                background: thresh === s ? SEV_BG[s] : "transparent",
                color: thresh === s ? SEV_COLOR[s] : "var(--text-3)",
                fontSize: 9,
                fontFamily: "DM Mono, monospace",
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              {s}
            </button>
          ))}
        </div>
        <div style={{ fontSize: 10, color: "var(--text-3)", marginTop: 8 }}>
          Receive alerts for severity ≥ <span style={{ color: SEV_COLOR[thresh] }}>{thresh}</span>
        </div>
      </section>

      <section style={{ marginBottom: 28 }}>
        <div
          style={{
            fontSize: 10,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "var(--text-3)",
            textTransform: "uppercase",
            marginBottom: 12,
          }}
        >
          Data Sources
        </div>
        {sources.map(({ name, status, detail }) => (
          <div
            key={name}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "8px 0",
              borderBottom: "1px solid var(--border)",
            }}
          >
            <span style={{ fontSize: 11, color: "var(--text-2)" }}>{name}</span>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 9, color: "var(--text-3)" }}>{detail}</span>
              <span
                className="mono"
                style={{
                  fontSize: 9,
                  fontWeight: 600,
                  color: status === "Active" ? "#4fd9b0" : "#f5b942",
                  padding: "2px 6px",
                  background: status === "Active" ? "rgba(79,217,176,0.1)" : "rgba(245,185,66,0.1)",
                  borderRadius: 3,
                }}
              >
                {status}
              </span>
            </div>
          </div>
        ))}
      </section>

      <div style={{ display: "flex", gap: 10 }}>
        <button
          onClick={save}
          style={{
            padding: "9px 24px",
            background: saved ? "#4fd9b0" : "var(--primary)",
            color: "#000",
            fontWeight: 700,
            fontSize: 11,
            fontFamily: "DM Mono, monospace",
            letterSpacing: "0.08em",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
            transition: "background 0.3s",
          }}
        >
          {saved ? "Saved ✓" : "Save settings"}
        </button>
      </div>
    </div>
  )
}


// ─── Dashboard View ───────────────────────────────────────────────────────────

function DashboardView({
  alerts,
  routes,
  reroutes,
  activeAlert,
  setActiveAlert,
  onDismissAlert,
  onToggleWatch,
  onApplyReroute,
  onDismissReroute,
  onGenerateReroutes,
  generatingReroutes,
  mlScores,
  mlRunning,
  aiAnalysis,
  aiLoading,
  onGenerateAI,
  expandedMLId,
  setExpandedMLId,
}: {
  alerts: AlertEvent[]
  routes: PHCEntry[]
  reroutes: RedistributionEntry[]
  activeAlert: number | null
  setActiveAlert: (id: number | null) => void
  onDismissAlert: (id: number) => void
  onToggleWatch: (id: string) => void
  onApplyReroute: (id: number) => void
  onDismissReroute: (id: number) => void
  onGenerateReroutes: () => void
  generatingReroutes: boolean
  mlScores: Record<number, MLScore>
  mlRunning: boolean
  aiAnalysis: string
  aiLoading: boolean
  onGenerateAI: () => void
  expandedMLId: number | null
  setExpandedMLId: (id: number | null) => void
}) {
  const [activeTab, setActiveTab] = useState<"alerts" | "watchlist">("alerts")
  const liveAlerts = alerts.filter(a => !a.dismissed)

  const statesCovered = new Set(routes.map(r => r.state)).size
  const criticalPhcs = routes.filter(r => r.risk === "critical").length
  const avgRiskScore = routes.length ? Math.round(routes.reduce((sum, r) => sum + r.score, 0) / routes.length) : 0
  const dashboardStats: { label: string; value: string; delta: string | null; sub: string }[] = [
    { label: "PHCs Monitored", value: String(routes.length), delta: null, sub: "in network" },
    { label: "States Covered", value: String(statesCovered), delta: null, sub: "across India" },
    { label: "Active Stock-Outs", value: String(criticalPhcs), delta: null, sub: "critical risk" },
    { label: "Avg Risk Score", value: String(avgRiskScore), delta: null, sub: "0–100 scale" },
  ]

  return (
    <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
      {/* Left column: map, stats, tabs, list */}
      <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
        {/* Map */}
        <div style={{ flex: 1, minHeight: 0, position: "relative", borderBottom: "1px solid var(--border)" }}>
          <div
            style={{
              width: "100%",
              height: "100%",
              minHeight: 0,
              position: "relative",
              overflow: "hidden",
            }}
          >
            <LiveMapCanvas phcs={routes} />
          </div>
          {mlRunning && (
            <div style={{ position: "absolute", top: 44, right: 12 }}>
              <div
                style={{
                  padding: "3px 8px",
                  background: "rgba(54,199,165,0.07)",
                  border: "1px solid rgba(54,199,165,0.22)",
                  borderRadius: 4,
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                }}
              >
                <span
                  style={{
                    width: 5,
                    height: 5,
                    borderRadius: "50%",
                    background: "var(--primary)",
                    animation: "blink 0.8s ease infinite",
                    display: "inline-block",
                  }}
                />
                <span className="mono" style={{ fontSize: 8, color: "var(--primary)" }}>
                  ML MODEL RUNNING
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Stats row */}
        <div style={{ display: "flex", gap: 12, padding: "16px 16px 0", flexShrink: 0 }}>
          {dashboardStats.map((s, i) => {
            const isNeg = s.delta?.startsWith("−") || s.delta?.startsWith("-")
            return (
              <div key={i} className="glass-card" style={{ padding: "14px 16px", flex: 1 }}>
                <div
                  style={{
                    fontSize: 12,
                    color: "var(--text-2)",
                    fontWeight: 500,
                    marginBottom: 8,
                  }}
                >
                  {s.label}
                </div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                  <span className="mono" style={{ fontSize: 24, fontWeight: 600, color: "var(--text)", letterSpacing: "-0.02em" }}>
                    {s.value}
                  </span>
                  {s.delta && (
                    <span className="mono" style={{ fontSize: 11, color: isNeg ? "#ef5b68" : "#4fd9b0" }}>
                      {s.delta}
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 4 }}>{s.sub}</div>
              </div>
            )
          })}
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", borderBottom: "1px solid var(--border)", flexShrink: 0, padding: "16px 16px 0", gap: 4 }}>
          {(["alerts", "watchlist"] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "9px 16px",
                background: "transparent",
                border: "none",
                borderBottom: `2px solid ${activeTab === tab ? "var(--primary)" : "transparent"}`,
                color: activeTab === tab ? "var(--primary)" : "var(--text-2)",
                fontSize: 13,
                fontWeight: 500,
                cursor: "pointer",
                fontFamily: "var(--font-ui)",
              }}
            >
              {tab === "alerts" ? `Health Alerts (${liveAlerts.length})` : `PHC Watchlist (${routes.filter(r => r.watched).length})`}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ flex: 1, overflowY: "auto", minHeight: 0 }}>
          {activeTab === "alerts"
            ? liveAlerts.map(a => {
                const ml = mlScores[a.id]
                const expanded = expandedMLId === a.id
                return (
                  <div
                    key={a.id}
                    onClick={() => setActiveAlert(activeAlert === a.id ? null : a.id)}
                    style={{
                      padding: "10px 14px",
                      borderBottom: "1px solid var(--border)",
                      cursor: "pointer",
                      background: activeAlert === a.id ? SEV_BG[a.severity] : "transparent",
                      borderLeft: `2px solid ${activeAlert === a.id ? SEV_COLOR[a.severity] : "transparent"}`,
                      transition: "all 0.15s",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                      <span className="mono" style={{ fontSize: 9, color: "var(--text-3)" }}>
                        {a.time}
                      </span>
                      <span
                        className="mono"
                        style={{
                          fontSize: 8,
                          letterSpacing: "0.1em",
                          fontWeight: 600,
                          padding: "1px 5px",
                          background: SEV_BG[a.severity],
                          color: SEV_COLOR[a.severity],
                          borderRadius: 2,
                        }}
                      >
                        {a.type}
                      </span>
                      <SeverityBadge sev={a.severity} />
                      {ml && (
                        <button
                          onClick={e => {
                            e.stopPropagation()
                            setExpandedMLId(expanded ? null : a.id)
                          }}
                          style={{
                            fontSize: 8,
                            fontFamily: "DM Mono, monospace",
                            padding: "1px 6px",
                            background: "var(--primary-dim)",
                            color: "var(--primary)",
                            border: "1px solid var(--primary)30",
                            borderRadius: 2,
                            cursor: "pointer",
                          }}
                        >
                          ML {ml.score} · {ml.confidence}% {expanded ? "▴" : "▾"}
                        </button>
                      )}
                      {!ml && mlRunning && (
                        <span className="mono" style={{ fontSize: 7, color: "var(--text-3)", animation: "blink 1s infinite" }}>
                          scoring…
                        </span>
                      )}
                      <span style={{ marginLeft: "auto", display: "flex", gap: 6 }} onClick={e => e.stopPropagation()}>
                        <Btn small danger onClick={() => onDismissAlert(a.id)}>
                          ✕
                        </Btn>
                      </span>
                    </div>
                    <div style={{ fontSize: 11, fontWeight: 500, marginBottom: 2 }}>{a.location}</div>
                    <div className="mono" style={{ fontSize: 9, color: "var(--text-3)", marginBottom: 3 }}>
                      {a.route}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-2)", lineHeight: 1.5 }}>{a.summary}</div>
                    {expanded && ml && (
                      <div
                        onClick={e => e.stopPropagation()}
                        style={{
                          marginTop: 8,
                          padding: "8px 10px",
                          background: "var(--panel-2)",
                          borderRadius: 4,
                          border: "1px solid var(--border)",
                        }}
                      >
                        <div style={{ fontSize: 8, fontWeight: 700, color: "var(--primary)", letterSpacing: "0.1em", marginBottom: 2 }}>
                          FEATURE IMPORTANCE
                        </div>
                        <FeatureChart features={ml.features} />
                      </div>
                    )}
                  </div>
                )
              })
            : routes.map(r => (
                <div
                  key={r.id}
                  style={{
                    padding: "8px 14px",
                    borderBottom: "1px solid var(--border)",
                    display: "grid",
                    gridTemplateColumns: "1fr auto",
                    gap: 8,
                    alignItems: "center",
                  }}
                >
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                      <span style={{ fontSize: 11, fontWeight: 600 }}>{r.name}</span>
                      <span style={{ fontSize: 9, color: "var(--text-3)", fontStyle: "italic" }}>{r.district}, {r.state}</span>
                    </div>
                    <RiskBar score={r.score} sev={r.risk} />
                    <div className="mono" style={{ fontSize: 9, color: "var(--text-3)", marginTop: 3 }}>
                      {r.type} · {r.isRemote ? "remote" : "accessible"}
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 5, alignItems: "flex-end" }}>
                    <span
                      className="mono"
                      style={{
                        fontSize: 8,
                        letterSpacing: "0.1em",
                        fontWeight: 700,
                        padding: "2px 6px",
                        background: SEV_BG[r.risk],
                        color: SEV_COLOR[r.risk],
                        borderRadius: 2,
                      }}
                    >
                      {r.risk.toUpperCase()}
                    </span>
                    <button
                      onClick={() => onToggleWatch(r.id)}
                      style={{
                        fontSize: 9,
                        fontFamily: "DM Mono, monospace",
                        padding: "2px 7px",
                        background: r.watched ? "rgba(54,199,165,0.12)" : "transparent",
                        color: r.watched ? "var(--primary)" : "var(--text-3)",
                        border: `1px solid ${r.watched ? "var(--primary)40" : "var(--border)"}`,
                        borderRadius: 3,
                        cursor: "pointer",
                      }}
                    >
                      {r.watched ? "★ WATCHING" : "☆ WATCH"}
                    </button>
                  </div>
                </div>
              ))}
        </div>
      </div>

      {/* Right: AI Redistribution + ML Panel */}
      <div style={{ width: 300, borderLeft: "1px solid var(--border)", display: "flex", flexDirection: "column", flexShrink: 0, background: "var(--panel)" }}>
        <div style={{ padding: "16px 16px 12px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>
            AI Redistribution
          </span>
          <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "#4fd9b0" }}>
            {reroutes.filter(r => !r.applied && !r.dismissed).length} suggestions
          </span>
        </div>
        <div style={{ padding: "12px 16px 0" }}>
          <button
            onClick={onGenerateReroutes}
            disabled={generatingReroutes}
            style={{
              width: "100%",
              padding: "9px 10px",
              fontSize: 12,
              fontFamily: "var(--font-ui)",
              fontWeight: 600,
              background: "var(--primary-dim)",
              color: "var(--primary)",
              border: "1px solid rgba(54,199,165,0.3)",
              borderRadius: 8,
              cursor: generatingReroutes ? "default" : "pointer",
              opacity: generatingReroutes ? 0.6 : 1,
            }}
          >
            {generatingReroutes ? "Generating…" : "⟳ Generate suggestions"}
          </button>
        </div>

        <div style={{ flex: 1, overflowY: "auto", paddingTop: 12 }}>
          {reroutes
            .filter(r => !r.dismissed)
            .map(rr => (
              <div
                key={rr.id}
                className="glass-card"
                style={{
                  margin: "0 16px 12px",
                  padding: "12px 14px",
                  borderLeft: "2px solid #4fd9b0",
                  borderRadius: 10,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                  <span style={{ fontSize: 11, color: "#4fd9b0", fontWeight: 600 }}>
                    {rr.applied ? "✓ Applied" : "AI suggestion"}
                  </span>
                  <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "#4fd9b0" }}>
                    {rr.confidence}% conf.
                  </span>
                </div>
                <div className="mono" style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>
                  {rr.resourceType} · {rr.quantity} units
                </div>
                <div className="mono" style={{ fontSize: 12, color: "var(--text)", fontWeight: 500, marginBottom: 6 }}>
                  {rr.fromPhc} → {rr.toPhc}
                </div>
                <div style={{ display: "flex", gap: 10, marginBottom: 8 }}>
                  <span className="mono" style={{ fontSize: 11, color: "#f5b942" }}>
                    +{rr.extraHours}h
                  </span>
                  <span className="mono" style={{ fontSize: 11, color: "#f5b942" }}>
                    {rr.extraCost}
                  </span>
                </div>
                <div style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.5, marginBottom: rr.applied ? 0 : 12 }}>{rr.reason}</div>
                {!rr.applied && (
                  <div style={{ display: "flex", gap: 8 }}>
                    <Btn small onClick={() => onApplyReroute(rr.id)}>
                      Apply
                    </Btn>
                    <Btn small danger onClick={() => onDismissReroute(rr.id)}>
                      Dismiss
                    </Btn>
                  </div>
                )}
              </div>
            ))}
          {reroutes.filter(r => !r.dismissed).length === 0 && (
            <div style={{ padding: 24, textAlign: "center", fontSize: 12, color: "var(--text-3)" }}>All suggestions processed.</div>
          )}
        </div>

        {/* ML Risk Engine */}
        <div style={{ borderTop: "1px solid var(--border)", padding: "14px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 10 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>
              ML Risk Engine
            </span>
            {mlRunning ? (
              <span className="mono" style={{ fontSize: 8, color: "var(--primary)", animation: "blink 0.9s infinite" }}>
                ◉ RUNNING
              </span>
            ) : Object.keys(mlScores).length > 0 ? (
              <span className="mono" style={{ fontSize: 8, color: "#4fd9b0" }}>
                ✓ SCORED
              </span>
            ) : null}
          </div>
          {alerts
            .filter(a => !a.dismissed)
            .slice(0, 4)
            .map(a => {
              const ml = mlScores[a.id]
              return (
                <div key={a.id} style={{ marginBottom: 9 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                    <span style={{ fontSize: 9, color: "var(--text-2)" }}>
                      {a.type} · {a.location.split(",")[0]}
                    </span>
                    <span className="mono" style={{ fontSize: 9, color: SEV_COLOR[a.severity] }}>
                      {ml ? (
                        <>
                          {ml.score}
                          <span style={{ color: "var(--text-3)", fontSize: 8, marginLeft: 3 }}>{ml.confidence}%</span>
                        </>
                      ) : (
                        "—"
                      )}
                    </span>
                  </div>
                  {ml ? <RiskBar score={ml.score} sev={a.severity} /> : <ShimmerBar />}
                </div>
              )
            })}
        </div>

        {/* Groq AI Analysis */}
        <div style={{ borderTop: "1px solid var(--border)", padding: "10px 14px" }}>
          {aiAnalysis ? (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 7 }}>
                <span style={{ fontSize: 9, fontWeight: 700, color: "#4fd9b0", letterSpacing: "0.08em" }}>✦ GROQ AI · LLAMA 3.3</span>
                <button
                  onClick={onGenerateAI}
                  style={{
                    marginLeft: "auto",
                    fontSize: 8,
                    fontFamily: "DM Mono, monospace",
                    padding: "1px 6px",
                    background: "transparent",
                    color: "var(--text-3)",
                    border: "1px solid var(--border)",
                    borderRadius: 2,
                    cursor: "pointer",
                  }}
                >
                  REFRESH
                </button>
              </div>
              <div style={{ fontSize: 10, color: "var(--text-2)", lineHeight: 1.7 }}>{aiAnalysis}</div>
            </div>
          ) : (
            <>
              <Btn small onClick={onGenerateAI}>
                {aiLoading ? "✦ ANALYZING…" : "✦ AI SITUATION ANALYSIS"}
              </Btn>
              {aiLoading && (
                <div style={{ marginTop: 8 }}>
                  {[88, 68, 50].map((w, i) => (
                    <div
                      key={i}
                      style={{
                        height: 2,
                        borderRadius: 1,
                        marginBottom: 5,
                        background: "linear-gradient(90deg, #1c2530 0%, #242e3b 50%, #1c2530 100%)",
                        backgroundSize: "200% 100%",
                        animation: "shimmer 1.4s ease infinite",
                        width: `${w}%`,
                      }}
                    />
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        <div style={{ borderTop: "1px solid var(--border)", padding: "5px 14px" }}>
          <span className="mono" style={{ fontSize: 7, color: "var(--text-3)" }}>
            sigmoid risk · z-score anomaly · LR forecast · groq llama-3.3
          </span>
        </div>
      </div>
    </div>
  )
}

// ─── Root App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState<NavPage>("dashboard")
  const [activeAlert, setActiveAlert] = useState<number | null>(1)
  const [alerts, setAlerts] = useState<AlertEvent[]>([])
  const [routes, setRoutes] = useState<PHCEntry[]>([])
  const [reroutes, setReroutes] = useState<RedistributionEntry[]>([])
  const [stocks, setStocks] = useState<ResourceStockView[]>([])
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError, setDataError] = useState<string | null>(null)
  const [currentTime, setCurrentTime] = useState("")
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchVal, setSearchVal] = useState("")

  // ML state
  const [mlScores, setMlScores] = useState<Record<number, MLScore>>({})
  const [mlRunning, setMlRunning] = useState(false)
  const [expandedMLId, setExpandedMLId] = useState<number | null>(null)

  // Reroute generation state
  const [generatingReroutes, setGeneratingReroutes] = useState(false)
  const [generatingPhcId, setGeneratingPhcId] = useState<string | null>(null)

  // Sidebar collapse state
  const [navCollapsed, setNavCollapsed] = useState(false)

  // Gemini AI state
  const [aiAnalysis, setAiAnalysis] = useState("")
  const [aiLoading, setAiLoading] = useState(false)

  useEffect(() => {
    const update = () => setCurrentTime(new Date().toLocaleTimeString("en-IN", { hour12: false }))
    update()
    const id = setInterval(update, 1000)
    return () => clearInterval(id)
  }, [])

  // Load real data from the backend on mount, replacing the old hardcoded mocks.
  useEffect(() => {
    let cancelled = false
    fetchDashboardData()
      .then(data => {
        if (cancelled) return
        setAlerts(data.alerts)
        setRoutes(data.phcs)
        setReroutes(data.redistributions)
        setStocks(data.stocks)
        setDataError(null)
      })
      .catch(err => {
        if (!cancelled) setDataError(err instanceof Error ? err.message : "Failed to load data")
      })
      .finally(() => {
        if (!cancelled) setDataLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // ML scoring pipeline — runs whenever active alerts change
  useEffect(() => {
    const live = alerts.filter(a => !a.dismissed)
    if (live.length === 0) {
      setMlScores({})
      return
    }

    setMlRunning(true)
    setMlScores({})

    // Stagger computation to simulate pipeline stages
    const timers: ReturnType<typeof setTimeout>[] = []
    live.forEach((alert, i) => {
      const t = setTimeout(() => {
        const score = computeRiskScore(alert, routes)
        setMlScores(prev => ({ ...prev, [alert.id]: score }))
        if (i === live.length - 1) setMlRunning(false)
      }, 280 + i * 160)
      timers.push(t)
    })

    return () => timers.forEach(clearTimeout)
  }, [alerts, routes])

  // Simulate alert age ticking (re-triggers ML scoring via alert state change)
  useEffect(() => {
    const interval = setInterval(() => {
      setAlerts(prev => prev.map(a => (a.dismissed ? a : { ...a, ageMin: a.ageMin + 1 })))
    }, 60_000)
    return () => clearInterval(interval)
  }, [])

  async function generateAI() {
    if (aiLoading) return
    setAiLoading(true)
    const live = alerts.filter(a => !a.dismissed)
    const ranked = Object.entries(mlScores).sort((a, b) => b[1].score - a[1].score)
    const topEntry = ranked[0]
    const topAlert = topEntry ? live.find(a => a.id === Number(topEntry[0])) : live[0]

    try {
      const text = await callRiskAnalysis({
        alerts: live.map(a => ({ severity: a.severity, type: a.type, location: a.location, summary: a.summary })),
        top_location: topAlert && topEntry ? topAlert.location : undefined,
        top_score: topAlert && topEntry ? topEntry[1].score : undefined,
        top_confidence: topAlert && topEntry ? topEntry[1].confidence : undefined,
      })
      setAiAnalysis(text)
    } catch (e: any) {
      setAiAnalysis(`Error: ${e.message}`)
    }
    setAiLoading(false)
  }

  const dismissAlert = (id: number) => {
    setAlerts(prev => prev.map(a => (a.id === id ? { ...a, dismissed: true } : a)))
    dismissAlertApi(id).catch(err => console.error("Failed to dismiss alert on server:", err))
  }

  const notifyTeam = (id: number) => {
    const alert = alerts.find(a => a.id === id)
    if (!alert) return
    const message = `[${alert.severity.toUpperCase()}] ${alert.type} at ${alert.location}: ${alert.summary}`
    notifyTeamApi(id, message)
      .then(results => {
        const sent = results.filter(r => r.status === "sent" || r.status === "success").length
        window.alert(`Notified ${sent} of ${results.length} device(s).`)
        dismissAlert(id)
      })
      .catch(err => {
        console.error("Failed to notify team:", err)
        window.alert("Failed to send notification. Check console for details.")
      })
  }

  const toggleWatch = (id: string) => {
    setRoutes(prev => {
      const next = prev.map(r => (r.id === id ? { ...r, watched: !r.watched } : r))
      const updated = next.find(r => r.id === id)
      if (updated) {
        setPHCWatched(id, updated.watched)
      }
      return next
    })
  }

  const applyReroute = (id: number) => {
    setReroutes(prev => prev.map(r => (r.id === id ? { ...r, applied: true } : r)))
    applyRedistributionApi(id).catch(err => console.error("Failed to apply redistribution on server:", err))
  }

  const dismissReroute = (id: number) => {
    setReroutes(prev => prev.map(r => (r.id === id ? { ...r, dismissed: true } : r)))
    dismissRedistributionApi(id).catch(err => console.error("Failed to dismiss redistribution on server:", err))
  }

  const generateSuggestions = async () => {
    if (generatingReroutes) return
    setGeneratingReroutes(true)
    try {
      const targets = routes.filter(r => r.risk === "critical" || r.risk === "high")
      const results = await Promise.all(
        targets.map(r => generateRedistributionsApi(r.id).catch(err => {
          console.error(`Failed to generate redistributions for PHC ${r.id}:`, err)
          return [] as Awaited<ReturnType<typeof generateRedistributionsApi>>
        }))
      )
      const newReroutes = results.flat().map(rr => adaptRedistribution(rr))
      setReroutes(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = newReroutes.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
    } finally {
      setGeneratingReroutes(false)
    }
  }

  const generateForPhc = async (phcId: string) => {
    if (generatingPhcId === phcId) return
    setGeneratingPhcId(phcId)
    try {
      const results = await generateRedistributionsApi(phcId)
      const newReroutes = results.map(rr => adaptRedistribution(rr))
      setReroutes(prev => {
        const existingIds = new Set(prev.map(r => r.id))
        const deduped = newReroutes.filter(r => !existingIds.has(r.id))
        return [...prev, ...deduped]
      })
    } catch (err) {
      console.error(`Failed to generate redistributions for PHC ${phcId}:`, err)
    } finally {
      setGeneratingPhcId(null)
    }
  }

  const liveCount = alerts.filter(a => !a.dismissed).length
  const criticalCount = alerts.filter(a => !a.dismissed && a.severity === "critical").length

  const navItems: { icon: string; label: string; id: NavPage; badge?: number }[] = [
    { icon: "◈", label: "Dashboard", id: "dashboard" },
    { icon: "⬡", label: "Live Map", id: "map" },
    { icon: "◇", label: "Redistribution Planner", id: "planner" },
    { icon: "△", label: "Alerts", id: "alerts", badge: criticalCount },
    { icon: "□", label: "Analytics", id: "analytics" },
    { icon: "○", label: "Settings", id: "settings" },
  ]

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--bg)", overflow: "hidden" }}>
      {/* Top Bar */}
      <header
        style={{
          height: 44,
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          padding: "0 16px",
          gap: 12,
          flexShrink: 0,
          background: "var(--panel)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginRight: 4, cursor: "pointer" }} onClick={() => setPage("dashboard")}>
          <div
            style={{
              width: 28,
              height: 28,
              background: "linear-gradient(135deg, #36c7a5 0%, #1f8f76 100%)",
              borderRadius: 7,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span style={{ color: "#fff", fontSize: 13, fontWeight: 700, fontFamily: "var(--font-ui)" }}>P</span>
          </div>
          <span style={{ fontWeight: 700, fontSize: 17, letterSpacing: "-0.015em", color: "var(--text)" }}>PHC-Nexus</span>
          <span style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.08em", fontWeight: 500 }}>Supply Chain Intelligence</span>
        </div>

        <div style={{ width: 1, height: 20, background: "var(--border)" }} />

        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <LiveDot />
          <span className="mono" style={{ fontSize: 10, color: "#4fd9b0", fontWeight: 500 }}>
            LIVE
          </span>
        </div>

        {criticalCount > 0 && (
          <button
            onClick={() => setPage("alerts")}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              background: "rgba(239,91,104,0.08)",
              border: "1px solid rgba(239,91,104,0.25)",
              padding: "3px 9px",
              borderRadius: 4,
              cursor: "pointer",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: "#ef5b68",
                display: "inline-block",
                animation: "blink 1.2s ease infinite",
              }}
            />
            <span className="mono" style={{ fontSize: 10, color: "#ef5b68" }}>
              {liveCount} ACTIVE ALERTS
            </span>
          </button>
        )}

        {mlRunning && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              padding: "3px 8px",
              background: "rgba(54,199,165,0.05)",
              border: "1px solid rgba(54,199,165,0.18)",
              borderRadius: 4,
            }}
          >
            <span style={{ fontSize: 8, color: "var(--primary)", animation: "blink 0.7s infinite" }}>◉</span>
            <span className="mono" style={{ fontSize: 9, color: "var(--primary)" }}>
              ML SCORING
            </span>
          </div>
        )}

        <div style={{ marginLeft: 4, position: "relative" }}>
          {searchOpen ? (
            <input
              autoFocus
              value={searchVal}
              onChange={e => setSearchVal(e.target.value)}
              onBlur={() => {
                setSearchOpen(false)
                setSearchVal("")
              }}
              placeholder="Search routes, ports, alerts…"
              style={{
                width: 220,
                padding: "4px 10px",
                background: "var(--panel-2)",
                border: "1px solid var(--primary)50",
                borderRadius: 4,
                color: "var(--text)",
                fontSize: 11,
                fontFamily: "var(--font-ui)",
                outline: "none",
              }}
            />
          ) : (
            <button
              onClick={() => setSearchOpen(true)}
              style={{
                padding: "4px 10px",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: 4,
                color: "var(--text-2)",
                fontSize: 12,
                fontFamily: "var(--font-ui)",
                cursor: "pointer",
              }}
            >
              ⌕ Search
            </button>
          )}
        </div>

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          <span className="mono" style={{ fontSize: 10, color: "var(--text-3)" }}>
            IST {currentTime}
          </span>
          <button
            style={{
              padding: "5px 12px",
              background: "transparent",
              border: "1px solid var(--border)",
              borderRadius: 6,
              color: "var(--text-2)",
              fontSize: 12,
              fontFamily: "var(--font-ui)",
              cursor: "pointer",
            }}
            onClick={() => setPage("settings")}
          >
            ⚙ Config
          </button>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "3px 10px",
              background: "var(--panel-2)",
              borderRadius: 4,
              border: "1px solid var(--border)",
            }}
          >
            <span style={{ fontSize: 11, color: "var(--text-2)" }}>Aparna Dhiraj</span>
            <div
              style={{
                width: 22,
                height: 22,
                borderRadius: "50%",
                background: "linear-gradient(135deg, #36c7a5, #1f8f76)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span style={{ fontSize: 9, fontWeight: 700, color: "#fff" }}>AD</span>
            </div>
          </div>
        </div>
      </header>

      {(dataLoading || dataError) && (
        <div
          className="glass-card"
          style={{
            position: "fixed",
            bottom: 20,
            left: navCollapsed ? 72 : 188,
            zIndex: 50,
            padding: "8px 14px",
            fontSize: 12,
            color: dataError ? "#ef5b68" : "var(--primary)",
            borderColor: dataError ? "rgba(239,91,104,0.35)" : "rgba(54,199,165,0.3)",
          }}
        >
          {dataError ? `Backend error: ${dataError}` : "Loading live data…"}
        </div>
      )}

      {/* Body */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <nav
          style={{
            width: navCollapsed ? 52 : 168,
            borderRight: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
            padding: "16px 0",
            gap: 2,
            flexShrink: 0,
            background: "var(--panel)",
            transition: "width 0.15s",
          }}
        >
          <button
            onClick={() => setNavCollapsed(c => !c)}
            title={navCollapsed ? "Expand" : "Collapse"}
            style={{
              alignSelf: navCollapsed ? "center" : "flex-end",
              marginRight: navCollapsed ? 0 : 8,
              marginBottom: 6,
              width: 22,
              height: 22,
              borderRadius: 4,
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-3)",
              fontSize: 11,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {navCollapsed ? "»" : "«"}
          </button>
          {navItems.map(item => (
            <button
              key={item.id}
              title={item.label}
              onClick={() => setPage(item.id)}
              style={{
                position: "relative",
                width: navCollapsed ? 36 : "calc(100% - 16px)",
                margin: navCollapsed ? 0 : "0 8px",
                height: 38,
                borderRadius: 8,
                border: "none",
                borderLeft: !navCollapsed && page === item.id ? "2px solid var(--primary)" : "2px solid transparent",
                cursor: "pointer",
                background: page === item.id ? "var(--primary-dim)" : "transparent",
                color: page === item.id ? "var(--primary)" : "var(--text-2)",
                fontSize: 14,
                display: "flex",
                alignItems: "center",
                justifyContent: navCollapsed ? "center" : "flex-start",
                gap: 10,
                padding: navCollapsed ? 0 : "0 10px",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              <span>{item.icon}</span>
              {!navCollapsed && <span style={{ fontSize: 13, fontWeight: 500, fontFamily: "var(--font-ui)" }}>{item.label}</span>}
              {item.badge != null && item.badge > 0 && (
                <span
                  style={{
                    position: navCollapsed ? "absolute" : "static",
                    top: navCollapsed ? 4 : undefined,
                    right: navCollapsed ? 4 : undefined,
                    marginLeft: navCollapsed ? 0 : "auto",
                    width: 14,
                    height: 14,
                    borderRadius: "50%",
                    background: "#ef5b68",
                    fontSize: 8,
                    fontWeight: 700,
                    fontFamily: "DM Mono, monospace",
                    color: "#fff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {item.badge}
                </span>
              )}
            </button>
          ))}

          <div
            style={{
              marginTop: "auto",
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: navCollapsed ? "10px 0" : "10px 18px",
              justifyContent: navCollapsed ? "center" : "flex-start",
            }}
          >
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                background: "var(--success)",
                flexShrink: 0,
                boxShadow: "0 0 0 3px var(--success-dim)",
              }}
            />
            {!navCollapsed && (
              <div style={{ display: "flex", flexDirection: "column", lineHeight: 1.2 }}>
                <span style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 600, letterSpacing: "0.06em" }}>SYSTEM</span>
                <span style={{ fontSize: 11, color: "var(--text-2)", fontWeight: 500 }}>Operational</span>
              </div>
            )}
          </div>
        </nav>

        {page === "dashboard" && (
          <DashboardView
            alerts={alerts}
            routes={routes}
            reroutes={reroutes}
            activeAlert={activeAlert}
            setActiveAlert={setActiveAlert}
            onDismissAlert={dismissAlert}
            onToggleWatch={toggleWatch}
            onApplyReroute={applyReroute}
            onDismissReroute={dismissReroute}
            onGenerateReroutes={generateSuggestions}
            generatingReroutes={generatingReroutes}
            mlScores={mlScores}
            mlRunning={mlRunning}
            aiAnalysis={aiAnalysis}
            aiLoading={aiLoading}
            onGenerateAI={generateAI}
            expandedMLId={expandedMLId}
            setExpandedMLId={setExpandedMLId}
          />
        )}
        {page === "map" && <LiveMapPage phcs={routes} />}
        {page === "stock" && <StockPage stocks={stocks} phcs={routes} />}
        {page === "planner" && (
          <RoutePlannerPage
            phcs={routes}
            reroutes={reroutes}
            onGenerateForPhc={generateForPhc}
            generatingPhcId={generatingPhcId}
            onApplyReroute={applyReroute}
            onDismissReroute={dismissReroute}
          />
        )}
        {page === "alerts" && <AlertsPage alerts={alerts} onDismiss={dismissAlert} onNotify={notifyTeam} />}
        {page === "analytics" && <AnalyticsPage alerts={alerts} stocks={stocks} phcs={routes} reroutes={reroutes} />}
        {page === "settings" && <SettingsPage />}
      </div>
    </div>
  )
}