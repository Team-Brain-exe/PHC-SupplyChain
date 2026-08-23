// Talks to the PHC-Nexus backend (FastAPI) and adapts its response
// shapes into the types App.tsx expects. This is the one place that
// should know both shapes — nothing else in the frontend should.

export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

// ─── Backend response shapes (mirrors backend/app/schemas/*.py) ──────────────

export type BackendPHC = {
  id: number;
  name: string;
  state: string;
  district: string;
  latitude: number;
  longitude: number;
  type: string;
  is_remote: boolean;
};

export type BackendResourceStock = {
  id: number;
  phc_id: number;
  medicine_category: string;
  stock_units: number;
  stock_capacity: number;
  days_of_stock_remaining: number;
  bed_total: number;
  bed_occupied: number;
  staff_total: number;
  staff_present: number;
  patient_footfall_daily: number;
  updated_at: string;
};

export type BackendHealthAlert = {
  id: number;
  time: string;
  type: string;
  phc_name: string;
  district: string;
  severity: number; // 1-5
  summary: string;
  age_min: number;
  dismissed: boolean;
};

export type BackendRedistribution = {
  id: number;
  origin_phc_id: number;
  resource_type: string;
  from_phc: string;
  to_phc: string;
  quantity: number;
  extra_hours: number;
  extra_cost: number;
  confidence: number; // 0-1
  reason: string;
  applied: boolean;
  dismissed: boolean;
};

// GET /phcs/{id}/risk -> score_phc() in risk_scoring.py.
// NOTE: no stock record -> {score:0, confidence:0, features:{}, contributing_alerts:[]}
export type PHCRisk = {
  score: number; // 0-100
  confidence: number; // 0-1
  features: Record<string, number>;
  contributing_alerts: number[];
};

// ─── Frontend shapes (must match App.tsx's own types) ────────────────────────

type Severity = "critical" | "high" | "medium" | "low";

export type AlertEvent = {
  id: number;
  time: string;
  type: string;
  location: string; // phc_name
  route: string; // repurposed: district
  severity: Severity;
  summary: string;
  ageMin: number;
  dismissed?: boolean;
};

// Replaces old FrontendRoute. A PHC is a location, not a route.
export type PHC = {
  id: string;
  name: string;
  state: string;
  district: string;
  lat: number;
  lng: number;
  type: string;
  isRemote: boolean;
  risk: Severity;
  score: number;
  watched: boolean; // frontend-only, see WATCHED_STORAGE_KEY below
};

export type ResourceStockView = {
  id: number;
  phcId: number;
  category: string;
  units: number;
  capacity: number;
  daysRemaining: number;
  bedTotal: number;
  bedOccupied: number;
  staffTotal: number;
  staffPresent: number;
  footfallDaily: number;
  updatedAt: string;
};

// Replaces old FrontendReroute.
export type Redistribution = {
  id: number;
  originPhcId: number;
  resourceType: string;
  fromPhc: string;
  toPhc: string;
  quantity: number;
  extraHours: number;
  extraCost: string;
  confidence: number;
  reason: string;
  applied?: boolean;
  dismissed?: boolean;
};

// ─── Adapters ──────────────────────────────────────────────────────────────

function severityFromInt(n: number): Severity {
  if (n >= 5) return "critical";
  if (n >= 4) return "high";
  if (n >= 3) return "medium";
  return "low";
}

// score_phc() returns only a numeric score — no categorical band
// (unlike old Project44's routes.risk string). Bucket it here.
function riskFromScore(score: number): Severity {
  if (score >= 85) return "critical";
  if (score >= 65) return "high";
  if (score >= 40) return "medium";
  return "low";
}

export function adaptAlert(a: BackendHealthAlert): AlertEvent {
  return {
    id: a.id,
    time: a.time,
    type: a.type.toUpperCase(),
    location: a.phc_name,
    route: a.district,
    severity: severityFromInt(a.severity),
    summary: a.summary,
    ageMin: a.age_min,
    dismissed: a.dismissed,
  };
}

export function adaptPHC(p: BackendPHC, risk: PHCRisk | null, watched: boolean): PHC {
  const score = risk?.score ?? 0;
  return {
    id: String(p.id),
    name: p.name,
    state: p.state,
    district: p.district,
    lat: p.latitude,
    lng: p.longitude,
    type: p.type,
    isRemote: p.is_remote,
    risk: riskFromScore(score),
    score: Math.round(score),
    watched,
  };
}

export function adaptStock(s: BackendResourceStock): ResourceStockView {
  return {
    id: s.id,
    phcId: s.phc_id,
    category: s.medicine_category,
    units: s.stock_units,
    capacity: s.stock_capacity,
    daysRemaining: s.days_of_stock_remaining,
    bedTotal: s.bed_total,
    bedOccupied: s.bed_occupied,
    staffTotal: s.staff_total,
    staffPresent: s.staff_present,
    footfallDaily: s.patient_footfall_daily,
    updatedAt: s.updated_at,
  };
}

export function adaptRedistribution(r: BackendRedistribution): Redistribution {
  return {
    id: r.id,
    originPhcId: r.origin_phc_id,
    resourceType: r.resource_type,
    fromPhc: r.from_phc,
    toPhc: r.to_phc,
    quantity: r.quantity,
    extraHours: r.extra_hours,
    extraCost: `+₹${r.extra_cost.toLocaleString()}`,
    confidence: Math.round(r.confidence * 100),
    reason: r.reason,
    applied: r.applied,
    dismissed: r.dismissed,
  };
}

// ─── Fetchers ──────────────────────────────────────────────────────────────

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function patchJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function postJSON<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

// ─── Watched PHCs (frontend-only — no backend field/route for this yet) ─────

const WATCHED_STORAGE_KEY = "phcnexus.watchedPhcIds";

function getWatchedIds(): Set<string> {
  try {
    const raw = localStorage.getItem(WATCHED_STORAGE_KEY);
    return new Set(raw ? (JSON.parse(raw) as string[]) : []);
  } catch {
    return new Set();
  }
}

function saveWatchedIds(ids: Set<string>) {
  localStorage.setItem(WATCHED_STORAGE_KEY, JSON.stringify([...ids]));
}

export function setPHCWatched(id: string, watched: boolean) {
  const ids = getWatchedIds();
  if (watched) ids.add(id);
  else ids.delete(id);
  saveWatchedIds(ids);
}

// ─── Dashboard aggregate fetch ───────────────────────────────────────────────

export async function fetchDashboardData() {
  const [rawPHCs, rawStocks, rawAlerts, rawRedistributions] = await Promise.all([
    getJSON<BackendPHC[]>("/phcs"),
    getJSON<BackendResourceStock[]>("/stocks"),
    getJSON<BackendHealthAlert[]>("/alerts"),
    getJSON<BackendRedistribution[]>("/redistributions"),
  ]);

  const watchedIds = getWatchedIds();

  const risks = await Promise.all(
    rawPHCs.map(p =>
      getJSON<PHCRisk>(`/phcs/${p.id}/risk`).catch(() => null)
    )
  );

  const phcs = rawPHCs.map((p, i) => adaptPHC(p, risks[i], watchedIds.has(String(p.id))));

  return {
    phcs,
    stocks: rawStocks.map(adaptStock),
    alerts: rawAlerts.map(adaptAlert),
    redistributions: rawRedistributions.map(adaptRedistribution),
  };
}

export function dismissAlertApi(id: number) {
  return patchJSON<BackendHealthAlert>(`/alerts/${id}`, { dismissed: true });
}

export function applyRedistributionApi(id: number) {
  return patchJSON<BackendRedistribution>(`/redistributions/${id}/apply`);
}

export function dismissRedistributionApi(id: number) {
  return patchJSON<BackendRedistribution>(`/redistributions/${id}/dismiss`);
}

export function generateRedistributionsApi(phcId: string) {
  return postJSON<BackendRedistribution[]>(`/redistributions/generate/${phcId}`);
}

// ─── Notifications (unchanged from Project44) ────────────────────────────────

export type BackendNotification = {
  id: number;
  alert_id: number | null;
  device_id: number | null;
  phone_number: string;
  message: string;
  status: string;
  detail: string | null;
};

export function notifyTeamApi(alertId: number, message: string) {
  return postJSON<BackendNotification[]>("/notifications/send", {
    alert_id: alertId,
    message,
  });
}