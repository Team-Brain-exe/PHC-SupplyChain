"""
Generates real, model-written reasoning for redistribution suggestions
using Google Gemini (via AI Studio's free API), grounded in the actual
stock-out risk score and PHC context.

Falls back to a plain rule-based sentence if no API key is configured
or the API call fails, so the feature degrades gracefully.
"""

import google.generativeai as genai
from app.config import settings

_configured = False


def _ensure_configured() -> bool:
    global _configured
    if not settings.gemini_api_key:
        return False
    if not _configured:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True
    return True


def generate_ai_reason(from_phc, to_phc, risk_result: dict, resource_type: str) -> str | None:
    if not _ensure_configured():
        return None

    prompt = f"""You are a public health supply chain analyst in India. Write ONE concise sentence (under 25 words) explaining why {resource_type} should be redistributed from {to_phc.name} ({to_phc.district}, {to_phc.state}) to {from_phc.name} ({from_phc.district}, {from_phc.state}).

Stock-out risk score at {from_phc.name}: {risk_result['score']}/100 (confidence {risk_result['confidence']})
Contributing factors: {risk_result['features']}

Reply with ONLY the one sentence, no preamble, no quotes."""

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        text = response.text.strip()
        return text if text else None
    except Exception as e:
        print(f"[ai_redistribution] Gemini call failed: {e}")
        return None


def generate_district_briefing(district: str, alerts: list, avg_risk: float) -> str | None:
    """Used for the 'AI Situation Analysis' panel — a district-level summary."""
    if not _ensure_configured():
        return None

    alert_lines = "\n".join(
        f"- [{a.severity}/5] {a.type} at {a.phc_name}: {a.summary}" for a in alerts
    ) or "No active alerts."

    prompt = f"""You are a public health supply chain analyst. Write a 2-3 sentence operational briefing for district health officers in {district}, India.

Average stock-out risk across active alerts: {avg_risk:.0f}/100

Active alerts:
{alert_lines}

Be specific and direct. No preamble."""

    try:
        model = genai.GenerativeModel("gemini-3.6-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ai_redistribution] Gemini briefing failed: {e}")
        return None
