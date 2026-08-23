"""
Server-side Gemini call backing the "AI Situation Analysis" panel on the
dashboard. Runs on the backend so the API key never reaches the
browser bundle.
"""

import google.generativeai as genai
from app.config import settings
from app.schemas.ai_analysis import RiskAnalysisRequest

_configured = False
_MODEL = "gemini-3.6-flash"


def _ensure_configured() -> bool:
    global _configured
    if not settings.gemini_api_key:
        return False
    if not _configured:
        genai.configure(api_key=settings.gemini_api_key)
        _configured = True
    return True


def generate_risk_analysis(req: RiskAnalysisRequest) -> str:
    if not _ensure_configured():
        return "No API key configured. Set GEMINI_API_KEY on the backend to enable live AI analysis."

    alert_lines = "\n".join(
        f"- [{a.severity.upper()}] {a.type} at {a.location}: {a.summary}"
        for a in req.alerts
    ) or "No active alerts."

    top_line = ""
    if req.top_location and req.top_score is not None:
        top_line = f'Highest stock-out risk: "{req.top_location}" scored {req.top_score}/100 ({req.top_confidence}% confidence).'

    prompt = f"""You are PHC-Nexus, a federated AI platform for national-scale health resource and supply chain management across India's Primary Health Centre network.

Active alerts ({len(req.alerts)} events):
{alert_lines}
{top_line}

Context: India operates a vast network of PHCs facing persistent stock-out risk, uneven bed availability, and staffing gaps, especially in remote districts.

Write 2-3 concise operational sentences for district health officers. Be specific and direct."""

    try:
        model = genai.GenerativeModel(_MODEL)
        response = model.generate_content(prompt)
        return (response.text or "No response.").strip()
    except Exception as e:
        return f"Error: {e}"
