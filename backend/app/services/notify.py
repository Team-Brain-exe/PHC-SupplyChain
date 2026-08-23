import requests
from app.config import settings

FAST2SMS_URL = "https://www.fast2sms.com/dev/bulkV2"


def _clean_number(phone_number: str) -> str:
    """Fast2SMS wants bare 10-digit Indian numbers, no +91 / country code."""
    digits = "".join(c for c in phone_number if c.isdigit())
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def send_sms(phone_number: str, message: str) -> dict:
    """
    Sends an SMS via Fast2SMS. Returns a dict with at least:
      { "status": "sent" | "failed", "detail": <raw response or error text> }
    Never raises -- callers (the notifications router) log the result either way.

    DEMO MODE: when settings.demo_sms_mode is True (default), this skips the
    real Fast2SMS call entirely and returns a simulated "sent" result, tagged
    "[DEMO]" in the detail so it's obvious in the notification log that no
    real text message went out. Flip DEMO_SMS_MODE=false in the environment
    once the Fast2SMS account is reactivated to send for real again.
    """
    number = _clean_number(phone_number)
    if len(number) != 10:
        return {"status": "failed", "detail": f"invalid phone number: {phone_number}"}

    if settings.demo_sms_mode:
        return {
            "status": "sent",
            "detail": f"[DEMO] Simulated SMS to {number} -- Fast2SMS not called (demo mode on).",
        }

    if not settings.fast2sms_api_key:
        return {"status": "failed", "detail": "FAST2SMS_API_KEY not configured"}

    payload = {
        "route": "q",
        "message": message,
        "language": "english",
        "flash": 0,
        "numbers": number,
    }
    headers = {
        "authorization": settings.fast2sms_api_key,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        response = requests.post(FAST2SMS_URL, data=payload, headers=headers, timeout=10)
        body = response.json()
        if response.status_code == 200 and body.get("return") is True:
            return {"status": "sent", "detail": str(body)}
        return {"status": "failed", "detail": str(body)}
    except requests.RequestException as exc:
        return {"status": "failed", "detail": str(exc)}
