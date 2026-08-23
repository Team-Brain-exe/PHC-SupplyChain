from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base


class AppSettings(Base):
    """
    Single persisted settings record. No multi-user auth exists yet, so this
    is a singleton row (id=1) rather than per-user — real persistence without
    faking a login system that isn't there.
    """
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, index=True)
    contact_name = Column(String, default="")
    contact_email = Column(String, default="")
    contact_phone = Column(String, default="")
    notif_email = Column(Boolean, default=True)
    notif_whatsapp = Column(Boolean, default=False)
    notif_sms = Column(Boolean, default=True)
    alert_threshold = Column(String, default="high")  # "critical" | "high" | "medium" | "low"
