from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base


class HealthAlert(Base):
    """A stock-out risk, staffing shortage, or capacity warning at a PHC."""
    __tablename__ = "health_alerts"

    id = Column(Integer, primary_key=True, index=True)
    time = Column(String)
    type = Column(String)  # "stockout" / "low_bed" / "staff_shortage" / "demand_spike"
    phc_name = Column(String)
    district = Column(String)
    severity = Column(Integer)  # 1-5
    summary = Column(String)
    age_min = Column(Integer)
    dismissed = Column(Boolean, default=False)
