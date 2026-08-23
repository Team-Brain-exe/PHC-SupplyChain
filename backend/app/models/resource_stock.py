from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from datetime import datetime
from app.database import Base


class ResourceStock(Base):
    """Point-in-time snapshot of a PHC's resource levels."""
    __tablename__ = "resource_stocks"

    id = Column(Integer, primary_key=True, index=True)
    phc_id = Column(Integer, ForeignKey("phcs.id"))
    medicine_category = Column(String)  # e.g. "antibiotics", "vaccines", "ORS", "insulin"
    stock_units = Column(Integer)
    stock_capacity = Column(Integer)
    days_of_stock_remaining = Column(Float)
    bed_total = Column(Integer)
    bed_occupied = Column(Integer)
    staff_total = Column(Integer)
    staff_present = Column(Integer)
    patient_footfall_daily = Column(Integer)
    updated_at = Column(DateTime, default=datetime.utcnow)
