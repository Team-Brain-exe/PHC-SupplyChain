from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from app.database import Base


class Redistribution(Base):
    """A recommended resource transfer from a surplus PHC to a deficit PHC."""
    __tablename__ = "redistributions"

    id = Column(Integer, primary_key=True, index=True)
    origin_phc_id = Column(Integer, ForeignKey("phcs.id"), nullable=True)
    resource_type = Column(String)  # matches medicine_category, or "beds" / "staff"
    from_phc = Column(String)
    to_phc = Column(String)
    quantity = Column(Integer)
    extra_hours = Column(Float)
    extra_cost = Column(Float)
    confidence = Column(Float)
    reason = Column(String)
    applied = Column(Boolean, default=False)
    dismissed = Column(Boolean, default=False)
