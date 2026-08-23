from sqlalchemy import Column, Integer, String, Float
from app.database import Base


class PHC(Base):
    """A Primary Health Centre — the core facility node in the network."""
    __tablename__ = "phcs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    state = Column(String)
    district = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    type = Column(String)  # "PHC" / "CHC" / "district_warehouse"
    is_remote = Column(Integer, default=0)  # 1 if hard to resupply (chokepoint equivalent)
