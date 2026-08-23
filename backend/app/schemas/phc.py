from pydantic import BaseModel


class PHCBase(BaseModel):
    name: str
    state: str
    district: str
    latitude: float
    longitude: float
    type: str
    is_remote: int = 0


class PHCCreate(PHCBase):
    pass


class PHCOut(PHCBase):
    id: int

    class Config:
        from_attributes = True
