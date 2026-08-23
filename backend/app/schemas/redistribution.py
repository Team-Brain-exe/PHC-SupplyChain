from pydantic import BaseModel


class RedistributionBase(BaseModel):
    origin_phc_id: int | None = None
    resource_type: str
    from_phc: str
    to_phc: str
    quantity: int
    extra_hours: float
    extra_cost: float
    confidence: float
    reason: str
    applied: bool = False
    dismissed: bool = False


class RedistributionCreate(RedistributionBase):
    pass


class RedistributionOut(RedistributionBase):
    id: int

    class Config:
        from_attributes = True


class RedistributionUpdate(BaseModel):
    applied: bool | None = None
    dismissed: bool | None = None
