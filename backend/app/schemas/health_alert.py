from pydantic import BaseModel


class HealthAlertBase(BaseModel):
    time: str
    type: str
    phc_name: str
    district: str
    severity: int
    summary: str
    age_min: int
    dismissed: bool = False


class HealthAlertCreate(HealthAlertBase):
    pass


class HealthAlertOut(HealthAlertBase):
    id: int

    class Config:
        from_attributes = True


class HealthAlertUpdate(BaseModel):
    dismissed: bool | None = None
