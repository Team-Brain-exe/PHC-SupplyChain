from pydantic import BaseModel
from datetime import datetime


class ResourceStockBase(BaseModel):
    phc_id: int
    medicine_category: str
    stock_units: int
    stock_capacity: int
    days_of_stock_remaining: float
    bed_total: int
    bed_occupied: int
    staff_total: int
    staff_present: int
    patient_footfall_daily: int


class ResourceStockCreate(ResourceStockBase):
    pass


class ResourceStockOut(ResourceStockBase):
    id: int
    updated_at: datetime

    class Config:
        from_attributes = True


class ResourceStockUpdate(BaseModel):
    stock_units: int | None = None
    bed_occupied: int | None = None
    staff_present: int | None = None
    patient_footfall_daily: int | None = None
