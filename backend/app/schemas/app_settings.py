from pydantic import BaseModel


class AppSettingsOut(BaseModel):
    id: int
    contact_name: str
    contact_email: str
    contact_phone: str
    notif_email: bool
    notif_whatsapp: bool
    notif_sms: bool
    alert_threshold: str

    class Config:
        from_attributes = True


class AppSettingsUpdate(BaseModel):
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    notif_email: bool | None = None
    notif_whatsapp: bool | None = None
    notif_sms: bool | None = None
    alert_threshold: str | None = None
