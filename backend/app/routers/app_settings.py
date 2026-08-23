import pathlib
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config import settings as app_config
from app.database import get_db
from app.models.app_settings import AppSettings
from app.models.phc import PHC
from app.models.resource_stock import ResourceStock
from app.schemas.app_settings import AppSettingsOut, AppSettingsUpdate
from app.services.live_simulator import scheduler as live_scheduler

router = APIRouter(prefix="/settings", tags=["settings"])


def _get_or_create(db: Session) -> AppSettings:
    settings = db.query(AppSettings).filter(AppSettings.id == 1).first()
    if not settings:
        settings = AppSettings(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


@router.get("", response_model=AppSettingsOut)
def get_settings(db: Session = Depends(get_db)):
    return _get_or_create(db)


@router.patch("", response_model=AppSettingsOut)
def update_settings(update: AppSettingsUpdate, db: Session = Depends(get_db)):
    settings = _get_or_create(db)
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(settings, field, value)
    db.commit()
    db.refresh(settings)
    return settings


@router.get("/system-status")
def get_system_status(db: Session = Depends(get_db)):
    """
    Reports genuine integration status — no fabricated 'X/6 states' strings.
    Each entry reflects something actually checkable in this codebase.
    """
    model_path = pathlib.Path(__file__).parent.parent / "ml" / "artifacts" / "model.pkl"
    model_exists = model_path.exists()

    total_states = db.query(PHC.state).distinct().count()

    cutoff = datetime.utcnow() - timedelta(seconds=90)
    recent_stock_phc_ids = (
        db.query(ResourceStock.phc_id)
        .filter(ResourceStock.updated_at >= cutoff)
        .distinct()
        .all()
    )
    recent_phc_ids = [r[0] for r in recent_stock_phc_ids]
    reporting_states = (
        db.query(PHC.state).filter(PHC.id.in_(recent_phc_ids)).distinct().count()
        if recent_phc_ids else 0
    )

    return {
        "sources": [
            {
                "name": "scikit-learn Risk Model",
                "status": "Active" if model_exists else "Not loaded",
                "detail": "Model artifact found" if model_exists else "model.pkl missing",
            },
            {
                "name": "Gemini AI (Redistribution Reasoning)",
                "status": "Active" if app_config.gemini_api_key else "Not configured",
                "detail": "API key set" if app_config.gemini_api_key else "No API key",
            },
            {
                "name": "Fast2SMS Notifications",
                "status": "Active" if app_config.fast2sms_api_key else "Not configured",
                "detail": "API key set" if app_config.fast2sms_api_key else "No API key",
            },
            {
                "name": "Live Telemetry Simulator",
                "status": "Active" if live_scheduler.running else "Stopped",
                "detail": f"{reporting_states}/{total_states} states reporting in last 90s",
            },
        ]
    }
