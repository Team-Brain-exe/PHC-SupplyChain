from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.health_alert import HealthAlert
from app.schemas.health_alert import HealthAlertOut, HealthAlertCreate, HealthAlertUpdate

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[HealthAlertOut])
def list_alerts(db: Session = Depends(get_db)):
    return db.query(HealthAlert).all()


@router.get("/{alert_id}", response_model=HealthAlertOut)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(HealthAlert).filter(HealthAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.post("", response_model=HealthAlertOut)
def create_alert(alert: HealthAlertCreate, db: Session = Depends(get_db)):
    db_alert = HealthAlert(**alert.model_dump())
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert


@router.patch("/{alert_id}", response_model=HealthAlertOut)
def update_alert(alert_id: int, update: HealthAlertUpdate, db: Session = Depends(get_db)):
    alert = db.query(HealthAlert).filter(HealthAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(alert, field, value)
    db.commit()
    db.refresh(alert)
    return alert
