"""
Connects PHC/stock data to the ML model — the glue between the database
and app.ml.model.predict().
"""

from sqlalchemy.orm import Session

from app.models.phc import PHC
from app.models.resource_stock import ResourceStock
from app.models.health_alert import HealthAlert
from app.ml.model import predict

RISK_MEDIUM = 50  # threshold above which a PHC is considered redistribution-worthy


def score_phc(db: Session, phc: PHC) -> dict:
    stock = (
        db.query(ResourceStock)
        .filter(ResourceStock.phc_id == phc.id)
        .order_by(ResourceStock.updated_at.desc())
        .first()
    )

    if not stock:
        return {"score": 0.0, "confidence": 0.0, "features": {}, "contributing_alerts": []}

    stock_pct = (stock.stock_units / stock.stock_capacity * 100) if stock.stock_capacity else 0
    bed_occupancy_pct = (stock.bed_occupied / stock.bed_total * 100) if stock.bed_total else 0
    staff_attendance_pct = (stock.staff_present / stock.staff_total * 100) if stock.staff_total else 0

    district_alerts = (
        db.query(HealthAlert)
        .filter(HealthAlert.district == phc.district, HealthAlert.dismissed == False)  # noqa: E712
        .all()
    )

    features = {
        "days_of_stock_remaining": stock.days_of_stock_remaining,
        "stock_pct": stock_pct,
        "bed_occupancy_pct": bed_occupancy_pct,
        "staff_attendance_pct": staff_attendance_pct,
        "patient_footfall_daily": stock.patient_footfall_daily,
        "is_remote": phc.is_remote,
        "demand_trend_pct": 10,  # placeholder — would come from historical footfall comparison
        "district_alert_density": len(district_alerts),
    }

    result = predict(features)
    result["contributing_alerts"] = [a.id for a in district_alerts]
    return result
