"""
Generates redistribution suggestions for PHCs currently at high stock-out
risk — finds nearby PHCs with surplus of the same resource, estimates the
transfer cost/time, and (if Gemini is configured) generates a real
AI-written reason grounded in the actual risk data.
"""

from sqlalchemy.orm import Session

from app.models.phc import PHC
from app.models.resource_stock import ResourceStock
from app.models.redistribution import Redistribution
from app.services.risk_scoring import score_phc, RISK_MEDIUM
from app.services.ai_redistribution import generate_ai_reason

# Estimated transfer time/cost by distance tier (rough, not derived from a dataset)
TRANSFER_PROFILES = {
    "same_district": {"extra_hours": 3.0, "extra_cost": 4_000.0},
    "cross_district": {"extra_hours": 8.0, "extra_cost": 12_000.0},
    "cross_state": {"extra_hours": 24.0, "extra_cost": 35_000.0},
}

FALLBACK_REASON = "Elevated stock-out risk detected based on current inventory and demand trends."


def _transfer_tier(from_phc: PHC, to_phc: PHC) -> str:
    if from_phc.district == to_phc.district:
        return "same_district"
    if from_phc.state == to_phc.state:
        return "cross_district"
    return "cross_state"


def _surplus_phcs(db: Session, deficit_phc: PHC, medicine_category: str) -> list[tuple[PHC, ResourceStock]]:
    """PHCs with meaningfully higher stock_pct of the same resource, same state, excluding self."""
    candidates = (
        db.query(PHC, ResourceStock)
        .join(ResourceStock, ResourceStock.phc_id == PHC.id)
        .filter(
            PHC.id != deficit_phc.id,
            PHC.state == deficit_phc.state,
            ResourceStock.medicine_category == medicine_category,
        )
        .all()
    )
    surplus = [
        (phc, stock) for phc, stock in candidates
        if stock.stock_capacity and (stock.stock_units / stock.stock_capacity) > 0.5
    ]
    surplus.sort(key=lambda pair: pair[1].stock_units, reverse=True)
    return surplus[:3]


def generate_redistribution_candidates(db: Session, phc: PHC) -> list[dict]:
    risk_result = score_phc(db, phc)
    if risk_result["score"] < RISK_MEDIUM:
        return []

    stock = (
        db.query(ResourceStock)
        .filter(ResourceStock.phc_id == phc.id)
        .order_by(ResourceStock.updated_at.desc())
        .first()
    )
    if not stock:
        return []

    candidates = []
    for surplus_phc, surplus_stock in _surplus_phcs(db, phc, stock.medicine_category):
        tier = _transfer_tier(phc, surplus_phc)
        profile = TRANSFER_PROFILES[tier]

        ai_reason = generate_ai_reason(phc, surplus_phc, risk_result, stock.medicine_category)
        reason = ai_reason if ai_reason else FALLBACK_REASON

        transfer_qty = min(
            surplus_stock.stock_units - surplus_stock.stock_capacity // 2,
            stock.stock_capacity - stock.stock_units,
        )
        transfer_qty = max(transfer_qty, 0)

        candidates.append(
            {
                "origin_phc_id": phc.id,
                "resource_type": stock.medicine_category,
                "from_phc": surplus_phc.name,
                "to_phc": phc.name,
                "quantity": transfer_qty,
                "extra_hours": profile["extra_hours"],
                "extra_cost": profile["extra_cost"],
                "confidence": risk_result["confidence"],
                "reason": reason,
                "applied": False,
                "dismissed": False,
            }
        )

    candidates.sort(key=lambda c: c["extra_hours"])
    return candidates


def create_redistribution_suggestions(db: Session, phc: PHC) -> list[Redistribution]:
    candidates = generate_redistribution_candidates(db, phc)
    saved = []
    for c in candidates:
        r = Redistribution(**c)
        db.add(r)
        db.commit()
        db.refresh(r)
        saved.append(r)
    return saved
