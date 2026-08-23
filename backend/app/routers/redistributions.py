from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.redistribution import Redistribution
from app.models.phc import PHC
from app.schemas.redistribution import RedistributionOut, RedistributionCreate, RedistributionUpdate

router = APIRouter(prefix="/redistributions", tags=["redistributions"])


@router.get("", response_model=list[RedistributionOut])
def list_redistributions(db: Session = Depends(get_db)):
    return db.query(Redistribution).all()


@router.post("", response_model=RedistributionOut)
def create_redistribution(r: RedistributionCreate, db: Session = Depends(get_db)):
    db_r = Redistribution(**r.model_dump())
    db.add(db_r)
    db.commit()
    db.refresh(db_r)
    return db_r


@router.patch("/{r_id}/apply", response_model=RedistributionOut)
def apply_redistribution(r_id: int, db: Session = Depends(get_db)):
    r = db.query(Redistribution).filter(Redistribution.id == r_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Redistribution not found")
    r.applied = True
    db.commit()
    db.refresh(r)
    return r


@router.patch("/{r_id}/dismiss", response_model=RedistributionOut)
def dismiss_redistribution(r_id: int, db: Session = Depends(get_db)):
    r = db.query(Redistribution).filter(Redistribution.id == r_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Redistribution not found")
    r.dismissed = True
    db.commit()
    db.refresh(r)
    return r


@router.post("/generate/{phc_id}", response_model=list[RedistributionOut])
def generate_redistributions(phc_id: int, db: Session = Depends(get_db)):
    from app.services.redistribution_engine import create_redistribution_suggestions
    phc = db.query(PHC).filter(PHC.id == phc_id).first()
    if not phc:
        raise HTTPException(status_code=404, detail="PHC not found")
    return create_redistribution_suggestions(db, phc)
