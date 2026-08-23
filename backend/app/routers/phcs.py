from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.phc import PHC
from app.schemas.phc import PHCOut, PHCCreate
from app.services.risk_scoring import score_phc

router = APIRouter(prefix="/phcs", tags=["phcs"])


@router.get("", response_model=list[PHCOut])
def list_phcs(db: Session = Depends(get_db)):
    return db.query(PHC).all()


@router.get("/{phc_id}", response_model=PHCOut)
def get_phc(phc_id: int, db: Session = Depends(get_db)):
    phc = db.query(PHC).filter(PHC.id == phc_id).first()
    if not phc:
        raise HTTPException(status_code=404, detail="PHC not found")
    return phc


@router.post("", response_model=PHCOut)
def create_phc(phc: PHCCreate, db: Session = Depends(get_db)):
    db_phc = PHC(**phc.model_dump())
    db.add(db_phc)
    db.commit()
    db.refresh(db_phc)
    return db_phc


@router.get("/{phc_id}/risk")
def get_phc_risk(phc_id: int, db: Session = Depends(get_db)):
    phc = db.query(PHC).filter(PHC.id == phc_id).first()
    if not phc:
        raise HTTPException(status_code=404, detail="PHC not found")
    return score_phc(db, phc)
