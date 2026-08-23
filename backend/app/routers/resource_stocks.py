from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.resource_stock import ResourceStock
from app.schemas.resource_stock import ResourceStockOut, ResourceStockCreate, ResourceStockUpdate

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("", response_model=list[ResourceStockOut])
def list_stocks(db: Session = Depends(get_db)):
    return db.query(ResourceStock).all()


@router.get("/{stock_id}", response_model=ResourceStockOut)
def get_stock(stock_id: int, db: Session = Depends(get_db)):
    stock = db.query(ResourceStock).filter(ResourceStock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock record not found")
    return stock


@router.post("", response_model=ResourceStockOut)
def create_stock(stock: ResourceStockCreate, db: Session = Depends(get_db)):
    db_stock = ResourceStock(**stock.model_dump())
    db.add(db_stock)
    db.commit()
    db.refresh(db_stock)
    return db_stock


@router.patch("/{stock_id}", response_model=ResourceStockOut)
def update_stock(stock_id: int, update: ResourceStockUpdate, db: Session = Depends(get_db)):
    stock = db.query(ResourceStock).filter(ResourceStock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock record not found")
    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(stock, field, value)
    db.commit()
    db.refresh(stock)
    return stock
