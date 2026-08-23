from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.models import phc, resource_stock, health_alert, redistribution, user_device, notification  # noqa: F401
from app.routers import ml, health_alerts, phcs, resource_stocks, redistributions, user_devices, notifications, ai
from app.services.live_simulator import start_simulator, stop_simulator

app = FastAPI(title="PHC-Nexus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.on_event("startup")
def _on_startup():
    start_simulator()


@app.on_event("shutdown")
def _on_shutdown():
    stop_simulator()


app.include_router(ml.router)
app.include_router(health_alerts.router)
app.include_router(phcs.router)
app.include_router(resource_stocks.router)
app.include_router(redistributions.router)
app.include_router(user_devices.router)
app.include_router(notifications.router)
app.include_router(ai.router)