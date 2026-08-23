"""
Simulated real-time telemetry layer.

Periodically nudges each PHC's resource_stocks row within realistic bounds,
so risk scores, alerts, and the frontend map reflect changing conditions
over time instead of a static seeded snapshot.

This does NOT connect to any real hospital system — it's a background
process that mutates state on a timer, standing in for a live data feed
until (or if) a real ingestion pipeline is built.
"""

import random
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler

from app.database import SessionLocal
from app.models.resource_stock import ResourceStock

STOCK_DRIFT_PCT = 0.05
BED_DRIFT = 1
STAFF_DRIFT = 1
FOOTFALL_DRIFT_PCT = 0.10

TICK_SECONDS = 15


def _jitter_stock_row(stock: ResourceStock) -> None:
    max_delta = max(1, int(stock.stock_units * STOCK_DRIFT_PCT))
    delta = random.randint(-max_delta, max_delta)
    stock.stock_units = max(0, min(stock.stock_capacity, stock.stock_units + delta))

    if stock.stock_capacity:
        stock_fraction = stock.stock_units / stock.stock_capacity
        baseline_days = stock.days_of_stock_remaining or 10
        stock.days_of_stock_remaining = round(max(0.0, baseline_days * stock_fraction * 1.05), 1)

    if stock.bed_total:
        bed_delta = random.randint(-BED_DRIFT, BED_DRIFT)
        stock.bed_occupied = max(0, min(stock.bed_total, stock.bed_occupied + bed_delta))

    if stock.staff_total:
        staff_delta = random.randint(-STAFF_DRIFT, STAFF_DRIFT)
        stock.staff_present = max(0, min(stock.staff_total, stock.staff_present + staff_delta))

    footfall_max_delta = max(1, int(stock.patient_footfall_daily * FOOTFALL_DRIFT_PCT))
    footfall_delta = random.randint(-footfall_max_delta, footfall_max_delta)
    stock.patient_footfall_daily = max(0, stock.patient_footfall_daily + footfall_delta)

    stock.updated_at = datetime.utcnow()


def tick() -> None:
    db = SessionLocal()
    try:
        rows = db.query(ResourceStock).all()
        for row in rows:
            _jitter_stock_row(row)
        db.commit()
        print(f"[live_simulator] tick @ {datetime.utcnow().isoformat()} — updated {len(rows)} stock rows")
    except Exception as e:
        db.rollback()
        print(f"[live_simulator] tick failed: {e}")
    finally:
        db.close()


scheduler = BackgroundScheduler()


def start_simulator() -> None:
    if not scheduler.running:
        scheduler.add_job(tick, "interval", seconds=TICK_SECONDS, id="live_simulator_tick", replace_existing=True)
        scheduler.start()
        print(f"[live_simulator] started — ticking every {TICK_SECONDS}s")


def stop_simulator() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[live_simulator] stopped")
