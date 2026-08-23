from app.database import SessionLocal, Base, engine
from app.models.phc import PHC
from app.models.resource_stock import ResourceStock
from app.models.health_alert import HealthAlert
from app.models.user_device import UserDevice
from datetime import datetime

Base.metadata.create_all(bind=engine)


def run():
    db = SessionLocal()

    if db.query(PHC).count() == 0:
        phcs = [
            PHC(name="PHC Bhiwandi", state="Maharashtra", district="Thane", latitude=19.2967, longitude=73.0631, type="PHC", is_remote=0),
            PHC(name="PHC Nandurbar", state="Maharashtra", district="Nandurbar", latitude=21.3667, longitude=74.2500, type="PHC", is_remote=1),
            PHC(name="CHC Pune Rural", state="Maharashtra", district="Pune", latitude=18.5204, longitude=73.8567, type="CHC", is_remote=0),
            PHC(name="PHC Barmer", state="Rajasthan", district="Barmer", latitude=25.7521, longitude=71.3961, type="PHC", is_remote=1),
            PHC(name="PHC Jaisalmer", state="Rajasthan", district="Jaisalmer", latitude=26.9157, longitude=70.9083, type="PHC", is_remote=1),
            PHC(name="District Warehouse Jaipur", state="Rajasthan", district="Jaipur", latitude=26.9124, longitude=75.7873, type="district_warehouse", is_remote=0),
            PHC(name="PHC Wayanad", state="Kerala", district="Wayanad", latitude=11.6854, longitude=76.1320, type="PHC", is_remote=1),
            PHC(name="PHC Kochi Urban", state="Kerala", district="Ernakulam", latitude=9.9312, longitude=76.2673, type="PHC", is_remote=0),
            PHC(name="PHC Koraput", state="Odisha", district="Koraput", latitude=18.8120, longitude=82.7100, type="PHC", is_remote=1),
            PHC(name="District Warehouse Bhubaneswar", state="Odisha", district="Khordha", latitude=20.2961, longitude=85.8245, type="district_warehouse", is_remote=0),
            PHC(name="PHC Gorakhpur Rural", state="Uttar Pradesh", district="Gorakhpur", latitude=26.7606, longitude=83.3732, type="PHC", is_remote=0),
            PHC(name="PHC Sonbhadra", state="Uttar Pradesh", district="Sonbhadra", latitude=24.6913, longitude=83.0672, type="PHC", is_remote=1),
        ]
        db.add_all(phcs)
        db.commit()

    all_phcs = db.query(PHC).all()

    if db.query(ResourceStock).count() == 0:
        stock_data = [
            # (phc_name, medicine_category, stock_units, capacity, days_remaining, beds_total, beds_occ, staff_total, staff_present, footfall)
            ("PHC Bhiwandi", "antibiotics", 800, 1000, 6.0, 20, 14, 12, 11, 95),
            ("PHC Nandurbar", "antibiotics", 60, 800, 0.8, 15, 14, 8, 5, 210),
            ("CHC Pune Rural", "antibiotics", 1400, 1500, 8.0, 40, 22, 25, 23, 150),
            ("PHC Barmer", "ORS", 90, 900, 1.2, 18, 16, 10, 6, 190),
            ("PHC Jaisalmer", "ORS", 850, 1000, 7.5, 12, 6, 7, 7, 70),
            ("District Warehouse Jaipur", "ORS", 3000, 3200, 20.0, 0, 0, 15, 14, 0),
            ("PHC Wayanad", "vaccines", 100, 1000, 1.5, 22, 19, 11, 7, 175),
            ("PHC Kochi Urban", "vaccines", 1200, 1300, 9.0, 30, 15, 18, 17, 110),
            ("PHC Koraput", "insulin", 40, 600, 0.6, 16, 15, 9, 5, 205),
            ("District Warehouse Bhubaneswar", "insulin", 2500, 2700, 18.0, 0, 0, 12, 11, 0),
            ("PHC Gorakhpur Rural", "insulin", 700, 900, 6.5, 25, 12, 14, 13, 100),
            ("PHC Sonbhadra", "vaccines", 70, 850, 1.0, 14, 13, 8, 4, 200),
        ]
        phc_by_name = {p.name: p for p in all_phcs}
        for name, cat, units, cap, days, bed_t, bed_o, staff_t, staff_p, footfall in stock_data:
            db.add(ResourceStock(
                phc_id=phc_by_name[name].id,
                medicine_category=cat,
                stock_units=units,
                stock_capacity=cap,
                days_of_stock_remaining=days,
                bed_total=bed_t,
                bed_occupied=bed_o,
                staff_total=staff_t,
                staff_present=staff_p,
                patient_footfall_daily=footfall,
                updated_at=datetime.utcnow(),
            ))
        db.commit()

    if db.query(HealthAlert).count() == 0:
        alerts = [
            HealthAlert(time="06:45", type="stockout", phc_name="PHC Nandurbar", district="Nandurbar", severity=5,
                        summary="Antibiotic stock critically low, less than 1 day remaining.", age_min=30, dismissed=False),
            HealthAlert(time="05:20", type="stockout", phc_name="PHC Barmer", district="Barmer", severity=5,
                        summary="ORS stock near depletion amid seasonal diarrheal outbreak.", age_min=95, dismissed=False),
            HealthAlert(time="08:10", type="staff_shortage", phc_name="PHC Sonbhadra", district="Sonbhadra", severity=4,
                        summary="Only 4 of 8 sanctioned staff present today.", age_min=15, dismissed=False),
            HealthAlert(time="07:00", type="stockout", phc_name="PHC Koraput", district="Koraput", severity=5,
                        summary="Insulin stock critically low in remote tribal block.", age_min=60, dismissed=False),
            HealthAlert(time="09:30", type="demand_spike", phc_name="PHC Wayanad", district="Wayanad", severity=4,
                        summary="Vaccine demand spike following local outbreak advisory.", age_min=5, dismissed=False),
        ]
        db.add_all(alerts)
        db.commit()

    if db.query(UserDevice).count() == 0:
        db.add(UserDevice(label="District Health Officer", phone_number="7356675700", active=True))
        db.commit()

    db.close()
    print(f"Seeded {len(all_phcs)} PHCs, {db.query(ResourceStock).count() if db.query(ResourceStock).count() else 'existing'} stock records, alerts, and 1 device.")


if __name__ == "__main__":
    run()
