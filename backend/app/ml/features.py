"""
Turns raw PHC/stock data into the numeric feature vector the model expects.
Order of features MUST match train.py and model.py exactly.
"""

FEATURE_NAMES = [
    "days_of_stock_remaining",
    "stock_pct",              # stock_units / stock_capacity
    "bed_occupancy_pct",
    "staff_attendance_pct",
    "patient_footfall_daily",
    "is_remote",
    "demand_trend_pct",       # week-over-week footfall change
    "district_alert_density", # active alerts per PHC in the same district
]


def build_features(data: dict) -> list[float]:
    return [float(data.get(name, 0)) for name in FEATURE_NAMES]


def features_to_dict(feature_values: list[float]) -> dict:
    return dict(zip(FEATURE_NAMES, feature_values))
