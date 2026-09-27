"""
Generate shipments and shipment_events.

Two types of shipments:
1. Inbound: Supplier → DC (one per purchase order)
2. Outbound: DC → Store (weekly replenishment runs)

Shipment events form a status log (2–5 events per shipment).
Delay probability amplified during severe weather.
"""

from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

from config import RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED + 4)

CARRIERS = ["FreightPro", "TransLink", "NationalTruck", "SwiftFreight", "RegionalHaul", "ExpressLTL"]
EVENT_TYPES = ["Dispatched", "In_Transit", "Arrived_DC", "Customs_Hold",
               "Delay_Weather", "Delay_Mechanical", "Delivered", "Exception"]


def _shipment_events(shipment_id: int, ship_date: date, actual_arrival: date,
                     is_delayed: bool, dest_type: str, rng: np.random.Generator) -> list:
    """Generate 2–5 events for a shipment."""
    events = []
    current_ts = datetime.combine(ship_date, datetime.min.time()).replace(
        hour=int(rng.integers(6, 14)), minute=int(rng.integers(0, 60))
    )

    # Event 1: always Dispatched
    events.append({
        "shipment_id":     shipment_id,
        "event_timestamp": current_ts,
        "event_type":      "Dispatched",
        "location":        "Origin",
        "notes":           None,
        "delay_hours":     0,
    })

    # Event 2: In_Transit (a day or two later)
    current_ts += timedelta(hours=int(rng.integers(12, 36)))
    events.append({
        "shipment_id":     shipment_id,
        "event_timestamp": current_ts,
        "event_type":      "In_Transit",
        "location":        "In Transit",
        "notes":           None,
        "delay_hours":     0,
    })

    # Delay events (if delayed)
    if is_delayed:
        current_ts += timedelta(hours=int(rng.integers(6, 24)))
        delay_type = "Delay_Weather" if rng.random() < 0.6 else "Delay_Mechanical"
        delay_hours = int(rng.integers(12, 72))
        events.append({
            "shipment_id":     shipment_id,
            "event_timestamp": current_ts,
            "event_type":      delay_type,
            "location":        "En Route",
            "notes":           f"Delay of {delay_hours}h reported",
            "delay_hours":     delay_hours,
        })
        current_ts += timedelta(hours=delay_hours)

    # Intermediate arrival event for DC destinations
    if dest_type == "DC" and rng.random() < 0.7:
        current_ts += timedelta(hours=int(rng.integers(2, 12)))
        events.append({
            "shipment_id":     shipment_id,
            "event_timestamp": current_ts,
            "event_type":      "Arrived_DC",
            "location":        "Destination DC",
            "notes":           None,
            "delay_hours":     0,
        })

    # Final: Delivered
    if actual_arrival is not None:
        delivery_ts = datetime.combine(actual_arrival, datetime.min.time()).replace(
            hour=int(rng.integers(8, 18)), minute=int(rng.integers(0, 60))
        )
        events.append({
            "shipment_id":     shipment_id,
            "event_timestamp": delivery_ts,
            "event_type":      "Delivered",
            "location":        "Destination",
            "notes":           None,
            "delay_hours":     0,
        })

    return events


def generate_shipments(
    purchase_orders: pd.DataFrame,
    dcs: pd.DataFrame,
    stores: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    shipment_records = []
    event_records    = []
    sid = 1

    # Build a weather severity lookup: (store_id, date) → severity
    wx_lookup = weather.set_index(["store_id", "date_id"])["weather_severity"].to_dict()

    # ── Inbound shipments: one per received/shipped PO ──────
    received_pos = purchase_orders[purchase_orders["status"].isin(["Received", "Shipped"])]
    for _, po in received_pos.iterrows():
        ship_date        = po["order_date"] + timedelta(days=int(rng.integers(1, 3)))
        expected_arrival = max(po["requested_delivery"], ship_date + timedelta(days=1))
        actual_delivery  = po["actual_delivery"]
        # Ensure actual_arrival is never before ship_date
        if actual_delivery is not None and actual_delivery < ship_date:
            actual_delivery = ship_date + timedelta(days=1)
        actual_arrival   = actual_delivery
        is_delayed       = bool(po["is_late"])
        status = "Delivered" if actual_arrival is not None else "In_Transit"
        if bool(po["is_late"]) and rng.random() < 0.05:
            status = "Delayed"

        rec = {
            "shipment_id":        sid,
            "shipment_number":    f"SHP-{sid:08d}",
            "po_id":              int(po["po_id"]),
            "origin_type":        "Supplier",
            "origin_supplier_id": int(po["supplier_id"]),
            "origin_dc_id":       None,
            "dest_type":          "DC",
            "dest_dc_id":         int(po["dc_id"]),
            "dest_store_id":      None,
            "carrier":            str(rng.choice(CARRIERS)),
            "ship_date":          ship_date,
            "expected_arrival":   expected_arrival,
            "actual_arrival":     actual_arrival,
            "status":             status,
            "total_units":        int(po["total_units"]) if po["total_units"] else 0,
            "total_weight_kg":    round(float(rng.uniform(500, 10000)), 2),
            "freight_cost":       round(float(rng.uniform(200, 2000)), 2),
        }
        shipment_records.append(rec)
        event_records.extend(_shipment_events(sid, ship_date, actual_arrival, is_delayed, "DC", rng))
        sid += 1

    # ── Outbound shipments: DC → Store (weekly) ──────────────
    # Simplified: each store receives a weekly DC shipment
    from datetime import date as date_type
    from config import START_DATE, END_DATE
    start = date_type.fromisoformat(START_DATE)
    end   = date_type.fromisoformat(END_DATE)

    current = start
    while current.weekday() != 0:
        current += timedelta(days=1)

    store_dc_map = stores[["store_id", "dc_id"]].to_dict("records")

    while current <= end:
        for sd in store_dc_map:
            store_id = int(sd["store_id"])
            dc_id    = int(sd["dc_id"])
            ship_date = current
            transit_days = int(rng.integers(1, 4))

            # Weather delay amplification
            wx_key = (store_id, ship_date + timedelta(days=transit_days))
            severity = wx_lookup.get(wx_key, 1)
            base_delay_prob = 0.05
            delay_prob = base_delay_prob + (severity - 1) * 0.05

            is_delayed    = rng.random() < delay_prob
            extra_days    = int(rng.integers(1, 3)) if is_delayed else 0
            expected_arr  = ship_date + timedelta(days=transit_days)
            actual_arr    = expected_arr + timedelta(days=extra_days)
            if actual_arr > end:
                actual_arr = None
                status = "In_Transit"
            else:
                status = "Delayed" if is_delayed else "Delivered"

            rec = {
                "shipment_id":        sid,
                "shipment_number":    f"SHP-{sid:08d}",
                "po_id":              None,
                "origin_type":        "DC",
                "origin_supplier_id": None,
                "origin_dc_id":       dc_id,
                "dest_type":          "Store",
                "dest_dc_id":         None,
                "dest_store_id":      store_id,
                "carrier":            str(rng.choice(CARRIERS)),
                "ship_date":          ship_date,
                "expected_arrival":   expected_arr,
                "actual_arrival":     actual_arr,
                "status":             status,
                "total_units":        int(rng.integers(200, 2001)),
                "total_weight_kg":    round(float(rng.uniform(200, 3000)), 2),
                "freight_cost":       round(float(rng.uniform(50, 400)), 2),
            }
            shipment_records.append(rec)
            event_records.extend(_shipment_events(sid, ship_date, actual_arr, is_delayed, "Store", rng))
            sid += 1

        current += timedelta(weeks=1)

    ship_df  = pd.DataFrame(shipment_records)
    event_df = pd.DataFrame(event_records)
    print(f"  Shipments: {len(ship_df):,}  |  Events: {len(event_df):,}")
    return ship_df, event_df
