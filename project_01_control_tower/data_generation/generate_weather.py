"""
Generate the weather table.
One row per store × date (300 stores × 731 days ≈ 219,300 rows).

Temperature model:
  - Sinusoidal seasonal curve based on latitude (northern stores have colder winters)
  - Gaussian noise added daily
  - Precipitation correlated with season and climate zone
  - Severity events: 5% notable, 1% severe, 0.1% extreme per store-day
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import RANDOM_SEED, START_DATE, END_DATE

rng = np.random.default_rng(RANDOM_SEED + 2)

CONDITIONS = ["Clear", "Partly Cloudy", "Cloudy", "Rain", "Snow", "Storm", "Fog"]


def _climate_zone(lat: float, state: str) -> str:
    """Assign a broad climate zone from latitude and state."""
    if state in ("FL", "HI"):
        return "tropical"
    if lat >= 45:
        return "northern"
    if lat >= 38:
        return "temperate"
    if lat >= 30:
        return "southern"
    return "subtropical"


def _temp_params(climate: str, doy: float) -> tuple:
    """
    Return (mean_temp_f, amplitude) for a given climate zone.
    doy: day of year 1–365, converted to radians for sinusoidal model.
    Phase shift so that peak is ~July 15 (doy≈196).
    """
    phase = 2 * np.pi * (doy - 196) / 365
    params = {
        "northern":    (45.0, 35.0),
        "temperate":   (55.0, 28.0),
        "southern":    (63.0, 22.0),
        "subtropical": (70.0, 18.0),
        "tropical":    (78.0, 8.0),
    }
    mean_annual, amplitude = params[climate]
    mean_temp = mean_annual - amplitude * np.cos(phase)
    return mean_temp


def _precip_prob(climate: str, month: int) -> float:
    """Probability of precipitation on any given day."""
    base = {"northern": 0.35, "temperate": 0.30, "southern": 0.25, "subtropical": 0.25, "tropical": 0.40}
    p = base[climate]
    # Seasonal adjustment: more rain in spring/fall for temperate; summer for tropical
    if climate in ("tropical", "subtropical") and month in (6, 7, 8, 9):
        p += 0.15
    if climate == "northern" and month in (11, 12, 1, 2):
        p += 0.05  # winter precipitation (snow)
    return min(p, 0.60)


def generate_weather(stores: pd.DataFrame) -> pd.DataFrame:
    all_dates = pd.date_range(START_DATE, END_DATE).date.tolist()
    records = []

    store_list = stores[["store_id", "latitude", "state"]].to_dict("records")

    for store in store_list:
        sid    = store["store_id"]
        lat    = store["latitude"]
        state  = store["state"]
        climate = _climate_zone(lat, state)

        # Per-store random offset so stores aren't identical
        store_noise = rng.normal(0, 3.0)

        for d in all_dates:
            doy   = d.timetuple().tm_yday
            month = d.month

            mean_temp = _temp_params(climate, doy) + store_noise
            daily_noise = rng.normal(0, 5.0)
            avg_temp  = round(float(mean_temp + daily_noise), 1)
            temp_range = max(5.0, float(rng.uniform(8, 20)))
            min_temp  = round(avg_temp - temp_range / 2, 1)
            max_temp  = round(avg_temp + temp_range / 2, 1)

            precip_prob = _precip_prob(climate, month)
            has_precip  = rng.random() < precip_prob
            snow_prob   = 0.0
            if min_temp <= 32 and month in (11, 12, 1, 2, 3):
                snow_prob = max(0.0, (32 - min_temp) / 32 * 0.6)

            precip_in = 0.0
            snow_in   = 0.0
            if has_precip:
                if rng.random() < snow_prob:
                    snow_in   = round(float(rng.exponential(2.0)), 2)
                    precip_in = round(snow_in * 0.1, 2)
                else:
                    precip_in = round(float(rng.exponential(0.4)), 2)

            # Condition
            if snow_in > 3:
                condition = "Snow"
            elif snow_in > 0:
                condition = "Snow"
            elif precip_in > 1.0:
                condition = "Storm"
            elif precip_in > 0.1:
                condition = "Rain"
            elif precip_in > 0:
                condition = "Rain"
            elif avg_temp < 35 or avg_temp > 95:
                condition = "Partly Cloudy"
            elif rng.random() < 0.1:
                condition = "Fog"
            elif rng.random() < 0.3:
                condition = "Partly Cloudy"
            elif rng.random() < 0.2:
                condition = "Cloudy"
            else:
                condition = "Clear"

            # Severity
            severity = 1
            r = rng.random()
            if snow_in > 6 or precip_in > 2.0 or max_temp > 105 or min_temp < 0:
                severity = 4  # extreme
            elif snow_in > 3 or precip_in > 1.0 or max_temp > 100 or min_temp < 10:
                severity = 3  # severe
            elif snow_in > 0.5 or precip_in > 0.5 or max_temp > 95 or min_temp < 20:
                severity = 2  # notable
            # Additional stochastic bumps (5% notable, 1% severe override)
            if severity == 1:
                if r < 0.01:
                    severity = 3
                elif r < 0.05:
                    severity = 2

            records.append({
                "store_id":         sid,
                "date_id":          d,
                "avg_temp_f":       avg_temp,
                "min_temp_f":       min_temp,
                "max_temp_f":       max_temp,
                "precipitation_in": precip_in,
                "snow_in":          snow_in,
                "condition":        condition,
                "weather_severity": severity,
            })

    df = pd.DataFrame(records)
    print(f"  Weather: {len(df):,} rows")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from generate_dimensions import generate_distribution_centers, generate_stores
    dcs    = generate_distribution_centers()
    stores = generate_stores(dcs)
    weather = generate_weather(stores)
    print(weather.head())
    print(weather.groupby("weather_severity").size())
