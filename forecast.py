"""Attendance trend forecasting based on historical fest category records."""

import numpy as np
import pandas as pd


def forecast_attendance(
    past: pd.DataFrame,
    target_year: int = 2026,
    degree: int = 1,
) -> pd.DataFrame:
    """Forecast future attendance per event category using polynomial trend regression."""
    rows = []
    categories = sorted(list(past["category"].unique()))

    for cat in categories:
        sub = past[past["category"] == cat].sort_values("year")
        years = sub["year"].to_numpy(dtype=float)
        att = sub["attendance"].to_numpy(dtype=float)

        if len(years) < 2:
            last_actual = int(att[-1]) if len(att) > 0 else 0
            rows.append(
                {
                    "category": cat,
                    "forecast": last_actual,
                    "slope": 0.0,
                    "r2": 1.0,
                    "last_actual": last_actual,
                }
            )
            continue

        last_actual = int(att[np.argmax(years)])
        coeffs = np.polyfit(years, att, degree)
        pred_target = np.polyval(coeffs, target_year)
        forecast_val = max(0, int(round(pred_target)))
        slope = float(coeffs[0]) if degree == 1 else float(coeffs[-2])

        y_fitted = np.polyval(coeffs, years)
        ss_res = np.sum((att - y_fitted) ** 2)
        ss_tot = np.sum((att - np.mean(att)) ** 2)

        if ss_tot == 0.0 or ss_res < 1e-12:
            r2 = 1.0
        else:
            r2 = max(0.0, min(1.0, float(1.0 - (ss_res / ss_tot))))

        rows.append(
            {
                "category": cat,
                "forecast": forecast_val,
                "slope": slope,
                "r2": r2,
                "last_actual": last_actual,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["category", "forecast", "slope", "r2", "last_actual"],
    )
