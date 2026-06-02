import pandas as pd
from pathlib import Path

# each hour maps directly to its own column
HOUR_TO_WINDOW = {h: f"{h:02d}:00 - {(h+1):02d}:00" for h in range(24)}
# special case: hour 23 → "23:00 - 00:00"
HOUR_TO_WINDOW[23] = "23:00 - 00:00"

_OD_CACHE:         dict = {}
_MULTIPLIER_CACHE: dict = {}
_DF_CACHE:         pd.DataFrame = None


def load_tomtom_od(csv_path: str = None) -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE
    if csv_path is None:
        csv_path = "data/raw/od_matrix_hour.csv"
    _DF_CACHE = pd.read_csv(csv_path)
    return _DF_CACHE


def _get_col(hour: int, is_weekend: bool, data_type: str) -> str:
    """Return the CSV column name for a given hour, day type, and data type."""
    day_type   = "Weekends" if is_weekend else "Weekdays"
    window     = HOUR_TO_WINDOW[hour % 24]
    return f"{day_type} {window} {data_type}"


def build_od_matrix(is_weekend: bool = False, hour: int = None) -> dict:
    """
    Build OD probability matrix from TomTom Percent columns.
    If hour is given, uses the matching time window.
    If hour is None, averages across all windows.
    """
    df     = load_tomtom_od()
    cities = sorted(set(df["Origin"].unique()) | set(df["Destination"].unique()))
    matrix = {c: {d: 0.0 for d in cities} for c in cities}

    if hour is not None:
        col = _get_col(hour, is_weekend, "Percent")
        for _, row in df.iterrows():
            origin = row["Origin"]
            dest   = row["Destination"]
            if origin != dest and origin in cities and dest in cities:
                matrix[origin][dest] = row[col] / 100.0

    else:
        day_type  = "Weekends" if is_weekend else "Weekdays"
        perc_cols = [c for c in df.columns
                     if day_type in c and "Percent" in c]
        for _, row in df.iterrows():
            origin = row["Origin"]
            dest   = row["Destination"]
            if origin != dest and origin in cities and dest in cities:
                matrix[origin][dest] = row[perc_cols].mean() / 100.0

    return matrix


def get_multiplier(hour: int, is_weekend: bool = False) -> float:
    """
    Derive hourly demand multiplier from TomTom trip counts.
    Normalized so average multiplier across 24h = 1.0.
    """
    df        = load_tomtom_od()
    day_type  = "Weekends" if is_weekend else "Weekdays"
    trip_cols = [c for c in df.columns if day_type in c and "Trips" in c]

    total_trips  = df[trip_cols].sum().sum()
    col          = _get_col(hour, is_weekend, "Trips")
    hour_trips = df[col].sum()

    hourly_share = hour_trips / total_trips
    avg_share    = 1.0 / 24
    return round(hourly_share / avg_share, 3)


def build_origin_weights(is_weekend: bool = False) -> dict:
    """Derive origin demand weights from total outgoing trip counts."""
    df        = load_tomtom_od()
    day_type  = "Weekends" if is_weekend else "Weekdays"
    trip_cols = [c for c in df.columns if day_type in c and "Trips" in c]

    origin_totals = df.groupby("Origin")[trip_cols].sum().sum(axis=1)
    weights       = origin_totals / origin_totals.sum()
    return weights.to_dict()


def get_od_matrix_cached(is_weekend: bool, hour: int) -> dict:
    key = (is_weekend, hour)
    if key not in _OD_CACHE:
        _OD_CACHE[key] = build_od_matrix(is_weekend=is_weekend, hour=hour)
    return _OD_CACHE[key]


def get_multiplier_cached(hour: int, is_weekend: bool) -> float:
    key = (is_weekend, hour)
    if key not in _MULTIPLIER_CACHE:
        _MULTIPLIER_CACHE[key] = get_multiplier(hour, is_weekend)
    return _MULTIPLIER_CACHE[key]


def preload_all_caches():
    print("  Preloading OD matrices and multipliers...", end=" ")
    for is_weekend in [False, True]:
        for hour in range(24):
            get_od_matrix_cached(is_weekend, hour)
            get_multiplier_cached(hour, is_weekend)
    print(f"done ({len(_OD_CACHE)} OD matrices, "
          f"{len(_MULTIPLIER_CACHE)} multipliers)")


def get_daily_profile(is_weekend: bool = False) -> list:
    return [get_multiplier(h, is_weekend) for h in range(24)]


def print_od_matrix(matrix: dict):
    cities = list(matrix.keys())
    col_w  = 12
    header = f"{'':15}" + "".join(f"{c:>{col_w}}" for c in cities)
    print(header)
    print("-" * len(header))
    for origin in cities:
        row = f"{origin:<15}"
        for dest in cities:
            val = matrix[origin][dest]
            row += f"{val:>{col_w}.3f}" if val > 0 else f"{'—':>{col_w}}"
        print(row)


def print_daily_profile(is_weekend: bool = False):
    label = "Weekend" if is_weekend else "Weekday"
    print(f"\n{label} demand profile (from TomTom):")
    for hour in range(24):
        mult = get_multiplier(hour, is_weekend)
        bar  = "█" * int(mult * 20)
        print(f"  {hour:02d}:00  {bar:<30} {mult:.3f}")


if __name__ == "__main__":
    print("Weekday morning OD matrix (07:00-08:00):")
    print_od_matrix(build_od_matrix(is_weekend=False, hour=7))
    print("\nWeekend midday OD matrix (10:00-14:00):")
    print_od_matrix(build_od_matrix(is_weekend=True, hour=12))
    print_daily_profile(is_weekend=False)
    print_daily_profile(is_weekend=True)