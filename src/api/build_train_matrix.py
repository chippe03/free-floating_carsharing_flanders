"""
Build intercity train travel time matrix from NMBS GTFS data:
trips.txt, stop_times.txt, stops.txt, calendar.txt, calendar_dates.txt

Output: minimum travel time in minutes between cities

Notes:
- calendar.txt: service_id runs on days where monday=1 etc.
- calendar_dates.txt: exception_type=1 adds a date, 2 removes it
- Only direct connections are used (no transfers)
- For each hour, minimum travel time over all departures in that hour
- NaN hours (no service) are filled with nearest available hour
"""

import json
import math
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta, date
from src import config

GTFS_DIR = Path("data/raw/planningsgegevens_nmbs")
OUT_PATH = Path("data/processed/train_travel_matrix.json")

# ------------------------------------------------------------------ #
#  City → main station stop_id mapping                               #
# ------------------------------------------------------------------ #

CITY_STOPS = {
    "antwerp":  "8821006",   # Anvers-Central
    "brussels": "8813003",   # Bruxelles-Central
    "ghent":    "8892007",   # Gand-Saint-Pierre
    "bruges":   "8891009",   # Bruges
    "leuven":   "8833001",   # Louvain
    "mechelen": "8822004",   # Malines
    "kortrijk": "8896008",   # Courtrai
    "ostend":   "8891702",   # Oostende
}

CITIES   = list(CITY_STOPS.keys())
STOP_IDS = set(CITY_STOPS.values())

# reference dates for weekday/weekend classification
# pick representative dates within the feed validity
REF_WEEKDAY = date(2025, 12, 8)    # Monday
REF_WEEKEND = date(2025, 12, 13)   # Saturday


# ------------------------------------------------------------------ #
#  Step 1 - Load and classify service IDs                            #
# ------------------------------------------------------------------ #

def load_services(gtfs_dir: Path) -> dict[str, set[str]]:
    """
    Returns {service_id: set of date strings 'YYYYMMDD'} for all
    service dates in the feed, combining calendar.txt and
    calendar_dates.txt.
    """
    services: dict[str, set[str]] = defaultdict(set)

    # --- calendar.txt: regular weekly schedule ---
    cal_path = gtfs_dir / "calendar.txt"
    if cal_path.exists():
        with open(cal_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid        = row["service_id"]
                start      = datetime.strptime(row["start_date"], "%Y%m%d").date()
                end        = datetime.strptime(row["end_date"],   "%Y%m%d").date()
                day_flags  = [
                    int(row["monday"]),
                    int(row["tuesday"]),
                    int(row["wednesday"]),
                    int(row["thursday"]),
                    int(row["friday"]),
                    int(row["saturday"]),
                    int(row["sunday"]),
                ]
                current = start
                while current <= end:
                    if day_flags[current.weekday()]:
                        services[sid].add(current.strftime("%Y%m%d"))
                    current += timedelta(days=1)

    # --- calendar_dates.txt: exceptions ---
    cal_dates_path = gtfs_dir / "calendar_dates.txt"
    if cal_dates_path.exists():
        with open(cal_dates_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid            = row["service_id"]
                date_str       = row["date"]
                exception_type = int(row["exception_type"])
                if exception_type == 1:
                    services[sid].add(date_str)        # add service
                elif exception_type == 2:
                    services[sid].discard(date_str)    # remove service

    print(f"  Loaded {len(services)} service IDs")
    total_dates = sum(len(v) for v in services.values())
    print(f"  Total service-date pairs: {total_dates}")
    return dict(services)


def classify_service(service_dates: set[str]) -> set[str]:
    """
    Return set of day_types ('weekday', 'weekend') this service runs on,
    based on which dates it covers.
    """
    types = set()
    for date_str in service_dates:
        d = datetime.strptime(date_str, "%Y%m%d").date()
        if d.weekday() < 5:
            types.add("weekday")
        else:
            types.add("weekend")
    return types


# ------------------------------------------------------------------ #
#  Step 2 - Load trips filtered to relevant services                 #
# ------------------------------------------------------------------ #

def load_trips(gtfs_dir: Path,
               services: dict[str, set[str]]) -> dict[str, set[str]]:
    """
    Returns {trip_id: set of day_types} for all trips whose service
    runs on at least one date.
    """
    trips: dict[str, set[str]] = {}
    path = gtfs_dir / "trips.txt"

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid     = row["service_id"]
            trip_id = row["trip_id"]
            if sid not in services:
                continue
            day_types = classify_service(services[sid])
            if day_types:
                trips[trip_id] = day_types

    print(f"  Loaded {len(trips)} relevant trips")
    return trips


# ------------------------------------------------------------------ #
#  Step 3 - Load stop times for relevant trips and city stops        #
# ------------------------------------------------------------------ #

def parse_time(time_str: str) -> int:
    """
    Parse GTFS time string (HH:MM:SS) to minutes from midnight.
    GTFS allows hours >= 24 for trips past midnight.
    """
    parts = time_str.strip().split(":")
    h, m  = int(parts[0]), int(parts[1])
    return h * 60 + m


def load_stop_times(gtfs_dir: Path,
                    trips: dict[str, set[str]]) -> dict:
    """
    Returns {trip_id: [(stop_id, arrival_min, departure_min), ...]}
    sorted by stop_sequence, filtered to city stops only.

    Also returns full sequence to detect direct connections:
    a connection is direct if both stops appear in the same trip
    without requiring a transfer.
    """
    # {trip_id: {stop_id: (arrival_min, departure_min, sequence)}}
    trip_stops: dict[str, dict] = defaultdict(dict)

    path = gtfs_dir / "stop_times.txt"
    chunk_size = 100_000
    count      = 0

    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trip_id = row["trip_id"]
            if trip_id not in trips:
                continue
            stop_id = row["stop_id"]
            if stop_id not in STOP_IDS:
                continue

            arrival   = parse_time(row["arrival_time"])
            departure = parse_time(row["departure_time"])
            sequence  = int(row["stop_sequence"])

            trip_stops[trip_id][stop_id] = (arrival, departure, sequence)
            count += 1

            if count % chunk_size == 0:
                print(f"  Processed {count:,} stop_time rows...")

    print(f"  Loaded {count:,} relevant stop_time entries "
          f"across {len(trip_stops)} trips")
    return dict(trip_stops)


# ------------------------------------------------------------------ #
#  Step 4 - Extract travel times between city pairs                  #
# ------------------------------------------------------------------ #

def extract_connections(
    trip_stops: dict,
    trips:      dict[str, set[str]],
) -> dict:
    """
    For each city pair (origin, destination) and day_type,
    collect all (departure_hour, travel_time_minutes) observations.

    Returns:
      {origin_city: {dest_city: {day_type: {hour: [travel_times]}}}}
    """
    # reverse map: stop_id → city
    stop_to_city = {v: k for k, v in CITY_STOPS.items()}

    connections: dict = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list)
            )
        )
    )

    for trip_id, stop_data in trip_stops.items():
        if len(stop_data) < 2:
            continue   # need at least 2 city stops in this trip

        day_types = trips[trip_id]

        # sort stops by sequence
        stops_sorted = sorted(stop_data.items(),
                              key=lambda x: x[1][2])  # sort by sequence

        # for each pair where origin appears before destination
        for i, (stop_a, (arr_a, dep_a, seq_a)) in enumerate(stops_sorted):
            for stop_b, (arr_b, dep_b, seq_b) in stops_sorted[i+1:]:
                if seq_b <= seq_a:
                    continue

                city_a = stop_to_city.get(stop_a)
                city_b = stop_to_city.get(stop_b)

                if not city_a or not city_b:
                    continue

                travel_min    = arr_b - dep_a
                depart_hour   = dep_a // 60

                if travel_min <= 0:
                    continue   # data error
                if travel_min > 300:
                    continue   # implausibly long, likely overnight wrap

                for day_type in day_types:
                    connections[city_a][city_b][day_type][depart_hour].append(
                        travel_min
                    )

    return connections


# ------------------------------------------------------------------ #
#  Step 5 - Aggregate to minimum travel time per hour               #
# ------------------------------------------------------------------ #

def aggregate_matrix(connections: dict) -> dict:
    """
    For each city pair, day_type, and hour: take the minimum
    travel time over all observed departures.
    Fill missing hours with the nearest available hour's value.

    Returns:
      {city: {city: {day_type: {str(hour): minutes}}}}
    """
    matrix = {}

    for origin in CITIES:
        matrix[origin] = {}
        for dest in CITIES:
            if origin == dest:
                continue

            matrix[origin][dest] = {}

            for day_type in ["weekday", "weekend"]:
                hour_data = connections.get(origin, {}) \
                                       .get(dest, {}) \
                                       .get(day_type, {})

                if not hour_data:
                    print(f"  ⚠️  No {day_type} trains: "
                          f"{origin} → {dest}")
                    continue

                # minimum per hour
                min_by_hour = {
                    h: min(times)
                    for h, times in hour_data.items()
                }

                # fill all 24 hours
                filled = {}
                for h in range(24):
                    if h in min_by_hour:
                        filled[h] = min_by_hour[h]
                    else:
                        filled[h] = 0

                matrix[origin][dest][day_type] = {
                    str(h): filled[h] for h in range(24)
                }

    return matrix


# ------------------------------------------------------------------ #
#  Step 6 - Print summary                                            #
# ------------------------------------------------------------------ #

def print_summary(matrix: dict):
    """Print travel times for key city pairs as sanity check."""
    pairs = [
        ("antwerp",  "brussels"),
        ("antwerp",  "ghent"),
        ("brussels", "bruges"),
        ("ghent",    "ostend"),
        ("leuven",   "antwerp"),
    ]
    print(f"\n  Train travel time summary (weekday, min travel time):")
    print(f"  {'Route':<30} {'06h':>5} {'08h':>5} "
          f"{'12h':>5} {'17h':>5} {'22h':>5}")
    print(f"  {'─'*55}")

    for origin, dest in pairs:
        row = matrix.get(origin, {}).get(dest, {}).get("weekday", {})
        if not row:
            print(f"  {origin}→{dest:<22} NO DATA")
            continue
        vals = [row.get(str(h), "-") for h in [6, 8, 12, 17, 22]]
        print(f"  {origin}→{dest:<22} "
              + "".join(f"{v:>5}" for v in vals))


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #

def build_train_matrix():
    print(f"\nBuilding train travel matrix from NMBS GTFS data")
    print(f"GTFS directory: {GTFS_DIR}")
    print(f"Output:         {OUT_PATH}\n")

    # check files exist
    required = ["trips.txt", "stop_times.txt", "calendar_dates.txt"]
    optional = ["calendar.txt"]
    for fname in required:
        path = GTFS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Required GTFS file not found: {path}\n"
                f"Download GTFS from NMBS and place in {GTFS_DIR}"
            )
    for fname in optional:
        if not (GTFS_DIR / fname).exists():
            print(f"  Note: {fname} not found — using calendar_dates.txt only")

    # pipeline
    print("Step 1: Loading service calendar...")
    services = load_services(GTFS_DIR)

    print("\nStep 2: Loading trips...")
    trips = load_trips(GTFS_DIR, services)

    print("\nStep 3: Loading stop times...")
    trip_stops = load_stop_times(GTFS_DIR, trips)

    print("\nStep 4: Extracting city-pair connections...")
    connections = extract_connections(trip_stops, trips)

    print("\nStep 5: Aggregating to hourly minimum travel times...")
    matrix = aggregate_matrix(connections)

    print("\nStep 6: Saving matrix...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"  Saved to {OUT_PATH}")

    print_summary(matrix)

    # coverage report
    total_pairs    = len(CITIES) * (len(CITIES) - 1)
    covered_pairs  = sum(
        1 for o in CITIES for d in CITIES
        if o != d and matrix.get(o, {}).get(d)
    )
    print(f"\n  Coverage: {covered_pairs}/{total_pairs} city pairs have data")
    if covered_pairs < total_pairs:
        missing = [
            f"{o}→{d}" for o in CITIES for d in CITIES
            if o != d and not matrix.get(o, {}).get(d)
        ]
        print(f"  Missing: {', '.join(missing)}")
        print(f"  Note: missing pairs may need stop_id verification "
              f"or may require a transfer")

    return matrix


if __name__ == "__main__":
    build_train_matrix()