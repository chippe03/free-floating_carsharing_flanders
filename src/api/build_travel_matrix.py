import json
from pathlib import Path
from datetime import datetime
from src import config
from src.api.tomtom_client import TomTomClient

cities = config.cities["cities"]

city_names = list(cities.keys())
matrix = {o: {d: None for d in city_names} for o in city_names}

# representative past dates (ex. no holidays)
REPRESENTATIVE_DATES = {
    "weekday": {
        str(h): f"2024-03-04T{h:02d}:30:00"   # Monday, middle of each hour
        for h in range(24)
    },
    "weekend": {
        str(h): f"2024-03-09T{h:02d}:30:00"   # Saturday, middle of each hour
        for h in range(24)
    }
}

def build_timed_matrix():
    client     = TomTomClient()
    city_names = list(cities.keys())

    # structure: matrix[origin][destination][day_type][hour_str] = {km, min}
    matrix = {
        origin: {
            dest: {"weekday": {}, "weekend": {}}
            for dest in city_names
        }
        for origin in city_names
    }

    total = len(city_names) ** 2 * 2 * len(REPRESENTATIVE_DATES["weekday"])
    done  = 0

    for origin_name, origin_cfg in cities.items():
        for dest_name, dest_cfg in cities.items():
            if origin_name == dest_name:
                continue

            origin_coords = (origin_cfg["lat"], origin_cfg["lon"])
            dest_coords   = (dest_cfg["lat"],   dest_cfg["lon"])

            for day_type, hours in REPRESENTATIVE_DATES.items():
                for hour_str, depart_at in hours.items():
                    result = client.get_route_at_time(origin_coords, dest_coords, depart_at)
                    if result:
                        matrix[origin_name][dest_name][day_type][hour_str] = {
                            "distance_km":  result["distance_km"],
                            "duration_min": result["duration_min"],
                        }
                    done += 1
                    print(f"  [{done}/{total}] {origin_name} → {dest_name} "
                          f"({day_type} {hour_str}): "
                          f"{result['duration_min']:.1f} min" if result else "  FAILED")

    output_path = Path("data/processed/travel_matrix_timed.json")
    with open(output_path, "w") as f:
        json.dump(matrix, f, indent=2)

    print(f"\nTimed travel matrix saved to {output_path}")
    print(f"Total API calls: {done}")
    return matrix


if __name__ == "__main__":
    build_timed_matrix()