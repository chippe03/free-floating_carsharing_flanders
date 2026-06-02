import requests, json, hashlib
from pathlib import Path
import osmnx as ox
from src import config

class TomTomClient:
    BASE_URL = config.tomtom["api"]["base_url"]

    def __init__(self: str, cache_dir: str = "data/raw"):
        self.api_key = config.tomtom["api"]["key"]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, endpoint, params):
        h = hashlib.md5(json.dumps({**params, "ep": endpoint}, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{h}.json"

    def get_route_at_time(self,
                          origin: tuple, destination: tuple,
                          depart_at: str,
                          ) -> dict:
        """
        Get travel time for a specific departure datetime.
        TomTom requires a past date for historical traffic data.
        depart_at format: "YYYY-MM-DDTHH:MM:SS"
        """
        cache_file = self._cache_key(
            "route_timed",
            {"o": origin, "d": destination, "t": depart_at}
        )

        if cache_file.exists():
            return json.loads(cache_file.read_text())

        url = (f"{self.BASE_URL}/routing/1/calculateRoute/"
            f"{origin[0]},{origin[1]}:{destination[0]},{destination[1]}/json")

        resp = requests.get(url, params={
            "key":         self.api_key,
            "travelMode":  "car",
            "traffic":     "true",
            "departAt":    depart_at,
        }).json()

        if "routes" not in resp:
            print(f"  API error for {depart_at}: {resp}")
            return None

        summary = resp["routes"][0]["summary"]
        result  = {
            "origin": origin,
            "destination": destination,
            "distance_km":  round(summary["lengthInMeters"] / 1000, 2),
            "duration_min": round(summary["travelTimeInSeconds"] / 60, 2),
            "depart_at":    depart_at,
        }

        cache_file.write_text(json.dumps(result))
        return result

    def get_parkings_with_capacity(self, city_name: str,
                                    min_capacity: int = 50) -> list:
        try:
            osm_gdf = ox.features_from_place(
                query = f"{city_name}, Belgium",
                tags  = {"amenity": "parking"}
            )
        except Exception as e:
            print(f"  OSMnx failed: {e}")
            return []

        def _clean(val) -> str:
            """Convert OSM field to string, returning empty string for NaN/None."""
            if val is None:
                return ""
            s = str(val)
            return "" if s == "nan" else s.strip()
        
        parkings = []
        for idx, row in osm_gdf.iterrows():
            cap = row.get("capacity")
            if cap is None or str(cap) == "nan":
                continue
            try:
                cap_int = int(float(cap))
            except (ValueError, TypeError):
                continue
            if cap_int < min_capacity:
                continue

            try:
                geom = row.geometry
                lat  = round(geom.centroid.y, 6)
                lon  = round(geom.centroid.x, 6)
            except Exception:
                continue

            # build address from OSM tags
            street   = _clean(row.get("addr:street"))
            number   = _clean(row.get("addr:housenumber"))
            postcode = _clean(row.get("addr:postcode"))
            city     = _clean(row.get("addr:city")) or city_name

            address = " ".join(filter(None, [street, number])).strip()
            if postcode or city:
                address += f", {' '.join(filter(None, [postcode, city]))}"
            if not address.strip(", "):
                address = "address unknown"

            name = _clean(row.get("name")) or "unnamed"
            if str(name) == "nan" or not name:
                name = "unnamed"

            parkings.append({
                "name":          name,
                "lat":           lat,
                "lon":           lon,
                "capacity":      cap_int,
                "address":       address,
                "osm_type":      row.get("parking",   "unknown"),
                "park_and_ride": row.get("park_ride") == "yes",
                "operator":      row.get("operator",  ""),
                "city":          city_name,
            })

        parkings.sort(key=lambda x: x["capacity"], reverse=True)
        print(f"  Found {len(parkings)} parkings with capacity "
            f">= {min_capacity} in {city_name}")
        return parkings