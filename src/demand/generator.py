import numpy as np
from src.demand.od_matrix import build_od_matrix, get_multiplier, build_origin_weights
from src.demand.od_matrix import get_od_matrix_cached, get_multiplier_cached
from src.entities.trip import Trip
from src import config


class DemandGenerator:
    """
    Generates a stream of Trip requests using a Poisson process.

    The base rate (trips per hour across the whole network) is scaled by:
      - the OD matrix            → which city pair is chosen (time-specific)
      - the hourly pattern       → how busy this hour is
      - the city's demand weight → how much demand originates here
    """

    def __init__(self, base_rate: float = None, is_weekend: bool = False, seed: int = 123):
        np.random.seed(seed)
        self.is_weekend = is_weekend

        demand_weights  = build_origin_weights(is_weekend=is_weekend)

        self.city_names = list(demand_weights.keys())
        self.origin_weights = [demand_weights[c] for c in self.city_names]

        # pre-build OD matrices per time window to avoid reloading CSV every trip
        self._od_cache: dict[int, dict] = {}

        if base_rate is not None:
            self.base_rate = base_rate
        else:
            sim_cfg    = config.simulation["demand"]
            rate_key   = "base_rate_weekend" if is_weekend else "base_rate_weekday"
            self.base_rate = sim_cfg[rate_key]

    def _get_od_matrix(self, hour: int) -> dict:
        return get_od_matrix_cached(self.is_weekend, hour)

    def _current_rate(self, sim_minute: float) -> float:
        hour       = int(sim_minute // 60) % 24
        multiplier = get_multiplier_cached(hour, self.is_weekend)
        return (self.base_rate * multiplier) / 60

    def _sample_origin(self) -> str:
        """Sample an origin city weighted by demand_weight."""
        weights = np.array(self.origin_weights)
        probs   = weights / weights.sum()
        return np.random.choice(self.city_names, p=probs)

    def _sample_destination(self, origin: str, hour: int) -> str:
        """Sample destination from time-specific OD matrix."""
        matrix = self._get_od_matrix(hour)
        row    = matrix.get(origin, {})

        dests  = list(row.keys())
        probs  = np.array(list(row.values()))

        # guard against missing origin or all-zero row
        if len(dests) == 0 or probs.sum() == 0:
            # fallback: uniform over other cities
            dests  = [c for c in self.city_names if c != origin]
            probs  = np.ones(len(dests))

        probs = probs / probs.sum()
        return np.random.choice(dests, p=probs)

    def generate_day(self, day: int = 0) -> list[Trip]:
        """Generate all trips for one simulated day."""
        trips      = []
        sim_minute = day * 24 * 60
        end_minute = sim_minute + 24 * 60

        while sim_minute < end_minute:
            hour = int((sim_minute % (24 * 60)) // 60)
            rate = self._current_rate(sim_minute)

            inter_arrival = np.random.exponential(1 / rate) if rate > 0 else 60
            sim_minute   += inter_arrival

            if sim_minute >= end_minute:
                break

            origin      = self._sample_origin()
            destination = self._sample_destination(origin, hour)  # ← pass hour

            trips.append(Trip(
                origin       = origin,
                destination  = destination,
                request_time = round(sim_minute, 2)
            ))

        return trips

    def generate_period(self, n_days: int = 7) -> list[Trip]:
        """Generate trips across multiple days."""
        all_trips = []
        for day in range(n_days):
            all_trips.extend(self.generate_day(day=day))
        return all_trips

    def summary(self, trips: list[Trip]) -> dict:
        from collections import Counter
        origins      = Counter(t.origin for t in trips)
        destinations = Counter(t.destination for t in trips)
        pairs        = Counter(f"{t.origin}→{t.destination}" for t in trips)
        return {
            "total_trips":    len(trips),
            "by_origin":      dict(origins.most_common()),
            "by_destination": dict(destinations.most_common()),
            "top_pairs":      dict(pairs.most_common(10)),
        }


if __name__ == "__main__":
    gen   = DemandGenerator(is_weekend=True)
    trips = gen.generate_day()
    print(f"Generated {len(trips)} trips for one day\n")
    stats = gen.summary(trips)
    print("Top OD pairs:")
    for pair, count in stats["top_pairs"].items():
        print(f"  {pair:<25} {count} trips")