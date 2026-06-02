import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

from matplotlib import lines
from src import config


class Metrics:
    """Compute KPIs from a simulation event log."""

    def __init__(self, event_log: List[dict], label: str = "simulation",
                 n_days: int = None, fleet_size: int = None):
        self.log   = event_log
        self.label = label
        self._n_days = n_days
        self._fleet_size = fleet_size

        self.cities = config.cities["cities"].keys()

        # split log by event type
        self.assigned   = [e for e in self.log if e["event"] == "TRIP_ASSIGNED"]
        self.completed  = [e for e in self.log if e["event"] == "TRIP_COMPLETED"]
        self.unmet      = [e for e in self.log if e["event"] == "TRIP_UNMET"]
        self.relocated  = [e for e in self.log if e["event"] == "RELOCATION_COMPLETED"]

    # ------------------------------------------------------------------ #
    #  Load                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_log(cls, filename: str, label: str = None,
                 n_days: int = None, fleet_size: int = None):
        """Load from a saved event log in data/results/."""
        path = Path("./data/results") / filename
        with open(path) as f:
            log = json.load(f)
        return cls(log,
                   label=label or filename.replace("_events.json", ""),
                   n_days=n_days,
                   fleet_size=fleet_size)

    # ------------------------------------------------------------------ #
    #  Core KPIs                                                         #
    # ------------------------------------------------------------------ #

    @property
    def n_days(self):
        if self._n_days is not None:
            return self._n_days
        if not self.log:
            return None
        # derive from max day in log
        days = [e["day"] for e in self.log if "day" in e]
        return max(days) + 1 if days else None

    @property
    def fleet_size(self):
        if self._fleet_size is not None:
            return self._fleet_size
        # derive from unique vehicle IDs in log
        vehicle_ids = set(e["vehicle_id"] for e in self.log
                        if e.get("vehicle_id"))
        return len(vehicle_ids) if vehicle_ids else None

    @property
    def total_requests(self) -> int:
        return len(self.assigned) + len(self.unmet)

    @property
    def total_served(self) -> int:
        return len(self.assigned)
    
    @property
    def total_distance_km(self) -> float:
        return sum(e["distance_km"] for e in self.assigned if e["distance_km"])

    @property
    def avg_distance_km(self) -> float:
        valid = [e["distance_km"] for e in self.assigned if e["distance_km"]]
        return (sum(valid) / len(valid)) if valid else 0.0
    
    @property
    def total_duration_min(self) -> float:
        return sum(e["duration_min"] for e in self.assigned if e["duration_min"])

    @property
    def avg_duration_min(self) -> float:
        valid = [e["duration_min"] for e in self.assigned if e["duration_min"]]
        return (sum(valid) / len(valid)) if valid else 0.0

    @property
    def total_unmet(self) -> int:
        return len(self.unmet)

    @property
    def unmet_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_unmet / self.total_requests

    @property
    def total_relocations(self) -> int:
        return len(self.relocated)

    # ------------------------------------------------------------------ #
    #  Demand breakdown                                                  #
    # ------------------------------------------------------------------ #

    def unmet_by_city(self) -> Dict[str, dict]:
        """Unmet demand per origin city."""
        demand  = defaultdict(int)
        unmet   = defaultdict(int)

        for e in self.assigned + self.unmet:
            demand[e["origin"]] += 1
        for e in self.unmet:
            unmet[e["origin"]] += 1

        return {
            city: {
                "demand":     demand[city],
                "unmet":      unmet[city],
                "unmet_rate": round(unmet[city] / demand[city], 3)
                              if demand[city] > 0 else 0.0
            }
            for city in self.cities
        }

    def demand_by_hour(self) -> Dict[int, dict]:
        """Trip requests and unmet count per hour of day."""
        demand = defaultdict(int)
        unmet  = defaultdict(int)

        for e in self.assigned + self.unmet:
            hour = int(e["hour"])
            demand[hour] += 1
        for e in self.unmet:
            hour = int(e["hour"])
            unmet[hour] += 1

        return {
            hour: {
                "demand": demand[hour],
                "unmet":  unmet[hour],
                "unmet_rate": round(unmet[hour] / demand[hour], 3)
                              if demand[hour] > 0 else 0.0
            }
            for hour in sorted(demand.keys())
        }

    def demand_by_weekday(self) -> Dict[str, dict]:
        """Trip requests and unmet count per weekday."""
        demand = defaultdict(int)
        unmet  = defaultdict(int)

        for e in self.assigned + self.unmet:
            demand[e["weekday"]] += 1
        for e in self.unmet:
            unmet[e["weekday"]] += 1

        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        return {
            day: {
                "demand": demand[day],
                "unmet":  unmet[day],
                "unmet_rate": round(unmet[day] / demand[day], 3)
                              if demand[day] > 0 else 0.0
            }
            for day in order if day in demand
        }

    # ------------------------------------------------------------------ #
    #  OD pair analysis                                                  #
    # ------------------------------------------------------------------ #

    def top_od_pairs(self, n: int = 10) -> List[dict]:
        """Most frequent origin-destination pairs."""
        counts = defaultdict(int)
        for e in self.assigned + self.unmet:
            counts[(e["origin"], e["destination"])] += 1

        sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [
            {"origin": o, "destination": d, "trips": c}
            for (o, d), c in sorted_pairs[:n]
        ]

    # ------------------------------------------------------------------ #
    #  Relocation KPIs                                                   #
    # ------------------------------------------------------------------ #

    def relocation_stats(self) -> dict:
        if not self.relocated:
            return {"total": 0, "total_km": 0, "total_min": 0}

        distances = [e.get("distance_km", 0) for e in self.relocated]
        durations = [e["duration_min"] for e in self.relocated if e.get("duration_min")]
        return {
            "total":    len(distances),
            "total_km": round(sum(distances), 1),
            "total_min": round(sum(durations), 1),
        }
    
    def worker_stats(self) -> dict:
        """Worker utilization stats"""
        started   = [e for e in self.log if e["event"] == "RELOCATION_STARTED"]
        completed = [e for e in self.log if e["event"] == "RELOCATION_COMPLETED"]
        
        pending   = [e for e in self.log if e["event"] == "RELOCATION_PENDING"]
        forgotten = [e for e in self.log if e["event"] == "RELOCATIONS_FORGOTTEN"]
        total_forgotten = sum(e.get("count", 1) for e in forgotten)

        worker_summary = next((e for e in self.log if e["event"] == "SIMULATION_SUMMARY"), {})

        return {
            "relocations_executed":  len(completed),
            "relocations_queued":    len(pending),
            "relocations_forgotten": total_forgotten,
            "peak_workers_active":   worker_summary.get("peak_workers", 0),
            "total_workers":         worker_summary.get("total_workers", 0)
        }
        
    # ------------------------------------------------------------------ #
    #  Summary dict (used for comparison)                                #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        rel = self.relocation_stats()
        wor = self.worker_stats()

        return {
            "label":             self.label,
            "total_requests":    self.total_requests,
            "total_served":      self.total_served,
            "total_unmet":       self.total_unmet,
            "unmet_rate":        f"{self.unmet_rate:.1%}",
            "total_distance_km": round(self.total_distance_km, 1),
            "avg_distance_km":   round(self.avg_distance_km, 1),
            "total_duration_min":round(self.total_duration_min, 1),
            "avg_duration_min":  round(self.avg_duration_min, 1),

            "total_relocations": rel["total"],
            "relocation_km":     rel["total_km"],
            "relocation_min":    round(rel.get("total_min", 0), 1),

            "relocations_queued": wor["relocations_queued"],
            "relocations_forgotten": wor["relocations_forgotten"],
            "peak_workers_active": wor["peak_workers_active"],
            "total_workers": wor["total_workers"],
        }

    # ------------------------------------------------------------------ #
    #  Print reports                                                     #
    # ------------------------------------------------------------------ #

    def print_report(self):
        s = self.summary()

        print(f"\n{'='*55}")
        print(f"  METRICS REPORT  -  {self.label.upper()}")
        print(f"{'='*55}")
        if self.n_days:
            print(f"  Simulation days:       {self.n_days}")
        if self.fleet_size:
            print(f"  Fleet size:            {self.fleet_size} vehicles")
        print(f"  Total requests:        {s['total_requests']}")
        print(f"  Served:                {s['total_served']}")
        print(f"  Unmet:                 {s['total_unmet']}  ({s['unmet_rate']})")
        print()
        print(f"  Total km driven:       {s['total_distance_km']} km")
        print(f"  Avg trip distance:     {s['avg_distance_km']} km")
        print(f"  Total duration:        {s['total_duration_min']} min")
        print(f"  Avg trip duration:     {s['avg_duration_min']} min")
        print()
        print(f"  Total workers:         {s['total_workers']}")
        print(f"  Peak workers:          {s['peak_workers_active']}")
        print(f"  Relocations:           {s['total_relocations']}")
        print(f"  Relocation km:         {s['relocation_km']} km")
        print(f"  Relocation min:        {s['relocation_min']} min")
        print(f"  Relocations queued:    {s['relocations_queued']}")
        print(f"  Relocations forgotten: {s['relocations_forgotten']}")


        print(f"\n  Unmet demand by city:")
        for city, data in self.unmet_by_city().items():
            rate = data['unmet_rate']
            bar  = "█" * int(rate * 40)
            print(f"    {city:<12} {bar:<20} {rate:.1%}  "
                  f"({data['unmet']}/{data['demand']})")

        print(f"\n  Unmet demand by hour:")
        for hour, data in self.demand_by_hour().items():
            rate = data['unmet_rate']
            bar  = "█" * int(rate * 40)
            print(f"    {hour:02d}:00  {bar:<20} {rate:.1%}")

        print(f"\n  Unmet demand by day:")
        for day, data in self.demand_by_weekday().items():
            print(f"    {day:<12}  {data['unmet']:>4} unmet / "
                  f"{data['demand']:>4} total  ({data['unmet_rate']:.1%})")

        print(f"\n  Top OD pairs:")
        for pair in self.top_od_pairs(5):
            print(f"    {pair['origin']:<12} → {pair['destination']:<12} "
                  f"{pair['trips']:>4} trips")

        print(f"{'='*55}\n")

    # ------------------------------------------------------------------ #
    #  Compare runs                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare(metrics_list: List["Metrics"], filename: str = None):
        """Side-by-side comparison of multiple simulation runs."""
        col = 16
        keys = [
            "total_requests",
            "total_served",
            "total_unmet",
            "unmet_rate",
            "total_distance_km",
            "avg_distance_km",
            "total_duration_min",
            "avg_duration_min",
            "total_relocations",
            "relocation_km",
            "relocation_min",
            ]

        labels = [m.label for m in metrics_list]

        lines = []
        lines.append(f"\n{'='*(32 + col * len(labels))}")
        lines.append(f"  COMPARISON")
        lines.append(f"{'='*(32 + col * len(labels))}")
        lines.append(f"  {'':30}" + "".join(f"{l:>{col}}" for l in labels))
        lines.append(f"  {'-'*(30 + col * len(labels))}")

        for key in keys:
            row = f"  {key:<30}"
            for m in metrics_list:
                val = m.summary()[key]
                row += f"{str(val):>{col}}"
            lines.append(row)

        lines.append(f"{'='*(32 + col * len(labels))}\n")

        # print to console
        for line in lines:
            print(line)

        # save only if filename provided
        if filename:
            Metrics.save_comparison(metrics_list, lines, filename)
    
    @staticmethod
    def save_comparison(metrics_list: List["Metrics"], lines: list[str], filename: str):
        output_dir = Path("./data/metrics")
        output_dir.mkdir(parents=True, exist_ok=True)

        # txt
        txt_path = output_dir / f"{filename}.txt"
        txt_path.write_text("\n".join(lines))
        print(f"\n  Comparison saved to {txt_path.name}")

        # json
        json_path = output_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "filename": filename,
                    "policies": [m.summary() for m in metrics_list],
                },
                f,
                indent=2
            )
        print(f"  Comparison saved to {json_path.name}")


if __name__ == "__main__":
    m = Metrics.from_log("baseline_events.json", label="baseline")
    m.print_report()