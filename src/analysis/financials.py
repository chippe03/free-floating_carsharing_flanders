"""
Financial analysis of carsharing simulation runs.
Reads from event logs and applies a cost/revenue model to evaluate
the financial viability of each relocation policy scenario.

All monetary values in EUR.
"""

import json
from pathlib import Path
from typing import Optional
from src import config


class FinancialModel:
    """
    Configurable cost/revenue model for intercity carsharing.
    All parameters can be overridden via financials.yaml or directly.
    """

    def __init__(self, goodwill_multiplier: float = None):
        try:
            cfg = config.financials
        except:
            cfg = {}
            print("  Warning: financial parameters not found, using defaults")

        # --- revenue ---
        self.base_fare_eur        = cfg.get("base_fare_eur",        2.50)
        self.price_per_km_eur     = cfg.get("price_per_km_eur",     0.35)
        self.price_per_min_eur    = cfg.get("price_per_min_eur",    0.00)

        # --- vehicle costs ---
        self.cost_per_vehicle_per_day_eur = cfg.get(
            "cost_per_vehicle_per_day_eur", 15.00
        )
        self.fuel_cost_per_km_eur = cfg.get("cost_per_km_eur", 0.12)

        # --- worker costs ---
        self.worker_wage_day_eur   = cfg.get("worker_wage_day_eur",            20.00)
        self.worker_wage_night_eur = cfg.get("worker_wage_night_eur",          25.00)
        self.worker_km_cost_eur    = cfg.get("worker_km_cost_eur",              0.12)
        self.train_ticket_cost_eur = cfg.get("repositioning_ticket_cost_eur",   8.00)

        # --- unmet demand ---
        self.goodwill_multiplier = (goodwill_multiplier if goodwill_multiplier is not None
                                    else cfg.get("goodwill_multiplier", 0.0))

    def trip_revenue(self, distance_km: float,
                     duration_min: float = 0.0) -> float:
        """Revenue from a single completed trip."""
        return (self.base_fare_eur
                + self.price_per_km_eur  * distance_km
                + self.price_per_min_eur * duration_min)

    def relocation_cost(self, duration_min: float,
                        distance_km: float,
                        start_hour: int = 12) -> float:
        """
        Cost of one relocation.
        Uses night rate if relocation starts between 20:00 and 06:00.
        """
        is_night  = start_hour >= 20 or start_hour < 6
        wage      = self.worker_wage_night_eur if is_night \
                    else self.worker_wage_day_eur
        wage_cost = wage * (duration_min / 60)
        km_cost   = self.worker_km_cost_eur * distance_km
        return wage_cost + km_cost

    def reposition_cost(self, duration_min: float,
                        start_hour: int = 12) -> float:
        """
        Cost of one reposition.
        Uses night rate if reposition starts between 20:00 and 06:00.
        """
        is_night   = start_hour >= 20 or start_hour < 6
        wage       = self.worker_wage_night_eur if is_night \
                     else self.worker_wage_day_eur
        wage_cost  = wage * (duration_min / 60)
        train_cost = self.train_ticket_cost_eur
        return wage_cost + train_cost
    
    def vehicle_cost(self, fleet_size: int, n_days: int) -> float:
        """Total vehicle ownership cost over simulation period."""
        return self.cost_per_vehicle_per_day_eur * fleet_size * n_days


class FinancialResults:
    """
    Computes financial KPIs from a simulation event log.
    """

    def __init__(self, event_log: list[dict],
                 label:        str            = "simulation",
                 fleet_size:   Optional[int]  = None,
                 model:        FinancialModel = None):

        self.log        = event_log
        self.label      = label
        self.model      = model or FinancialModel()

        self.goodwill_multiplier = model.goodwill_multiplier

        # derive fleet size from log if not provided
        if fleet_size is not None:
            self.fleet_size = fleet_size
        else:
            vehicle_ids     = set(e["vehicle_id"] for e in self.log
                                  if e.get("vehicle_id"))
            self.fleet_size = len(vehicle_ids)

        # derive n_days from log
        days            = [e["day"] for e in self.log if "day" in e]
        self.n_days     = max(days) + 1 if days else 1

        # split log by event type
        self.trips       = [e for e in self.log
                            if e["event"] == "TRIP_COMPLETED"]
        self.unmet       = [e for e in self.log
                            if e["event"] == "TRIP_UNMET"]
        self.relocations = [e for e in self.log
                            if e["event"] == "RELOCATION_COMPLETED"]
        self.reposition  = [e for e in self.log
                            if e["event"] == "WORKER_REPOSITION_COMPLETED"]

    # ------------------------------------------------------------------ #
    #  Revenue                                                           #
    # ------------------------------------------------------------------ #

    @property
    def total_revenue(self) -> float:
        return sum(
            self.model.trip_revenue(
                e.get("distance_km", 0) or 0,
                e.get("duration_min", 0) or 0,
            )
            for e in self.trips
        )

    @property
    def avg_revenue_per_trip(self) -> float:
        if not self.trips:
            return 0.0
        return self.total_revenue / len(self.trips)

    @property
    def lost_revenue(self) -> float:
        """
        Revenue lost due to unmet demand.
        goodwill_multiplier = 0.0: no lost revenue (default)
        goodwill_multiplier = 1.0: only direct lost revenue
        goodwill_multiplier > 1.0: multiplies cost per unmet trip to
        reflect customer churn, negative reviews, and lifetime value loss.
        """
        # estimate using average revenue per served trip
        return self.avg_revenue_per_trip * len(self.unmet)

    # ------------------------------------------------------------------ #
    #  Costs                                                             #
    # ------------------------------------------------------------------ #

    @property
    def total_vehicle_cost(self) -> float:
        return self.model.vehicle_cost(self.fleet_size, self.n_days)

    @property
    def total_fuel_cost(self) -> float:
        """Fuel cost for all customer trips."""
        return sum(
            self.model.fuel_cost_per_km_eur * (e.get("distance_km", 0) or 0)
            for e in self.trips
        )
    
    @property
    def total_relocation_cost(self) -> float:
        return sum(
            self.model.relocation_cost(
                duration_min = e.get("duration_min", 0) or 0,
                distance_km  = e.get("distance_km",  0) or 0,
                start_hour   = e.get("hour", 12),      # ← from event log
            )
            for e in self.relocations
        )

    @property
    def total_repositioning_cost(self) -> float:
        return sum(
            self.model.reposition_cost(
                duration_min = e.get("duration_min", 0) or 0,
                start_hour   = e.get("hour", 12),
            )
            for e in self.reposition
        )

    @property
    def total_cost(self) -> float:
        return (self.total_vehicle_cost
                + self.total_fuel_cost
                + self.total_relocation_cost
                + self.total_repositioning_cost)

    # ------------------------------------------------------------------ #
    #  Profit                                                            #
    # ------------------------------------------------------------------ #

    @property
    def gross_profit(self) -> float:
        return self.total_revenue - self.total_cost - (self.lost_revenue * self.goodwill_multiplier)

    @property
    def gross_profit_per_day(self) -> float:
        return self.gross_profit / self.n_days

    @property
    def revenue_per_vehicle_per_day(self) -> float:
        if self.fleet_size == 0 or self.n_days == 0:
            return 0.0
        return self.total_revenue / (self.fleet_size * self.n_days)

    @property
    def cost_per_trip(self) -> float:
        if not self.trips:
            return 0.0
        return self.total_cost / len(self.trips)

    @property
    def breakeven_trips_per_day(self) -> float:
        """How many trips per day needed to cover all costs."""
        if self.avg_revenue_per_trip == 0:
            return float("inf")
        daily_cost = self.total_cost / self.n_days
        return daily_cost / self.avg_revenue_per_trip

    # ------------------------------------------------------------------ #
    #  Summary                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        return {
            "label":                      self.label,
            "n_days":                     self.n_days,
            "fleet_size":                 self.fleet_size,

            # volume
            "trips_served":               len(self.trips),
            "trips_unmet":                len(self.unmet),
            "trips_per_day":              round(len(self.trips) / self.n_days, 1),

            # revenue
            "total_revenue_eur":          round(self.total_revenue, 2),
            "avg_revenue_per_trip_eur":   round(self.avg_revenue_per_trip, 2),
            "lost_revenue_eur":           round(self.lost_revenue, 2),

            # costs
            "vehicle_cost_eur":           round(self.total_vehicle_cost, 2),
            "fuel_cost_eur":              round(self.total_fuel_cost, 2),
            "relocation_cost_eur":        round(self.total_relocation_cost, 2),
            "repositioning_cost_eur":     round(self.total_repositioning_cost, 2),
            "total_cost_eur":             round(self.total_cost, 2),

            # profit
            "gross_profit_eur":           round(self.gross_profit, 2),
            "gross_profit_per_day_eur":   round(self.gross_profit_per_day, 2),
            "revenue_per_vehicle_per_day":round(self.revenue_per_vehicle_per_day, 2),
            "cost_per_trip_eur":          round(self.cost_per_trip, 2),
            "breakeven_trips_per_day":    round(self.breakeven_trips_per_day, 1),
        }

    def print_report(self):
        s = self.summary()
        w = 38   # column width

        def fmt(amount: float) -> str:
            return f"€{amount:>14,.2f}"

        print(f"\n{'='*60}")
        print(f"  FINANCIAL REPORT  —  {self.label.upper()}")
        print(f"  {self.n_days} days  |  fleet: {self.fleet_size} vehicles")
        print(f"{'='*60}")

        print(f"\n  VOLUME")
        print(f"  {'Trips served':<{w}} {s['trips_served']:>10}")
        print(f"  {'Trips unmet':<{w}} {s['trips_unmet']:>10}")
        print(f"  {'Trips per day':<{w}} {s['trips_per_day']:>10.1f}")

        print(f"\n  REVENUE")
        print(f"  {'Total revenue':<{w}} {fmt(s['total_revenue_eur'])}")
        print(f"  {'Avg revenue per trip':<{w}} {fmt(s['avg_revenue_per_trip_eur'])}")
        print(f"  {'Lost revenue (unmet demand)':<{w}} {fmt(s['lost_revenue_eur'])}")

        print(f"\n  COSTS")
        print(f"  {'Vehicle ownership':<{w}} {fmt(s['vehicle_cost_eur'])}")
        print(f"  {'Fuel (customer trips)':<{w}} {fmt(s['fuel_cost_eur'])}")
        print(f"  {'Relocation (worker wages + km)':<{w}} {fmt(s['relocation_cost_eur'])}")
        print(f"  {'Repositioning (worker travel)':<{w}} {fmt(s['repositioning_cost_eur'])}")
        print(f"  {'─'*55}")
        print(f"  {'Total cost':<{w}} {fmt(s['total_cost_eur'])}")

        print(f"\n  PROFIT")
        print(f"  {'Gross profit':<{w}} {fmt(s['gross_profit_eur'])}"
            f"  {'← LOSS' if s['gross_profit_eur'] < 0 else '← PROFIT'}")
        print(f"  {'Gross profit per day':<{w}} {fmt(s['gross_profit_per_day_eur'])}")
        print(f"  {'Revenue per vehicle per day':<{w}} {fmt(s['revenue_per_vehicle_per_day'])}")
        print(f"  {'Cost per served trip':<{w}} {fmt(s['cost_per_trip_eur'])}")
        print(f"  {'Breakeven trips/day needed':<{w}} {s['breakeven_trips_per_day']:>10.1f}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    #  Load from file                                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_log(cls, filename: str, label: str = None,
                 fleet_size: int = None,
                 model: FinancialModel = None) -> "FinancialResults":
        path = Path(f"data/results/{filename}")
        with open(path) as f:
            log = json.load(f)
        return cls(
            event_log  = log,
            label      = label or filename.replace("_events.json", ""),
            fleet_size = fleet_size,
            model      = model,
        )

    # ------------------------------------------------------------------ #
    #  Compare multiple scenarios                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compare(results_list: list["FinancialResults"],
                save_path: str = None):
        """Side-by-side financial comparison of multiple scenarios."""
        col    = 16
        keys   = [
            "trips_served",
            "trips_unmet",
            "total_revenue_eur",
            "lost_revenue_eur",
            "vehicle_cost_eur",
            "fuel_cost_eur",
            "relocation_cost_eur",
            "repositioning_cost_eur",
            "total_cost_eur",
            "gross_profit_eur",
            "gross_profit_per_day_eur",
            "revenue_per_vehicle_per_day",
            "breakeven_trips_per_day",
        ]
        labels = [r.label for r in results_list]

        lines  = []
        lines.append(f"\n{'='*(12 + col * len(labels))}")
        lines.append(f"  FINANCIAL COMPARISON")
        lines.append(f"{'='*(12 + col * len(labels))}")
        lines.append(
            f"  {'':35}" + "".join(f"{l:>{col}}" for l in labels)
        )
        lines.append(f"  {'-'*(35 + col * len(labels))}")

        for key in keys:
            row = f"  {key:<35}"
            for r in results_list:
                val = r.summary()[key]
                if isinstance(val, float) and "eur" in key:
                    row += f"€{val:>{col-1},.2f}"
                elif isinstance(val, float):
                    row += f"{val:>{col}.2f}"
                else:
                    row += f"{str(val):>{col}}"
            lines.append(row)

        lines.append(f"{'='*(12 + col * len(labels))}\n")

        for line in lines:
            print(line)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            Path(save_path).write_text("\n".join(lines))
            print(f"  Financial comparison saved to {save_path}")


if __name__ == "__main__":
    model = FinancialModel()

    results = [
        FinancialResults.from_log("baseline_events.json",  label="baseline",  model=model),
        FinancialResults.from_log("reactive_events.json",  label="reactive",  model=model),
        FinancialResults.from_log("proactive_events.json", label="proactive", model=model),
        FinancialResults.from_log("nightly_events.json",   label="nightly",   model=model),
    ]

    for r in results:
        r.print_report()

    FinancialResults.compare(
        results,
        save_path="data/metrics/financial_comparison.txt"
    )