"""
Nightly rebalancing policy.

At trigger_hour, an LP determines the optimal decisions to
bring the fleet as close as possible to the initial distribution.
"""

import math
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

from pulp import (
    LpProblem, LpMinimize, LpVariable, LpInteger,
    lpSum, value, PULP_CBC_CMD, LpStatus
)

from src.relocation.base_policy import BaseRelocationPolicy
from src.simulation.state import SimulationState
from src import config


class NightlyRebalancePolicy(BaseRelocationPolicy):

    def __init__(
            self,
            check_interval_minutes: int = None,
            trigger_hour:           int = None,
            penalty_unmet:        float = None,
            cost_per_relocation:  float = None,
            verbose:               bool = False,
    ):
        """
        check_interval_minutes: how often the policy is called
        trigger_hour:           hour of day (0-23) at which rebalancing fires

        penalty_unmet:          cost per unmet trip (average revenue from trips)
        cost_per_relocation:    cost per relocation (average relocation cost)
        """
        super().__init__(name="nightly")
        self.persist_pending: bool = True
        self.sim_cfg = config.simulation

        self.trigger_hour           = (trigger_hour
                                       or self.sim_cfg["relocation"]["trigger_hour"])
        self.check_interval_minutes = (check_interval_minutes
                                       or self.sim_cfg["relocation"]["check_interval_minutes"])
        
        self.penalty_unmet          = (penalty_unmet
                                       or self.sim_cfg["relocation"]["penalty_unmet"])
        self.cost_per_relocation    = (cost_per_relocation
                                       or self.sim_cfg["relocation"]["cost_per_relocation"])
        
        self.verbose = verbose

        self._last_triggered_day   = -1
        self._workers_returned_day = -1

        # diagnostics
        self.nightly_log: list = []


    # ------------------------------------------------------------------ #
    #  Main entry point                                                  #
    # ------------------------------------------------------------------ #

    def relocate(self, state: SimulationState) -> list[tuple[str, str]]:
        current_minute = state.sim_time % (24 * 60)
        current_hour   = int(current_minute // 60)
        current_day    = int(state.sim_time // (24 * 60))

        # only fire at trigger_hour
        if (current_hour != self.trigger_hour
                or current_day == self._last_triggered_day):
            return []

        self._last_triggered_day = current_day
        cities = list(state.zones.keys())

        if self.verbose:
            print(f"\n  [NIGHTLY REBALANCE] day={current_day+1} "
                  f"t={state.sim_time:.0f} ({current_hour:02d}:00)")

        # exclude vehicles already mid-relocation
        in_progress = {
            r.vehicle_id
            for r in state.relocations.values()
            if not r.completed
        }

        # --- step 1: get ideal state ---
        ideal = self._get_ideal(state, cities)
        if not ideal:
            print("  ⚠️ Couldn't get ideal state - skipping rebalance")
            return []

        # --- step 2: compute target distribution via LP ---
        x = self._target_lp(state, cities, ideal)

        if x is None:
            # print("  ⚠️ LP failed - falling back to proportional distribution")
            # x = self._proportional_fallback(state, cities, ideal)
            print("  ⚠️ LP failed - no redistribution")
            return []
        
        if self.verbose:
            print(f"  supply:    { {c: state.zones[c].available_count() for c in cities} }")
            print(f"  ideal:     { {c: round(ideal.get(c, 0), 1) for c in cities} }")
            print(f"  target:    {x}")

        # --- step 3: derive moves from current supply → target ---
        moves = self._compute_moves(state, cities, x)

        if self.verbose:
            print(f"  moves:     {len(moves)} relocations planned")

        # --- step 4: convert to (vehicle_id, destination) decisions ---
        decisions = self._extract_decisions(state, moves, in_progress)

        start_dt    = datetime.strptime(self.sim_cfg["time"]["start_datetime"], "%Y-%m-%d %H:%M")
        actual_date = start_dt + timedelta(days=current_day)
        is_weekend  = actual_date.weekday() >= 5

        self.nightly_log.append({
            "day":             current_day,
            "is_weekend":      is_weekend,
            "n_moves_planned": len(moves),
            "n_decisions":     len(decisions),
            "x":               x,
        })

        if self.verbose:
            print(f"  → {len(decisions)} decisions returned\n")

        return decisions


    # ------------------------------------------------------------------ #
    #  Ideal state                                                       #
    # ------------------------------------------------------------------ #

    def _get_ideal(self, state: SimulationState, cities: list) -> dict:
        """Target distribution = initial fleet deployment."""
        return {
            city: state.initial_distribution.get(city, 0)
            for city in cities
        }


    # ------------------------------------------------------------------ #
    #  LP: target distribution                                           #
    # ------------------------------------------------------------------ #

    def _target_lp(self, state: SimulationState,
                   cities: list, ideal: dict
    ) -> Optional[dict]:
        """
        LP to find the cost-optimal target vehicle counts per city.
 
        Variables:
            x[s]   = target = vehicles city s should have after rebalancing  (integer ≥ 0)
            r[s,d] = flow   = vehicles relocated from s to d                 (integer ≥ 0)
            u[s]   = slack  = vehicles short of ideal at city s           (continuous ≥ 0)
 
        Objective:
            min  penalty_unmet × Σ_s u[s]
                  + cost_per_relocation × Σ_{s≠d} r[s,d]
 
        Constraints:
            1. Fleet conservation:    Σ_s x[s] = total_available
            2. Flow conservation:     x[s] = current[s] + Σ_d r[d,s] - Σ_d r[s,d]   ∀s
            3. Shortfall:             u[s] ≥ ideal[s] - x[s]        ∀s
            4. Worker capacity:       Σ_{s,d} r[s,d] ≤ n_workers × window_hours
                                        where window_hours = 6 - trigger_hour
            5. Supply limit:          Σ_d r[s,d] ≤ current[s]       ∀s
        """
        current         = {s: state.zones[s].available_count() for s in cities}
        total_available = sum(current.values())

        n_workers       = state.worker_pool.idle_count()
        window_hours = max(1, 6 - self.trigger_hour)

        prob = LpProblem("nightly_target", LpMinimize)


        # --- variables ---

        # target
        x = {
            s: LpVariable(f"x_{s}", lowBound=0, cat=LpInteger)
            for s in cities
        }

        # flow: relocations from s to d
        r = {
            (s, d): LpVariable(f"r_{s}_{d}", lowBound=0, cat=LpInteger)
            for s in cities for d in cities if s != d
        }
        
        # slack: unmet demand
        u = {
            s: LpVariable(f"u_{s}", lowBound=0)
            for s in cities
        }
        

        # --- objective ---
        prob += (
            self.penalty_unmet         * lpSum(u[s] for s in cities)
            + self.cost_per_relocation * lpSum(
                r[(s, d)]
                for s in cities for d in cities if s != d
            )
        )


        # --- constraints ---

        # 1. fleet conservation
        prob += lpSum(x[s] for s in cities) == total_available

        # 4. worker capacity
        prob += (
            lpSum(r[(s, d)] for s in cities for d in cities if s != d)
            <= n_workers * window_hours
        )

        for s in cities:
            # 2. flow conservation
            prob += (
                x[s] == current[s]
                      + lpSum(r[(d, s)] for d in cities if d != s)
                      - lpSum(r[(s, d)] for d in cities if d != s)
            )

            # 3. shortfall
            prob += u[s] >= ideal.get(s, 0) - x[s]

            # 5. supply limit
            prob += (
                lpSum(r[(s, d)] for d in cities if d != s)
                <= current[s]
            )


        # --- solve ---

        solver = PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)

        if LpStatus[prob.status] not in ("Optimal", "Feasible"):
            print(f"  LP status: {LpStatus[prob.status]}")
            return None

        return {s: max(0, int(round(value(x[s]) or 0))) for s in cities}
 

    # # ------------------------------------------------------------------ #
    # #  Fallback: proportional  distribution                              #
    # # ------------------------------------------------------------------ #

    # def _proportional_fallback(
    #     self, state: SimulationState, cities: list, ideal: dict
    # ) -> dict:
    #     """Distribute the available fleet proportionally to the ideal state."""
    #     total      = sum(state.zones[s].available_count() for s in cities)
    #     total_ideal = sum(ideal.values())
 
    #     if total_ideal == 0:
    #         per_city = total // len(cities)
    #         return {s: per_city for s in cities}
 
    #     shares = {s: math.floor(ideal[s] / total_ideal * total) for s in cities}
 
    #     # give remainder to cities with the largest fractional shortfall
    #     remainder = total - sum(shares.values())
    #     order = sorted(
    #         cities,
    #         key=lambda s: (ideal[s] / total_ideal * total) - shares[s],
    #         reverse=True,
    #     )
    #     for s in order[:remainder]:
    #         shares[s] += 1
 
    #     return shares


    # ------------------------------------------------------------------ #
    #  Derive moves from supply → target                                 #
    # ------------------------------------------------------------------ #

    def _compute_moves(
        self, state: SimulationState, cities: list, x: dict
    ) -> list[tuple[str, str]]:
        """
        Greedy matching of surplus cities to deficit cities.
        Returns list of (origin, destination) pairs.
        """
        current = {s: state.zones[s].available_count() for s in cities}
        surplus = {s: max(0, current[s] - x[s]) for s in cities}
        deficit = {s: max(0, x[s] - current[s]) for s in cities}

        def travel_cost(s, d):
            try:
                travel = state.get_travel(s, d, state.sim_time)
                return travel["duration_min"] if travel else float("inf")
            except Exception:
                return float("inf")
 
        # sort candidate pairs by travel cost before matching
        pairs = sorted(
            [(s, d) for s in cities for d in cities
             if s != d and surplus[s] > 0 and deficit[d] > 0],
            key=lambda sd: travel_cost(sd[0], sd[1])
        )
 
        moves = []
        for src, dst in pairs:
            n = min(surplus[src], deficit[dst])
            if n > 0:
                moves.extend([(src, dst)] * n)
                surplus[src] -= n
                deficit[dst] -= n
 
        return moves


    # ------------------------------------------------------------------ #
    #  Extract decisions                                                 #
    # ------------------------------------------------------------------ #

    def _extract_decisions(
        self, state: SimulationState,
        moves: list[tuple[str, str]],
        in_progress: set,
    ) -> list[tuple[str, str]]:
        """
        Convert (origin, destination) moves to (vehicle_id, destination)
        decisions, excluding vehicles already being moved.
        """
        decisions = []
        claimed   = set() | in_progress

        for origin, destination in moves:
            available = [
                v for v in state.zones[origin].get_available_vehicles()
                if v.id not in claimed
            ]
            if not available:
                continue
            vehicle = available[0]
            claimed.add(vehicle.id)
            decisions.append((vehicle.id, destination))

        return decisions
    

    # ------------------------------------------------------------------ #
    #  Workers return home at the end                                    #
    # ------------------------------------------------------------------ #

    def get_worker_returns(self, state):
        current_day = int(state.sim_time // (24 * 60))
        if current_day != self._last_triggered_day:
            return []
        if current_day == self._workers_returned_day:
            return []
        current_hour = int((state.sim_time % (24 * 60)) // 60)
        if current_hour < 6:
            return []
        active = [r for r in state.relocations.values() if not r.completed]
        if active:
            return []

        self._workers_returned_day = current_day
        returns = []
        for worker in state.worker_pool.get_available():
            if worker.city != worker.home_city:
                returns.append((worker.id, worker.home_city))
        return returns


    # ------------------------------------------------------------------ #
    #  Diagnostics                                                       #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        if not self.nightly_log:
            return {}
        return {
            "n_nights":            len(self.nightly_log),
            "total_moves_planned": sum(e["n_moves_planned"] for e in self.nightly_log),
            "total_decisions":     sum(e["n_decisions"]     for e in self.nightly_log),
            "weekday_nights":      sum(1 for e in self.nightly_log if not e["is_weekend"]),
            "weekend_nights":      sum(1 for e in self.nightly_log if e["is_weekend"]),
        }
