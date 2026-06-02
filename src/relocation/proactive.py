"""
Proactive (MILP-based) relocation policy.

At each check interval, predicts an ideal state to be achieved
at the end of planning_period_minutes to satisfy predicted demand
over planning_period_minutes + lookahead_period_minutes.

An MILP computes relocation decisions that minimize unmet demand
and relocation cost.
"""

import math
import time as time_module
from collections import defaultdict
from datetime import datetime, timedelta

from pulp import (
    LpProblem, LpMinimize, LpVariable, LpContinuous,
    lpSum, value, PULP_CBC_CMD, LpStatus
)

from src.relocation.base_policy import BaseRelocationPolicy
from src.simulation.state import SimulationState
from src import config


class ProactivePolicy(BaseRelocationPolicy):
    
    def __init__(
            self,
            check_interval_minutes:   int = None,
            planning_period_minutes:  int = None,
            lookahead_period_minutes: int = None,
            time_step_minutes:        int = None,
            
            penalty_unmet:            float = None,
            cost_per_relocation:      float = None,
            
            solver_time_limit_sec:    int = None,
            verbose:                  bool = False,
    ):
        """
        check_interval_minutes:   how often the policy is called
        planning_period_minutes:  T - MILP optimizes over this window; relocation happens during T
        lookahead_period_minutes: L - ideal state is computed for the end of T, to meet as much demand from L as possible
        time_step_minutes:        Δt - discrete time step within the MILP horizon

        penalty_unmet:            cost per unmet trip (average revenue from trips)
        cost_per_relocation:      cost per relocation (average relocation cost)

        solver_time_limit_sec:    max seconds for MILP solver per interval
        """
        super().__init__(name="proactive")
        self.sim_cfg = config.simulation

        self.check_interval_minutes   = (check_interval_minutes
                                         or self.sim_cfg["relocation"]["check_interval_minutes"])
        self.planning_period_minutes  = (planning_period_minutes
                                         or self.sim_cfg["relocation"]["planning_period_minutes"])
        self.lookahead_period_minutes = (lookahead_period_minutes
                                         or self.sim_cfg["relocation"]["lookahead_period_minutes"])
        self.time_step_minutes        = (time_step_minutes
                                         or self.sim_cfg["relocation"]["time_step_minutes"])
        self.penalty_unmet            = (penalty_unmet
                                         or self.sim_cfg["relocation"]["penalty_unmet"])
        self.cost_per_relocation      = (cost_per_relocation
                                         or self.sim_cfg["relocation"]["cost_per_relocation"])
        self.solver_time_limit_sec    = (solver_time_limit_sec
                                         or self.sim_cfg["relocation"]["solver_time_limit_sec"])
        self.verbose = verbose

        # history log
        self.history:             list[dict] = []
        self._history_by_weekday: dict       = defaultdict(list)

        # diagnostics
        self._solve_times:    list[float] = []
        self._solve_statuses: list[str]   = []


    # ------------------------------------------------------------------ #
    #  History                                                           #
    # ------------------------------------------------------------------ #

    def load_history(self, event_log: list[dict]):
        self.history = [e for e in event_log if e["event"] == "TRIP_ASSIGNED"]
        self._reindex()
        print(f"\n  Proactive policy: loaded {len(self.history)} historical trips\n")


    def update_history(self, event_log: list[dict], sim_time: float):
        interval_start = sim_time - self.check_interval_minutes
        new_events = [
            e for e in event_log
            if e["event"] == "TRIP_ASSIGNED"
            and e["time"] >= interval_start
            and e["time"] < sim_time
        ]
        self.history.extend(new_events)
        self._reindex()


    def _reindex(self):
        self._history_by_weekday = defaultdict(list)
        for e in self.history:
            self._history_by_weekday[e["weekday"]].append(e)


    # ------------------------------------------------------------------ #
    #  Main entry point                                                  #
    # ------------------------------------------------------------------ #

    def relocate(self, state: SimulationState) -> list[tuple[str, str]]:
        if not self.history:
            return self._fallback_reactive(state)
        cities    = list(state.zones.keys())

        in_progress = {
            r.vehicle_id
            for r in state.relocations.values()
            if not r.completed
        }
    
        # time steps over planning_period_minutes
        # predict demand over planning_period_minutes + lookahead_period_minutes
        planning_steps = list(range(0,
                                    self.planning_period_minutes,
                                    self.time_step_minutes))
        full_steps     = list(range(0,
                                    self.planning_period_minutes + self.lookahead_period_minutes,
                                    self.time_step_minutes))
        n_planning     = len(planning_steps)

        # --- step 1: predict demand per city over T+L window ---
        demand_full = self._predict_demand_by_step(state, cities, full_steps)

        demand_T = {
            (s, t_idx): demand_full.get((s, t_idx), 0)
            for s in cities for t_idx in range(n_planning)
        }
        demand_L = {
            s: sum(demand_full.get((s, t_idx), 0)
                   for t_idx in range(n_planning, len(full_steps)))
            for s in cities
        }

        # --- step 2: get current state (supply & vehicles en route) ---
        supply_now = {
            s: state.zones[s].available_count()
            for s in cities
        }

        en_route = defaultdict(int)
        for rel in state.get_active_relocations():
            steps_until_arrival = int(
                (rel.end_time - state.sim_time) / self.time_step_minutes
            )
            arrival_step = min(steps_until_arrival, n_planning - 1)
            en_route[(rel.destination, arrival_step)] += 1

        n_workers = state.worker_pool.idle_count()
        max_assignments = n_workers * 3

        if self.verbose:
            print(f"\n  [PROACTIVE MILP] t={state.sim_time:.0f} "
                  f"workers={n_workers} planning_steps={n_planning}")
            
        # --- step 3: ideal state at the end of T = enough vehicles to cover demand_L ---
        total_fleet = self.sim_cfg["fleet"]["total_vehicles"]
        total_on_trips  = sum(1 for v in state.vehicles.values()
                            if v.status == "on_trip")
        total_supply    = total_fleet - total_on_trips

        ideal_state = self._compute_ideal_state(
            cities, supply_now, demand_L, demand_T, n_planning, total_supply
        )

        # --- step 4: solve MILP to get relocation decisions ---
        if self.verbose:
            print(f"  supply:  { {c: state.zones[c].available_count() for c in cities} }")
            print(f"  ideal:   {ideal_state}")
            print(f"  demand_lookahead: {demand_L}")

        decisions = self._solve_milp(
            cities          = cities,
            planning_steps  = planning_steps,
            n_planning      = n_planning,
            demand_T        = demand_T,
            ideal_state     = ideal_state,
            supply_now      = supply_now,
            en_route        = en_route,
            max_assignments = max_assignments,
            state           = state,
            in_progress     = in_progress,
        )

        if self.verbose:
            print(f"  Proactive policy found {len(decisions)} decisions")

        return decisions


    # ------------------------------------------------------------------ #
    #  Compute ideal state                                               #
    # ------------------------------------------------------------------ #

    def _compute_ideal_state(self, cities: list, supply_now: dict,
                             demand_L: dict, demand_T: dict,
                             n_planning: int, total_supply: int
    ) -> dict:
        """
        Compute ideal_state[city] = target vehicle count at the end of the
        planning period, based on predicted demand over the lookahead window.
        """
        total_demand_L = sum(demand_L.values())        
        total_fleet    = self.sim_cfg["fleet"]["total_vehicles"]
        scale = min(1.0, total_fleet / total_demand_L) if total_demand_L > 0 else 1.0
        
        ideal = {}
        for city in cities:
            ideal[city] = max(0, math.ceil(demand_L.get(city, 0) * scale))
        return ideal


    # ------------------------------------------------------------------ #
    #  MILP                                                              #
    # ------------------------------------------------------------------ #

    def _travel_steps(self, origin: str, destination: str,
                      state: SimulationState
    ) -> int:
        """
        Convert travel time between two cities to number of time steps.
        Rounds up so vehicles don't arrive before they should.
        """
        travel = state.get_travel(origin, destination, state.sim_time)
        if not travel:
            return 999
        return math.ceil(travel["duration_min"] / self.time_step_minutes)
    

    def _solve_milp(self, cities: list,
                    planning_steps: list, n_planning: int,
                    demand_T: dict, ideal_state: dict,
                    supply_now: dict, en_route: dict,
                    max_assignments: int, state: SimulationState,
                    in_progress: set = None,
    ) -> list[tuple[str, str]]:
        """
        Solve the MILP and return relocation decisions for t=0.

        Variables:
            x[s, t]    = vehicles available at city s at time step t (integer ≥ 0)
            r[s, d, t] = vehicles relocated from s to d at step t    (integer ≥ 0)
            u[s, t]    = unmet demand at city s at step t         (continuous ≥ 0)

        Objective:
            min penalty_unmet × Σ u[s,t] + cost_per_relocation × Σ r[s,d,t]

        Constraints:
            1. Initial supply:    x[s, 0] = available_count(s) 
            2. Flow conservation: x[s, t+1] = x[s,t]
                                        - predicted_demand[s,t] + u[s,t]
                                        - Σ_d r[s,d,t]
                                        + Σ_d r[d,s, t - t-T(d,s)]
                                        + en_route[s, t]  for all s
            3. Unmet demand:      u[s,t] ≥ predicted_demand[s,t] - x[s,t]
            4. Worker capacity:   Σ_{s,d} r[s,d,t] ≤ n_idle_workers * 3  for all t
        """
        t0  = time_module.time()
        prob = LpProblem("proactive_milp", LpMinimize)


        # --- variables ---

        # vehicles at city s after step t
        x = {
            (s, t): LpVariable(f"x_{s}_{t}", lowBound=0, cat=LpContinuous)
            for s in cities for t in range(n_planning + 1)
        }

        # relocations from s to d starting at step t
        r = {
            (s, d, t): LpVariable(f"r_{s}_{d}_{t}", lowBound=0, cat="Integer")
            for s in cities
            for d in cities if d != s
            for t in range(n_planning)
        }

        # unmet demand at city s at step t
        u = {
            (s, t): LpVariable(f"u_{s}_{t}", lowBound=0, cat=LpContinuous)
            for s in cities for t in range(n_planning)
        }

        # deviation from ideal state (similar to unmet demand)
        dev = {s: LpVariable(f"dev_{s}", lowBound=0, cat=LpContinuous) for s in cities}


        # --- objective ---

        prob += (
            self.penalty_unmet       * (lpSum(u.values()) + lpSum(dev.values())) +
            self.cost_per_relocation * lpSum(r.values())
        )


        # --- constraints ---

        # 1. initial supply
        for s in cities:
            prob += x[(s, 0)] == supply_now[s]

        for t_idx in range(n_planning):
            for s in cities:
                d_st = demand_T.get((s, t_idx), 0)

                # 3. unmet demand
                prob += u[(s, t_idx)] >= d_st - x[(s, t_idx)]

                # 2. flow conservation

                # relocations leaving s at this step
                leaving = lpSum(r[(s, d, t_idx)] for d in cities if d != s)

                # relocations arriving at s from previous steps
                arriving = lpSum(
                    r[(d, s, t_idx - self._travel_steps(d, s, state))]
                    for d in cities if d != s
                    if 0 <= t_idx - self._travel_steps(d, s, state) < n_planning
                ) + en_route.get((s, t_idx), 0)

                prob += (
                    x[(s, t_idx + 1)] == x[(s, t_idx)]
                                            - d_st
                                            + u[(s, t_idx)]
                                            - leaving
                                            + arriving
                )

            # 4. worker capacity (max simultaneous relocations)
            prob += (
                lpSum(r[(s, d, t_idx)]
                      for s in cities
                      for d in cities if d != s)
                <= max_assignments
            )

        # ideal state (soft constraint)
        for s in cities:
            prob += dev[s] >= ideal_state[s] - x[(s, n_planning)]


        # --- solve ---

        solver  = PULP_CBC_CMD(msg = 0, timeLimit = self.solver_time_limit_sec)
        prob.solve(solver)

        solve_time   = round(time_module.time() - t0, 2)
        solve_status = LpStatus[prob.status]

        self._solve_times.append(solve_time)
        self._solve_statuses.append(solve_status)

        if self.verbose:
            print(f"  Proactive MILP: {solve_status} in {solve_time}s")

        if solve_status not in ("Optimal", "Not Solved"):
            print(f"  Proactive MILP status: {solve_status} "
                  f"(t={solve_time}s) - falling back to reactive")
            return self._fallback_reactive(state)

        # --- extract decisions for t=0 ---
        decisions = []
        claimed   = set()

        for s in cities:
            for d in cities:
                if d == s:
                    continue

                n_relocate = int(round(value(r[(s, d, 0)]) or 0))
                for _ in range(n_relocate):
                    available = [
                        v for v in state.zones[s].get_available_vehicles()
                        if v.id not in claimed
                        and v.id not in (in_progress or set())
                    ]
                    if not available:
                        break

                    vehicle = available[0]
                    claimed.add(vehicle.id)
                    decisions.append((vehicle.id, d))

        return decisions


    # ------------------------------------------------------------------ #
    #  Demand prediction                                                 #
    # ------------------------------------------------------------------ #

    def _predict_demand_by_step(self, state: SimulationState,
                                cities: list, steps: list,
    ) -> dict:
        """
        Predict demand per city per time step.
        Matches current weekday & time-of-day window from historical data,
        averaged over available historical weeks.

        Returns {(city, step_index): predicted_trips}
        """
        day_names       = ["Monday","Tuesday","Wednesday","Thursday",
                           "Friday","Saturday","Sunday"]
        
        start_dt        = datetime.strptime(self.sim_cfg["time"]["start_datetime"], "%Y-%m-%d %H:%M")
        current_dt      = start_dt + timedelta(minutes=state.sim_time)
        current_weekday = current_dt.weekday()
        target_day      = day_names[current_weekday]

        day_events      = self._history_by_weekday[target_day]

        if not day_events:
            return {}

        unique_dates = set(e["datetime"][:10] for e in day_events)
        n_weeks      = max(1, len(unique_dates))
        demand       = defaultdict(float)

        for t_idx, t_offset in enumerate(steps):
            from_min = (state.sim_time + t_offset) % (24 * 60)
            to_min   = (state.sim_time + t_offset + self.time_step_minutes) % (24 * 60)

            for event in day_events:
                event_min = event["time"] % (24 * 60)

                if from_min <= to_min:
                    in_window = from_min <= event_min < to_min
                else:
                    in_window = (event_min >= from_min
                                 or event_min < to_min)

                if in_window:
                    demand[(event["origin"], t_idx)] += 1 / n_weeks

        return dict(demand)


    # ------------------------------------------------------------------ #
    #  Fallback reactive                                                 #
    # ------------------------------------------------------------------ #

    def _fallback_reactive(self, state: SimulationState) -> list[tuple]:
        """Used when no history available yet."""
        decisions       = []
        already_claimed = set()

        needy   = sorted(state.get_zones_needing_relocation(),
                         key=lambda z: z.available_count())
        surplus = state.get_zones_with_surplus()

        for needy_zone in needy:
            deficit = needy_zone.min_vehicles - needy_zone.available_count()

            for _ in range(deficit):
                donors = [z.name for z in surplus]
                donor  = self._find_best_donor(needy_zone.name, donors, state)
                if not donor:
                    break

                available = [
                    v for v in state.zones[donor].get_available_vehicles()
                    if v.id not in already_claimed
                ]
                if not available:
                    break

                vehicle = available[0]
                already_claimed.add(vehicle.id)
                decisions.append((vehicle.id, needy_zone.name))
                if state.zones[donor].surplus <= 1:
                    surplus = [z for z in surplus if z.name != donor]

        return decisions


    def _find_best_donor(self, needy_city: str, surplus_cities: list,
                         state: SimulationState) -> str | None:
        if not surplus_cities:
            return None
        
        def tt(city):
            try:
                t = state.get_travel(city, needy_city, state.sim_time)
                return t["duration_min"] if t else float("inf")
            except Exception:
                return float("inf")
            
        return min(surplus_cities, key=tt)


    # ------------------------------------------------------------------ #
    #  Diagnostics                                                       #
    # ------------------------------------------------------------------ #

    def solver_stats(self) -> dict:
        if not self._solve_times:
            return {}
        
        return {
            "n_solves":        len(self._solve_times),
            "avg_solve_sec":   round(sum(self._solve_times) / len(self._solve_times), 4),
            "max_solve_sec":   round(max(self._solve_times), 4),
            "optimal_pct":     round(
                sum(1 for s in self._solve_statuses if s == "Optimal")
                / len(self._solve_statuses) * 100, 1
            ),
        }