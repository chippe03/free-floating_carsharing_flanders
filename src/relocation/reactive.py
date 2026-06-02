"""
Reactive relocation policy.

At each check interval, compares available vehicles per city against
a minimum threshold. If available < threshold, the city is deficit
and triggers relocation from the nearest surplus city.

Options (mutually exclusive, checked in order):
   1.  thresholds: dict {city: int}
        Thresholds are explicitely given per city.
   2.  alpha: int, used to scale city demand weights
        min_vehicles[city] = max(1, round(alpha * demand_weight[city]))
   3.  default: use demand weights from config.cities
"""

from src.relocation.base_policy import BaseRelocationPolicy
from src.simulation.state import SimulationState
from src import config


class ReactivePolicy(BaseRelocationPolicy):

    def __init__(
            self,
            check_interval_minutes: int = None,
            thresholds:             dict = None,
            alpha:                  int = None,
            verbose:                bool = False,
    ):
        """
        check_interval_minutes: how often the policy is called
        thresholds:             explicit {city: min_vehicles} dict
        alpha:                  scaling factor for demand-weight thresholds
        """
        super().__init__(name="reactive")
        self.sim_cfg = config.simulation

        self.check_interval_minutes = (check_interval_minutes
                                       or self.sim_cfg["relocation"]["check_interval_minutes"])

        self.verbose    = verbose
        self.thresholds = thresholds
        self.alpha      = alpha

        if self.alpha is not None and self.thresholds is None:
            self.thresholds = self._compute_alpha_thresholds(alpha)

        if self.verbose:
            if self.thresholds is not None:
                print(f"\n  Reactive policy: thresholds = {self.thresholds}\n")
            elif self.alpha is not None:
                self.thresholds = self._compute_alpha_thresholds(alpha)
                print(f"\n  Reactive policy: alpha = {alpha}, thresholds computed from demand weights\n")
            else:
                print(f"\n  Reactive policy: using min_vehicles from config\n")
    

    # ------------------------------------------------------------------ #
    #  Compute thresholds                                                #
    # ------------------------------------------------------------------ #

    def _compute_alpha_thresholds(self, alpha: int) -> dict:
        """
        Compute thresholds from config.cities demand weights:
        min_vehicles[city] = max(1, round(alpha * demand_weight[city]))
        """
        cities_cfg = config.cities["cities"]
        thresholds = {}
        for city, cfg in cities_cfg.items():
            w = cfg.get("demand_weight")
            if w is None:
                print(f"   ⚠️ Reactive policy: unable to get demand weight for {city}.")
                w = 0.0
            thresholds[city] = max(1, round(alpha * w))
        return thresholds
    
    def _get_threshold(self, city: str, state: SimulationState) -> int:
        """
        Return the threshold for this city.
        """
        if self.thresholds is not None:
            return self.thresholds.get(city, state.zones[city].min_vehicles)
        return state.zones[city].min_vehicles
    

    # ------------------------------------------------------------------ #
    #  Main relocation logic                                             #
    # ------------------------------------------------------------------ #
        
    def relocate(self, state: SimulationState) -> list[tuple[str, str]]:
        """
        Returns relocation decisions as list of (vehicle_id, destination).
        Only relocates the minimum needed to bring cities back to threshold.

        Needy cities are served in order of largest deficit first.
        Donor is the nearest surplus city by car travel time.
        """
        decisions = []
        already_claimed = set()
        in_progress = {
            r.vehicle_id
            for r in state.relocations.values()
            if not r.completed
        }

        # determine deficit and surplus per city
        needy   = []
        surplus = []

        for city, zone in state.zones.items():
            threshold = self._get_threshold(city, state)
            available = zone.available_count()

            if available < threshold:
                needy.append((zone, (threshold - available), threshold))
            elif available > threshold:
                surplus.append(zone)

        if not needy or not surplus:
            return decisions

        # most starved cities first
        needy = sorted(needy, key=lambda x: x[2] - x[0].available_count(), reverse=True)

        for needy_zone, deficit, threshold in needy:
            for _ in range(deficit):
                # find the donor zone with the most surplus
                # that still has available vehicles after donating
                donor = self._find_best_donor(
                    needy_zone.name,
                    [z.name for z in surplus],
                    state,
                )
                if donor is None:
                    break

                # pick the first available vehicle not already claimed by another needy zone
                donor_zone = state.zones[donor]
                available  = [
                    v for v in donor_zone.get_available_vehicles()
                    if v.id not in already_claimed
                    and v.id not in in_progress
                ]
                if not available:
                    surplus = [z for z in surplus if z.name != donor]
                    continue

                vehicle = available[0]
                already_claimed.add(vehicle.id)
                decisions.append((vehicle.id, needy_zone.name))

                # remove donor from surplus if it is now at its threshold
                donor_threshold = self._get_threshold(donor, state)

                claimed_from_donor = sum(
                    1 for vid, _ in decisions
                    if vid in {v.id for v in donor_zone.get_available_vehicles()}
                )

                remaining = donor_zone.available_count() - len(in_progress & {
                    r.vehicle_id for r in state.relocations.values()
                    if not r.completed and r.origin == donor
                }) - claimed_from_donor

                if remaining <= donor_threshold:
                    surplus = [z for z in surplus if z.name != donor]

        return decisions


    # ------------------------------------------------------------------ #
    #  Donor selection                                                   #
    # ------------------------------------------------------------------ #

    def _find_best_donor(self, needy_city: str, surplus_zones: list,
                         state:SimulationState) -> str | None:
        """
        Find the surplus city with shortest car travel time to the needy city.
        """
        if not surplus_zones:
            return None

        def travel_time(zone: str) -> float:
            if zone == needy_city:
                return float("inf")
            try:
                travel = state.get_travel(zone, needy_city, state.sim_time)
                return travel["duration_min"] if travel else float("inf")
            except Exception:
                return float("inf")
        
        valid = [c for c in surplus_zones if travel_time(c) < float("inf")]
        return min(valid, key=travel_time) if valid else None