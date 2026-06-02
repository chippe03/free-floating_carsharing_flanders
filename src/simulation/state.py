import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.demand.od_matrix import HOUR_TO_WINDOW
from src.entities.vehicle import Vehicle, VehicleStatus
from src.entities.trip import Trip, TripStatus
from src.entities.zone import Zone
from src.entities.worker import Worker
from src.entities.worker_pool import WorkerPool
from src import config

@dataclass
class Relocation:
    """Tracks an in-progress operator relocation."""
    id:           str
    vehicle_id:   str
    origin:       str
    destination:  str
    start_time:   float
    end_time:     float          # start_time + travel duration
    distance_km:  float
    duration_min: float
    completed:    bool = False


class SimulationState:
    """
    Single source of truth for the simulation at any point in time.
    The engine reads and writes this object - nothing else holds state.
    """

    def __init__(self):
        self.sim_time:    float = 0.0     # current time in minutes
        self.day:         int   = 0       # current simulation day

        self.zones:     Dict[str, Zone]    = {}
        self.vehicles:  Dict[str, Vehicle] = {}
        self.trips:     Dict[str, Trip]    = {}

        self.relocations: Dict[str, Relocation] = {}
        self.worker_pool = WorkerPool()
        
        self.initial_distribution: dict[Zone, int] = {}

        self._load_config()
        self._init_zones()

    # ------------------------------------------------------------------ #
    #  Setup                                                             #
    # ------------------------------------------------------------------ #

    def _load_config(self):
        self.cities_cfg = config.cities["cities"]
        self.sim_cfg    = config.simulation

        start_str       = self.sim_cfg["time"]["start_datetime"]
        self._start_dt  = datetime.strptime(start_str, "%Y-%m-%d %H:%M")

        with open("./data/processed/travel_matrix_timed.json") as f:
            self.travel_matrix = json.load(f)

    def _init_zones(self):
        """Create one Zone per city from config."""
        for name, cfg in self.cities_cfg.items():
            self.zones[name] = Zone(
                name          = name,
                lat           = cfg["lat"],
                lon           = cfg["lon"],
                demand_weight = cfg["demand_weight"],
                min_vehicles  = cfg["min_vehicles"],
            )

    def _init_workers(self, distribution: dict = None):
        """Distribute workers across cities, inversely proportional to fleet distribution.
        Workers are stationed at high-demand cities (relocation destinations).
        """
        total = self.sim_cfg["workers"]["total_workers"]
        names = list(self.zones.keys())

        if distribution:
            fleet_total = sum(distribution.values())
            # invert: cities with fewer vehicles get more workers
            inverse = {n: fleet_total - distribution[n] for n in names}
            weights = np.array([inverse[n] for n in names], dtype=float)
        else:
            weights = np.array([self.zones[n].demand_weight for n in names], dtype=float)

        shares = np.round(weights / weights.sum() * total).astype(int)
        diff   = total - shares.sum()
        shares[np.argmax(shares)] += diff
        counts = dict(zip(names, shares.tolist()))

        worker_counter = 1
        for city, count in counts.items():
            for _ in range(count):
                wid = f"W{worker_counter:03d}"
                w   = Worker(id=wid, city=city, home_city=city)
                self.worker_pool.add_worker(w)
                worker_counter += 1

    def deploy_fleet(self, distribution: dict = None, verbose: bool = False):
        """
        If distribution is provided, use it directly.
        Otherwise distribute proportionally to demand_weight.
        """
        total = self.sim_cfg["fleet"]["total_vehicles"]

        if distribution:
            # use provided distribution
            counts = {city: distribution.get(city, 0)
                      for city in self.zones}
        else:
            # demand-weighted distribution
            names    = list(self.zones.keys())
            weights  = np.array([self.zones[n].demand_weight for n in names])
            shares   = np.round(weights / weights.sum() * total).astype(int)
            
            # fix rounding so total is exact
            diff = total - shares.sum()
            shares[np.argmax(shares)] += diff
            counts  = dict(zip(names, shares.tolist()))

        vehicle_counter = 1
        for city, count in counts.items():
            for _ in range(count):
                vid = f"V{vehicle_counter:03d}"
                v   = Vehicle(id=vid, city=city)
                self.vehicles[vid] = v
                self.zones[city].add_vehicle(v)
                vehicle_counter += 1
        
        self.initial_distribution = counts
        # distribute workers proportional to fleet counts
        self._init_workers(distribution=counts)
        
        if verbose:
            self.print_snapshot()

    # ------------------------------------------------------------------ #
    #  Travel lookup                                                     #
    # ------------------------------------------------------------------ #

    def get_travel(self, origin: str, destination: str,
                sim_time: float = None) -> dict:
        """
        Get travel time between two cities,
        using the time-dependent travel_matrix.
        """
        if origin == destination:
            return {"distance_km": 0.0, "duration_min": 0.0}
        
        is_weekend = self._is_weekend_minute(sim_time)
        hour       = int(sim_time % (24 * 60) // 60)
        day_type   = "weekend" if is_weekend else "weekday"

        try:
            return self.travel_matrix[origin][destination][day_type][str(hour)]
        except KeyError as e:
            print(f"  [get_travel ERROR] origin={origin!r} destination={destination!r} "
                f"day_type={day_type!r} hour={hour!r} missing_key={e}")
            print(f"  Available origins: {list(self.travel_matrix.keys())[:5]}")
            if origin in self.travel_matrix:
                print(f"  Available dests from {origin}: "
                    f"{list(self.travel_matrix[origin].keys())}")
            raise

        # return self.travel_matrix[origin][destination][day_type][str(hour)]

    def _is_weekend_minute(self, sim_time: float) -> bool:
        dt = self._start_dt + timedelta(minutes=sim_time)
        return dt.weekday() >= 5


    # ------------------------------------------------------------------ #
    #  Trip lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def request_trip(self, trip: Trip) -> Optional[Vehicle]:
        """
        Register a trip request and assign the nearest available vehicle.
        Returns the assigned Vehicle, or None if no vehicle available.
        """
        self.trips[trip.id] = trip
        self.zones[trip.origin].log_demand()

        vehicle = self._find_available_vehicle(trip.origin)

        if vehicle is None:
            trip.mark_unmet()
            self.zones[trip.origin].log_unmet()
            return None

        trip.assign(vehicle.id, self.sim_time)
        vehicle.pick_up(trip.id)
        return vehicle

    def complete_trip(self, trip_id: str):
        """Move vehicle to destination, update zone counts."""
        trip    = self.trips[trip_id]
        vehicle = self.vehicles[trip.vehicle_id]

        travel  = self.get_travel(trip.origin, trip.destination, self.sim_time)
        dist    = travel["distance_km"]
        dur     = travel["duration_min"]

        trip.complete(self.sim_time, dist, dur)

        # move vehicle between zones
        current_zone = self.zones.get(vehicle.city)
        if current_zone and vehicle in current_zone.vehicles:
            current_zone.remove_vehicle(vehicle)
        vehicle.drop_off(trip.destination, dist, dur)
        self.zones[trip.destination].add_vehicle(vehicle)

        self.zones[trip.origin].log_trip_start()
        self.zones[trip.destination].log_trip_end()

    # ------------------------------------------------------------------ #
    #  Relocation lifecycle                                              #
    # ------------------------------------------------------------------ #

    def start_relocation(self, vehicle_id: str, destination: str) -> Relocation:
        """Initiate a vehicle relocation. Returns the Relocation object."""
        import uuid
        vehicle = self.vehicles[vehicle_id]
        origin  = vehicle.city
        travel  = self.get_travel(origin, destination, self.sim_time)

        relocation = Relocation(
            id           = f"R{str(uuid.uuid4())[:6]}",
            vehicle_id   = vehicle_id,
            origin       = origin,
            destination  = destination,
            start_time   = self.sim_time,
            end_time     = self.sim_time + travel["duration_min"],
            distance_km  = travel["distance_km"],
            duration_min = travel["duration_min"],
        )
        self.relocations[relocation.id] = relocation

        self.zones[origin].remove_vehicle(vehicle)
        vehicle.start_relocation(destination, relocation.id)
        self.zones[origin].total_relocations_out += 1

        return relocation

    def complete_relocation(self, relocation_id: str):
        """Finish a relocation, park vehicle in destination zone."""
        relocation = self.relocations[relocation_id]
        vehicle    = self.vehicles[relocation.vehicle_id]

        vehicle.finish_relocation(relocation.distance_km, relocation.duration_min)
        self.zones[relocation.destination].add_vehicle(vehicle)
        self.zones[relocation.destination].total_relocations_in += 1
        relocation.completed = True

    # ------------------------------------------------------------------ #
    #  Queries                                                           #
    # ------------------------------------------------------------------ #

    def _find_available_vehicle(self, city: str) -> Optional[Vehicle]:
        """Return first available vehicle in a city, or None."""
        available = self.zones[city].get_available_vehicles()
        return available[0] if available else None

    def get_active_trips(self) -> List[Trip]:
        return [t for t in self.trips.values()
                if t.status == TripStatus.IN_PROGRESS]

    def get_active_relocations(self) -> List[Relocation]:
        return [r for r in self.relocations.values() if not r.completed]

    def get_zones_needing_relocation(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.needs_relocation]

    def get_zones_with_surplus(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.surplus > 0]

    def fleet_distribution(self) -> Dict[str, int]:
        """Available vehicles per city right now."""
        return {name: zone.available_count()
                for name, zone in self.zones.items()}

    # ------------------------------------------------------------------ #
    #  Snapshot                                                          #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """
        Full state snapshot at current sim_time.
        Useful for logging, visualization, and debugging.
        """
        # snapshot of vehicles (per city)
        fleet_snapshot = {
            name: {
                "available":        zone.available_count(),
                "total":            zone.total_count(),
                "needs_relocation": zone.needs_relocation,
                "surplus":          zone.surplus,
                }
                for name, zone in self.zones.items()
        }

        # snapshot of workers (per city)
        worker_snapshot = {}
        for city in self.zones:
            workers_here = [w for w in self.worker_pool.workers if w.city == city]
            worker_snapshot[city] = {
                "total":      len(workers_here),
                "idle":       sum(1 for w in workers_here if w.is_available),
                "relocating": sum(1 for w in workers_here if not w.is_available),
                "at_home":    sum(1 for w in workers_here if w.home_city == city),
                "away":       sum(1 for w in workers_here if w.home_city != city),
            }

        return {
            "sim_time":           self.sim_time,
            "day":                self.day,
            "hour":               round((self.sim_time % (24 * 60)) / 60, 2),
            "fleet":              fleet_snapshot,
            "workers":            worker_snapshot,
            "active_trips":       len(self.get_active_trips()),
            "active_relocations": len(self.get_active_relocations()),
            "total_trips":        len(self.trips),
            "total_unmet":        sum(z.total_unmet for z in self.zones.values()),
            "total_relocations":  len(self.relocations),
        }

    def print_snapshot(self):
        s    = self.snapshot()
        hour = int(s["hour"])
        mins = int((s["hour"] - hour) * 60)

        print(f"\n{'='*75}")
        print(f"  Day {s['day']+1}  |  {hour:02d}:{mins:02d}  |  t={s['sim_time']:.0f} min")
        print(f"{'='*75}")

        # fleet + workers
        print(f"  {'City':<12} {'Avail':>6} {'Total':>6} {'Surplus':>8} "
              f"{'Workers':>8} {'Idle':>5} {'Home':>5} {'Alert':>6}")
        print(f"  {'-'*70}")

        for name in s["fleet"]:
            fd = s["fleet"][name]
            wd = s["workers"].get(name, {})
            alert = "⚠️ " if fd["needs_relocation"] else ""
            home  = wd.get("at_home", 0)
            home_str  = f"({home}🏠)" if home > 0 else ""

            print(f"  {name:<12} {fd['available']:>6} {fd['total']:>6} "
                  f"{fd['surplus']:>8} {wd.get('total', 0):>8} "
                  f"{wd.get('idle', 0):>5} {home_str:>5} {alert:>6}")
            
        all_accounted = sum(
            len([w for w in self.worker_pool.workers if w.city == city])
            for city in self.zones
        )
        unaccounted = [
            w for w in self.worker_pool.workers
            if w.city not in self.zones
        ]
        if unaccounted:
            print(f"\n  ⚠️  {len(unaccounted)} workers with invalid city:")
            for w in unaccounted:
                print(f"    {w.id}: city='{w.city}' status={w.status} "
                    f"destination='{w.destination}' trip_id='{w.trip_id}'")

        print(f"  {'-'*70}")
        print(f"  Active trips:       {s['active_trips']}")
        print(f"  Active relocations: {s['active_relocations']}")
        print(f"  Workers in use:      {self.worker_pool.active_count()}"
              f"/{len(self.worker_pool.workers)}")
        print(f"  Total trips:        {s['total_trips']}")
        print(f"  Total unmet:        {s['total_unmet']}")
        print(f"  Total relocations:  {s['total_relocations']}")
        print(f"{'='*75}\n")


if __name__ == "__main__":
    state = SimulationState()
    state.deploy_fleet()
    state.print_snapshot()