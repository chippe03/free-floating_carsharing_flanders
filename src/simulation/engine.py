import simpy
import json
from pathlib import Path
from datetime import datetime, timedelta

from src import config
from src.simulation.state import SimulationState
from src.demand.generator import DemandGenerator

from src.entities.trip import Trip
from src.entities.worker import Worker, WorkerStatus

from src.relocation.base_policy import BaseRelocationPolicy
from src.relocation.create_policy import load_policy

class SimulationEngine:
    """
    Discrete-event simulation engine.
    """

    def __init__(self, n_days: int = None, snapshot_interval: int = None,
                 policy: BaseRelocationPolicy = None,
                 fleet_size_override: int = None, workers_override: dict = None,
                 verbose: bool = True,):
        """
        n_days:                how many simulated days to run
        snapshot_interval:     print state every N simulated minutes
        policy:                relocation policy to use (baseline / reactive / proactive / nightly)
        fleet_size_override:   override total fleet size in config
        workers_override:      override number of workers in config
        """
        self.verbose = verbose
        self.env     = simpy.Environment()
        self.state   = SimulationState()

        if policy is None:
            policy  = load_policy()
        self.policy  = policy

        self.sim_cfg = config.simulation
        self.start_dt = datetime.strptime(
            self.sim_cfg["time"]["start_datetime"], "%Y-%m-%d %H:%M"
        )

        if n_days is None:
            n_days = self.sim_cfg["time"]["duration_days"]
        self.n_days  = n_days

        self.snapshot_interval = snapshot_interval

        self.fleet_size_override = fleet_size_override
        self.workers_override    = workers_override

        # initialize pending relocations
        self._pending_relocations: list[tuple[str, str]] = []
        self._total_forgotten = 0
        self._total_queued    = 0

        # apply fleet size / workers override to config
        if fleet_size_override:
            self.sim_cfg["fleet"]["total_vehicles"]  = fleet_size_override
        if workers_override:
            self.sim_cfg["workers"]["total_workers"] = workers_override
        
        # load train travel matrix for worker repositioning
        train_path = Path("data/processed/train_travel_matrix.json")
        if train_path.exists():
            with open(train_path) as f:
                self._train_matrix = json.load(f)
            if verbose:
                print("  Engine: using train travel times for repositioning\n")
        else:
            self._train_matrix = None
            print("  Engine: train travel matrix not found, "
                  "falling back to car travel for worker repositioning\n")

        # pre-generate trips per day, respecting weekday/weekend
        self.trips = self._generate_all_trips()

        # log every event for analysis
        self.event_log = []


    # ------------------------------------------------------------------ #
    #  Time helpers                                                      #
    # ------------------------------------------------------------------ #

    def _current_datetime(self, sim_minute: float) -> datetime:
        return self.start_dt + timedelta(minutes=sim_minute)


    def _is_weekend(self, sim_minute: float) -> bool:
        dt = self._current_datetime(sim_minute)
        return dt.weekday() >= 5   # 5=Saturday, 6=Sunday
    
    
    def _check_interval(self) -> int:
        if self.policy is not None:
            return self.policy.check_interval_minutes
        return self.sim_cfg["relocation"]["check_interval_minutes"]
    

    # ------------------------------------------------------------------ #
    #  Generate trips                                                    #
    # ------------------------------------------------------------------ #

    def _generate_all_trips(self) -> list:
        """Generate trips day by day, switching weekend pattern automatically."""
        all_trips = []

        for day in range(self.n_days):
            day_start_min = day * 24 * 60
            dt            = self._current_datetime(day_start_min)
            is_weekend    = dt.weekday() >= 5

            gen = DemandGenerator(
                is_weekend = is_weekend,
                seed       = self.sim_cfg["demand"]["seed"] + day
            )
            trips = gen.generate_day(day=day)
            all_trips.extend(trips)

            if self.verbose:
                day_name = dt.strftime("%A %d %b")
                print(f"  Generated {len(trips):>4} trips for {day_name} "
                      f"({'weekend' if is_weekend else 'weekday'})")

        if self.verbose:
            print()
        return all_trips
    

    # ------------------------------------------------------------------ #
    #  Train travel time lookup                                          #
    # ------------------------------------------------------------------ #

    def _train_travel_time(self, origin: str, destination: str
                          ) -> float | None:
        """
        Look up train travel time (minutes) between two cities.
        Uses the NMBS GTFS-derived train_travel_matrix.json.
        Returns none if no data is available.
        """
        if origin == destination:
            return 0.0
        if self._train_matrix is None:
            return None
        
        hour     = str(int(self.env.now % (24 * 60)) // 60)
        day_type = "weekend" if self._is_weekend(self.env.now) else "weekday"

        try:
            mins = (self._train_matrix
                    .get(origin, {})
                    .get(destination, {})
                    .get(day_type, {})
                    .get(hour))
            return float(mins) if (mins is not None and mins > 0.0) else float("inf")
        except Exception:
            return float("inf")
        

    def _reposition_time(self, worker: Worker, destination: str) -> float:
        """
        Travel time for a worker to reposition to a city.
        Returns inf if unreachable.
        """
        if worker.city == destination:
            return 0.0
        
        train = self._train_travel_time(worker.city, destination)
        if train is not None and train > 0.0:
            return train
        else:
            if self.verbose:
                print(f"No train travel found between {worker.city} and {destination}")
            return float("inf")


    # ------------------------------------------------------------------ #
    #  Relocations                                                       #
    # ------------------------------------------------------------------ #

    def _try_execute_relocation(self, vehicle_id: str, destination: str):
        """
        Attempt to execute a single relocation decision.
        
        Priority for worker assignment:
            1. Worker already at origin: start immediately (no repositioning).
            2. Worker closest to the origin: reposition first,
               but only if they can arrive before the next check interval.
               If repositioning takes too long, decision is queued.
            3. If no worker is found, decision is queued.
        """
        vehicle = self.state.vehicles.get(vehicle_id)
        if not vehicle or not vehicle.is_available or str(vehicle.city) == destination:
            return
        
        origin = vehicle.city
        travel = self.state.get_travel(origin, destination, self.env.now)
        if not travel:
            return
        
        valid_cities = set(self.state.zones.keys())
        
        # priority 1: worker at origin
        worker_at_origin = self.state.worker_pool.assign_at_origin(
            origin        = origin,
            destination   = destination,
            valid_cities  = valid_cities,
        )
        if worker_at_origin:
            self._start_relocation(worker_at_origin, vehicle_id,
                                   origin, destination)
            return
        
        # priority 2: closest worker by train travel time
        check_interval = self._check_interval()
        if getattr(self.policy, 'persist_pending', False) and hasattr(self.policy, 'trigger_hour'):
            current_hour = int((self.env.now % (24 * 60)) // 60)
            remaining_minutes = max(check_interval, (6 - self.policy.trigger_hour) * 60)
        else:
            remaining_minutes = check_interval
        
        worker = self.state.worker_pool.assign_closest(
            origin          = origin,
            destination     = destination,
            train_travel_fn = self._train_travel_time,
            valid_cities    = valid_cities,
        )

        if worker:
            repo_time = self._reposition_time(worker, origin)
            if repo_time < remaining_minutes:
                # worker can reach origin before next interval
                # reposition first, then execute relocation
                worker.start_reposition(origin)
                self.env.process(
                    self._reposition_then_relocate(
                        worker, vehicle_id, origin, destination, repo_time
                    )
                )
            else:
                # worker is too far: release & queue
                self.state.worker_pool.unassign(worker)
                self._queue_relocation(vehicle_id, origin, destination)
            return
        
        # priority 3: no worker found
        self._queue_relocation(vehicle_id, origin, destination)
        
    
    def _start_relocation(self, worker: Worker, vehicle_id: str,
                          origin: str, destination: str):
        """Start relocation immediately (worker at origin)."""
        if not worker.is_available:
            print(f"  ⚠️  _start_relocation: worker {worker.id} is not available "
                  f"(status={worker.status.value}), skipping")
            return

        worker.start_relocation(trip_id=f"pending_{vehicle_id}", destination=destination)

        relocation     = self.state.start_relocation(vehicle_id=vehicle_id, destination=destination)
        worker.trip_id = relocation.id

        self.state.worker_pool._update_peak()

        self.env.process(self._execute_relocation(relocation.id, worker))
        self._log("RELOCATION_STARTED",
                 vehicle_id   = vehicle_id,
                 origin       = relocation.origin,
                 destination  = relocation.destination,
                 distance_km  = relocation.distance_km,
                 duration_min = relocation.duration_min)


    def _reposition_then_relocate(self, worker: Worker, vehicle_id: str,
                                  origin: str, destination: str, repo_time: float):
        """
        Worker travels to origin city by train, then executes the relocation.
        If the vehicle is no longer available when they arrive, worker goes idle.
        """
        if repo_time > 0:
            # get distance for logging: train has no distance
            try:
                car_travel = self.state.get_travel(
                    worker.city, origin, self.env.now
                )
                repo_dist = car_travel["distance_km"] if car_travel else 0.0
            except Exception:
                repo_dist = 0.0
        
        self._log("WORKER_REPOSITION_STARTED",
                 vehicle_id   = worker.id,
                 origin       = worker.city,
                 destination  = origin,
                 distance_km  = repo_dist,
                 duration_min = repo_time)
        
        yield self.env.timeout(repo_time)

        self.state.worker_pool.finish_reposition(
            worker_id    = worker.id,
            city         = origin,
            valid_cities = set(self.state.zones.keys())
        )
        self._log("WORKER_REPOSITION_COMPLETED",
                  vehicle_id   = worker.id,
                  origin       = worker.city,
                  destination  = origin,
                  distance_km  = repo_dist,
                  duration_min = repo_time)
        
        # check vehicle is still available after repositioning
        vehicle = self.state.vehicles.get(vehicle_id)
        if not vehicle or not vehicle.is_available:
            # vehicle was taken: try pending at this city
            executed = self._process_pending(worker, worker.city)
            if not executed:
                self.env.process(self._reposition_worker(worker=worker))
            return
        
        self._start_relocation(worker, vehicle_id, origin, destination)


    def _execute_relocation(self, relocation_id: str, worker):
        """Execute a relocation and handle worker afterwards."""
        relocation = self.state.relocations[relocation_id]
        yield self.env.timeout(relocation.duration_min)

        self.state.sim_time = self.env.now
        self.state.complete_relocation(relocation_id)
        self.state.worker_pool.release(
            worker       = worker,
            distance_km  = relocation.distance_km,
            duration_min = relocation.duration_min,
            valid_cities = set(self.state.zones.keys())
        )

        self._log("RELOCATION_COMPLETED",
                  vehicle_id   = relocation.vehicle_id,
                  origin       = relocation.origin,
                  destination  = relocation.destination,
                  distance_km  = relocation.distance_km,
                  duration_min = relocation.duration_min)
        
        # worker is now at relocation destination
        # check for pending relocations from this city
        if self._should_continue_nightly() or not getattr(self.policy, 'persist_pending', False):
            executed = self._process_pending(worker, worker.city)
        else:
            executed = None


        if not executed:
            if getattr(self.policy, 'persist_pending', False):
                # nightly policy: worker goes home, log forgotten relocations
                self._send_worker_home(worker)
            else:
                # no pending here, reposition to nearest city with pending relocations
                self.env.process(self._reposition_worker(worker=worker))
    

    def _reposition_worker(self, worker: Worker, target_city: str = None):
        """
        Move worker by train:
            - If target_city is given, go there.
            - Else, move to nearest city with pending relocations.
              If none exist, the worker stays idle.
        """
        def reposition_time(city: str) -> float:
            if city == worker.city:
                return 0.0
                
            train = self._train_travel_time(worker.city, city)
            return train if (train is not None and train > 0.0) else float("inf")
        
        check_pending_after = False
        if target_city is None:
            check_pending_after = True

            if not self._pending_relocations:
                return
            
            # collect cities with pending relocations
            pending_origins = set(
                self.state.vehicles[vid].city
                for vid, _ in self._pending_relocations
                if vid in self.state.vehicles
                and self.state.vehicles[vid].is_available
                and self.state.vehicles[vid].city in self.state.zones
            )
            if not pending_origins:
                return
            
            target_city = min(pending_origins, key=reposition_time)

        duration = reposition_time(target_city) if target_city != worker.city else 0.0

        if target_city == worker.city and check_pending_after:
            self._process_pending(worker, worker.city)
            return

        if duration == float("inf"):
            return
        
        try:
            car_travel = self.state.get_travel(worker.city, target_city, self.env.now)
            distance_km = car_travel["distance_km"] if car_travel else 0.0
        except Exception:
            distance_km = 0.0
        
        worker.start_reposition(target_city)
        self._log("WORKER_REPOSITION_STARTED",
                  vehicle_id   = worker.id,
                  origin       = worker.city,
                  destination  = target_city,
                  distance_km  = distance_km,
                  duration_min = duration)
        
        yield self.env.timeout(duration)

        self.state.worker_pool.finish_reposition(
            worker_id    = worker.id,
            city         = target_city,
            valid_cities = set(self.state.zones.keys()),
        )
        self._log("WORKER_REPOSITION_COMPLETED",
                  vehicle_id   = worker.id,
                  origin       = worker.city,
                  destination  = target_city,
                  distance_km  = distance_km,
                  duration_min = duration)
        
        # try pending relocations in this city (only for normal repositions)
        if check_pending_after:
            self._process_pending(worker, target_city)


    # ------------------------------------------------------------------ #
    #  Pending relocations                                               #
    # ------------------------------------------------------------------ #

    def _queue_relocation(self, vehicle_id: str, origin: str, destination: str):
        """Add a relocation to the pending queue and log it."""
        self._pending_relocations.append((vehicle_id, destination))
        self._total_queued += 1
        self._log("RELOCATION_PENDING",
                  vehicle_id  = vehicle_id,
                  origin      = origin,
                  destination = destination)


    def _process_pending(self, worker: Worker, city: str) -> bool:
        """
        Try to execute pending relocations from this city.
        Returns True if one was executed.
        """
        for i, (vehicle_id, destination) in enumerate(self._pending_relocations):
            vehicle = self.state.vehicles.get(vehicle_id)

            # skip if vehicle is no longer available
            if not vehicle or not vehicle.is_available:
                continue
            if vehicle.city != city or vehicle.city == destination:
                continue

            # found a pending relocation: execute
            self._pending_relocations.pop(i)
            self._start_relocation(worker, vehicle_id, city, destination)
            return True
        
        return False


    # ------------------------------------------------------------------ #
    #  Nightly relocation: workers go home                               #
    # ------------------------------------------------------------------ #
    def _should_continue_nightly(self) -> bool:
        """Returns True if it's still within the nightly rebalance window."""
        current_hour = int((self.env.now % (24 * 60)) // 60)
        return current_hour < 6
    
    def _send_worker_home(self, worker: Worker):
        """Send a nightly worker home and log any remaining pending as forgotten."""
        if worker.city != worker.home_city:
            self.env.process(
                self._reposition_worker(worker=worker, target_city=worker.home_city)
            )
        
        # log forgotten only when last worker returns
        all_home = all(
            w.city == w.home_city or w.status == WorkerStatus.IDLE
            for w in self.state.worker_pool.workers
        )
        if all_home and self._pending_relocations:
            n_forgotten = len(self._pending_relocations)
            self._total_forgotten += n_forgotten
            self._log_worker_event("RELOCATIONS_FORGOTTEN", count=n_forgotten)
            self._pending_relocations = []


    # ------------------------------------------------------------------ #
    #  Processes                                                         #
    # ------------------------------------------------------------------ #

    def _relocation_process(self):
        """Check for relocations every check_interval_minutes."""
        interval = self._check_interval()
        persist  = getattr(self.policy, 'persist_pending', False)
    
        while True:
            yield self.env.timeout(interval)
            if self.policy is None:
                continue

            self.state.sim_time = self.env.now

            if hasattr(self.policy, 'update_history'):
                self.policy.update_history(self.event_log, self.env.now)
            

            if persist:
                # nightly policy: keep pending
                self._pending_relocations = [
                    (vid, dest) for vid, dest in self._pending_relocations
                    if vid in self.state.vehicles
                    and self.state.vehicles[vid].is_available
                    and self.state.vehicles[vid].city != dest
                ]
                n_forgotten = 0
            else:
                # count & log forgotten relocations before clearing
                n_forgotten = len(self._pending_relocations)
                if n_forgotten > 0:
                    self._total_forgotten += n_forgotten
                    self._log_worker_event("RELOCATIONS_FORGOTTEN", count=n_forgotten)
                self._pending_relocations = []

            # get new decisions from policy
            decisions = self.policy.relocate(self.state)

            if getattr(self.policy, 'verbose', False):
                print(f"\n  Worker status snapshot:")
                print(self.state.worker_pool.status_snapshot())

            # nightly workers return home (reposition)
            if hasattr(self.policy, 'get_worker_returns'):
                returns = self.policy.get_worker_returns(self.state)
                for worker_id, home_city in returns:
                    worker = next(
                        (w for w in self.state.worker_pool.workers if w.id == worker_id), None
                    )
                    if worker and worker.is_available:
                        travel = self._train_travel_time(worker.city, home_city)
                        if travel:
                            worker.start_reposition(home_city)
                            self.env.process(
                                self._reposition_worker(worker=worker, target_city=home_city)
                            )
            
            # try to execute all decisions
            for vehicle_id, destination in decisions:
                self._try_execute_relocation(vehicle_id, destination)


    def _trip_arrival_process(self):
        """
        Feed trips into the simulation in chronological order.
        Waits until each trip's request_time, then processes it.
        """
        for trip in sorted(self.trips, key=lambda t: t.request_time):

            # wait until this trip is due
            wait = trip.request_time - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)

            # update sim time in state
            self.state.sim_time = self.env.now
            self.state.day      = int(self.env.now // (24 * 60))

            # try to assign a vehicle
            vehicle = self.state.request_trip(trip)

            if vehicle:
                # spawn a process to complete the trip after travel time
                self.env.process(self._trip_process(trip, vehicle))
                self._log("TRIP_ASSIGNED", trip=trip, vehicle_id=vehicle.id)
            else:
                self._log("TRIP_UNMET", trip=trip)


    def _trip_process(self, trip: Trip, vehicle):
        """
        Simulate a single trip.
        Waits for the travel duration, then completes it.
        """
        travel   = self.state.get_travel(trip.origin, trip.destination, self.env.now)
        duration = travel["duration_min"]

        trip.start(self.env.now)
        yield self.env.timeout(duration)

        self.state.sim_time = self.env.now
        self.state.complete_trip(trip.id)
        self._log("TRIP_COMPLETED", trip=trip, vehicle_id=vehicle.id)


    def _snapshot_process(self):
        """Print a state snapshot every snapshot_interval minutes."""
        if self.snapshot_interval is None:
            return
            yield

        while True:
            yield self.env.timeout(self.snapshot_interval)
            self.state.sim_time = self.env.now
            self.state.day      = int(self.env.now // (24 * 60))

            # derive current datetime and weekend status
            dt         = self._current_datetime(self.env.now)
            is_weekend = self._is_weekend(self.env.now)

            print(f"\n  {dt.strftime('%A %d %b %Y  %H:%M')}  "
                  f"{'[weekend]' if is_weekend else '[weekday]'}")
            self.state.print_snapshot()


    # ------------------------------------------------------------------ #
    #  Run                                                               #
    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = None):
        """Set up and run the simulation."""
        if verbose is not None:
            self.verbose = verbose
        
        self.state.deploy_fleet(
            verbose      = self.verbose
        )

        # register all processes
        self.env.process(self._trip_arrival_process())
        self.env.process(self._snapshot_process())
        self.env.process(self._relocation_process())

        duration = self.n_days * 24 * 60
        if self.verbose:
            print(f"Starting simulation")
            print(f"  From:   {self.start_dt.strftime('%A %d %b %Y %H:%M')}")
            print(f"  Length: {self.n_days} day(s)")
            print(f"  Trips:  {len(self.trips)} total\n")

        self.env.run(until=duration)

        if self.verbose:
            print("\nSimulation complete.")
            self.state.print_snapshot()
            self._print_summary()
        
        self._log("SIMULATION_SUMMARY",
                  peak_workers    = self.state.worker_pool._peak_active,
                  total_workers   = len(self.state.worker_pool.workers),
                  total_forgotten = self._total_forgotten,
                  total_queued    = self._total_queued)
        

    # ------------------------------------------------------------------ #
    #  Logging                                                           #
    # ------------------------------------------------------------------ #

    def _log_worker_event(self, event_type: str, vehicle_id: str = None,
                          origin: str = None, destination: str = None,
                          count: int = None):
        dt = self._current_datetime(self.env.now)
        self.event_log.append({
            "time":           round(self.env.now, 2),
            "datetime":       dt.strftime("%Y-%m-%d %H:%M"),
            "hour":           dt.hour,
            "day":            int(self.env.now // (24 * 60)),
            "weekday":        dt.strftime("%A"),
            "is_weekend":     self._is_weekend(self.env.now),
            "event":          event_type,
            "vehicle_id":     vehicle_id,
            "origin":         origin,
            "destination":    destination,
            "count":          count,
            "workers_active": self.state.worker_pool.active_count(),
            "workers_peak":   self.state.worker_pool.peak_active,
        })


    def _log(self, event_type: str,
             trip: Trip = None, vehicle_id: str = None,
             origin: str = None, destination: str = None,
             distance_km: float = None, duration_min: float = None,
             **extra):
        
        # derive from trip if not explicitly passed
        if trip:
            origin       = trip.origin
            destination  = trip.destination
            travel       = self.state.get_travel(trip.origin, trip.destination, self.env.now)
            distance_km  = travel["distance_km"]
            duration_min = travel["duration_min"]
        
        dt = self._current_datetime(self.env.now)

        entry = {
            "time":        round(self.env.now, 2),
            "datetime":    dt.strftime("%Y-%m-%d %H:%M"),
            "hour":        dt.hour,
            "day":         int(self.env.now // (24 * 60)),
            "weekday":     dt.strftime("%A"),
            "is_weekend":  self._is_weekend(self.env.now),
            "event":       event_type,
            "trip_id":     trip.id          if trip else None,
            "origin":      origin,
            "destination": destination,
            "vehicle_id":  vehicle_id,
            "distance_km": distance_km,
            "duration_min":duration_min,
            **extra
        }
        self.event_log.append(entry)


    def save_event_log(self, filename: str = "baseline_events.json"):
        output_dir = Path("./data/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / filename
        with open(path, "w") as f:
            json.dump(self.event_log, f, indent=2)
        print(f"Event log saved to {path}")


    # ------------------------------------------------------------------ #
    #  Summary                                                           #
    # ------------------------------------------------------------------ #

    def _print_summary(self):
        total_trips  = len([e for e in self.event_log if e["event"] == "TRIP_ASSIGNED"])
        total_unmet  = len([e for e in self.event_log if e["event"] == "TRIP_UNMET"])
        total        = total_trips + total_unmet
        unmet_rate   = total_unmet / total if total > 0 else 0

        # relocation stats
        relocations   = [e for e in self.event_log if e["event"] == "RELOCATION_COMPLETED"]
        total_rel     = len(relocations)
        total_rel_km  = round(sum(e["distance_km"] for e in relocations if e["distance_km"]), 1)
        total_rel_min = round(sum(e["duration_min"] for e in relocations if e["duration_min"]), 1)

        ws = self.state.worker_pool.summary()
        forgotten = self._total_forgotten
        queued    = len([e for e in self.event_log if e["event"] == "RELOCATION_PENDING"])
        self._total_queued = queued

        repositioning    = [e for e in self.event_log if e["event"] == "WORKER_REPOSITION_COMPLETED"]
        nr_repositioning = len(repositioning)
        reposition_km    = round(sum(e["distance_km"] for e in repositioning if e["distance_km"]), 1)
        reposition_min   = round(sum(e["duration_min"] for e in repositioning if e["duration_min"]), 1)

        print(f"\n{'='*80}")
        print(f"  RESULTS ({self.n_days} day simulation - policy: {self.policy.__class__.__name__} - fleet size: {self.sim_cfg['fleet']['total_vehicles']})")
        print(f"{'='*80}")
        print(f"  Total trip requests:   {total}")
        print(f"  Trips served:          {total_trips}")
        print(f"  Trips unmet:           {total_unmet}  ({unmet_rate:.1%})")

        print(f"\n  Relocations:           {total_rel}")
        print(f"  Relocation km:         {total_rel_km} km")
        print(f"  Relocation mins:       {total_rel_min} min")

        print(f"\n  Peak workers active:   {ws['peak_active']}")
        print(f"  Total relocations:     {ws['total_relocations']}")
        print(f"  Relocations queued:    {queued}")
        print(f"  Relocations forgotten: {forgotten}")
    
        print(f"\n  Repositioning:         {nr_repositioning}")
        print(f"  Repositioning km:      {reposition_km} km")
        print(f"  Repositioning mins:    {reposition_min} min")

        print(f"\n  Final fleet distribution:")
        for city in self.state.zones:
            vehicles = self.state.fleet_distribution()[city]
            workers  = [w for w in self.state.worker_pool.workers if w.city == city]
            at_home  = sum(1 for w in workers if w.home_city == city)
            away     = len(workers) - at_home
            bar      = "█" * vehicles
            print(f"    {city:<12} {bar} {len(workers)} workers ({at_home} 🏠)")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    engine = SimulationEngine(n_days= 20,snapshot_interval=1440)
    engine.run()
    engine.save_event_log()