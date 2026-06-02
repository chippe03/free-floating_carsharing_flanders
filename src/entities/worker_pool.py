from dataclasses import dataclass, field
from typing import List, Optional
from src.entities.worker import Worker, WorkerStatus


@dataclass
class WorkerPool:
    """
    Manages the pool of relocation workers across all cities.
    Tracks usage and peak concurrency.
    """
    workers: List[Worker] = field(default_factory=list)

    # usage tracking
    _peak_active:       int = 0
    _total_relocations: int = 0

    def add_worker(self, worker: Worker):
        self.workers.append(worker)

    def get_available(self) -> List[Worker]:
        """Return all idle workers."""
        return [w for w in self.workers if w.is_available]

    def get_available_in_city(self, city: str) -> List[Worker]:
        """Return idle workers in a specific city."""
        return [w for w in self.workers if w.is_available and w.city == city]
    
    
    # ------------------------------------------------------------------ #
    #  Assignment                                                        #
    # ------------------------------------------------------------------ #
    
    def assign_at_origin(self, origin: str, destination: str,
                         valid_cities: set = None,
    ) -> Optional[Worker]:
        """
        Priority 1: assign an idle worker already at the origin city.
        Results None if no idle worker is at origin
        """
        if valid_cities and destination not in valid_cities:
            return None
        
        available_at_origin = self.get_available_in_city(origin)
        # print(f"  [ASSIGN_ORIGIN] origin={origin} dest={destination} "
        #       f"candidates={[w.id for w in available_at_origin]}")
        
        return available_at_origin[0] if available_at_origin else None


    def assign_closest(self, origin: str, destination: str,
                       train_travel_fn, valid_cities: set = None,
    ) -> Optional[Worker]:
        """
        Priority 2: assign the closest idle worker not at the origin city.
        Results None if no idle worker exists elsewhere.
        Engine decides whether to reposition or queue them.
        """
        if valid_cities and destination not in valid_cities:
            return None
        
        available = [w for w in self.get_available() if w.city != origin]
        if not available:
            return None
        
        worker = self._find_closest(available, origin, train_travel_fn)
        # print(f"  [ASSIGN_CLOSEST] origin={origin} dest={destination} "
        #       f"selected={worker.id if worker else None} status={worker.status.value if worker else None}")
        return worker if worker else None


    def unassign(self, worker: Worker):
        """Return a worker to idle without completing a relocation."""
        # print(f"  [UNASSIGN] worker={worker.id} was status={worker.status.value} "
        #       f"trip_id={worker.trip_id} dest={worker.destination}")
        worker.status      = WorkerStatus.IDLE
        worker.trip_id     = None
        worker.destination = None


    def release(self, worker: Worker,
                distance_km: float, duration_min: float,
                valid_cities: set = None
    ):
        """Release a worker after relocation completes."""
        if worker not in self.workers:
            print(f"  ⚠️  release: worker {worker.id} not in pool")
            return
        else:
            # print(f"  [RELEASE] worker={worker.id} status={worker.status.value} "
            #       f"destination={worker.destination} trip_id={worker.trip_id}")
            worker.finish_relocation(
                distance_km  = distance_km,
                duration_min = duration_min,
                valid_cities = valid_cities
            )
            self._total_relocations += 1


    def finish_reposition(self, worker_id: str, city: str, valid_cities: set = None):
        worker = next(
            (w for w in self.workers if w.id == worker_id), None
        )
        if worker:
            worker.finish_reposition(city=city, valid_cities=valid_cities)


    # ------------------------------------------------------------------ #
    #  Priority helpers                                                  #
    # ------------------------------------------------------------------ #
   
    def _find_closest(self, available: List[Worker],
                      city: str, train_travel_fn
    ) -> Optional[Worker]:
        """Any idle worker closest to the origin by travel time."""
        def travel_time(worker: Worker) -> float:
            if worker.city == city:
                return 0.0
            train = train_travel_fn(worker.city, city)
            if train is not None and train > 0.0:
                return float(train)
            else:
                return float('inf')
            
        candidates = [w for w in available if travel_time(w) < float('inf')]
        return min(candidates, key=travel_time) if candidates else None
    

    # ------------------------------------------------------------------ #
    #  Stats                                                             #
    # ------------------------------------------------------------------ #

    def active_count(self) -> int:
        return sum(1 for w in self.workers if not w.is_available)
    
    def idle_count(self) -> int:
        return sum(1 for w in self.workers if w.is_available)
    
    def _update_peak(self):
        n_in_use = self.active_count()
        if n_in_use > self.peak_active:
            self._peak_active = n_in_use


    @property
    def peak_active(self) -> int:
        return self._peak_active

    def summary(self) -> dict:
        return {
            "total_workers":    len(self.workers),
            "currently_active": self.active_count(),
            "currently_idle":   self.idle_count(),
            "peak_active":      self._peak_active,
            "total_relocations":self._total_relocations,
            "total_km":         round(sum(w.total_km  for w in self.workers), 1),
            "total_min":        round(sum(w.total_min for w in self.workers), 1),
        }

    def __repr__(self):
        return (f"WorkerPool(total={len(self.workers)}, "
                f"idle={self.idle_count()}, peak={self._peak_active})")
    
    def status_snapshot(self) -> str:
        lines = []
        for w in self.workers:
            lines.append(f"  {w.id}: city={w.city} status={w.status.value} "
                        f"trip_id={w.trip_id} dest={w.destination}")
        return "\n".join(lines)