from enum import Enum
from dataclasses import dataclass
from typing import Optional

class WorkerStatus(Enum):
    IDLE           = "idle"
    RELOCATING     = "relocating"
    REPOSITIONING  = "repositioning"

@dataclass
class Worker:
    id:        str
    city:      str                          # current city
    home_city: str                          # permanent home city

    status:      WorkerStatus = WorkerStatus.IDLE
    trip_id:     Optional[str] = None       # active relocation
    destination: Optional[str] = None       # during traveling to track where worker is going (for better metrics)

    # lifetime stats
    total_relocations:  int   = 0
    total_km:           float = 0.0
    total_min:          float = 0.0

    def start_relocation(self, trip_id: str, destination: str):
        """Assign worker to a relocation task."""
        # print(f"  [START_RELOC] worker={self.id} trip_id={trip_id} destination={destination}")
        self.status      = WorkerStatus.RELOCATING
        self.trip_id     = trip_id
        self.destination = destination

    def start_reposition(self, destination: str):
        """Travel to another city with pending relocations."""
        self.status      = WorkerStatus.REPOSITIONING
        self.trip_id     = None
        self.destination = destination

    def finish_relocation(self, distance_km: float, duration_min: float, valid_cities: set = None):
        """End relocation, update stats, and become available again."""
        if self.destination is None:
            print(f"  ⚠️  Worker {self.id}: finish_relocation called with no destination set")
        elif valid_cities and self.destination not in valid_cities:
            print(f"  ⚠️  Worker {self.id}: invalid destination "
                  f"'{self.destination}' - not updating city")
        else:
            self.city = self.destination
    
        self.status             = WorkerStatus.IDLE
        self.trip_id            = None
        self.destination        = None
        self.total_relocations += 1
        self.total_km          += distance_km
        self.total_min         += duration_min

    def finish_reposition(self, city: str, valid_cities: set = None):
        """Worker arrived at repositioning city, now idle there."""
        if valid_cities and city not in valid_cities:
            print(f"  ⚠️  Worker {self.id} finish_reposition: invalid city '{city}'")
            return
        self.city        = city
        self.status      = WorkerStatus.IDLE
        self.destination = None
        self.trip_id     = None

    @property
    def is_available(self) -> bool:
        return self.status == WorkerStatus.IDLE

    def __repr__(self):
        return (f"Worker({self.id}, city={self.city}, "
                f"home={self.home_city}, status={self.status.value})")