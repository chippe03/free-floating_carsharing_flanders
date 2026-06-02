from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import uuid

class TripStatus(Enum):
    PENDING     = "pending"      # request created, no vehicle yet
    ASSIGNED    = "assigned"     # vehicle matched
    IN_PROGRESS = "in_progress"  # vehicle picked up
    COMPLETED   = "completed"    # trip finished
    UNMET       = "unmet"        # no vehicle available

@dataclass
class Trip:
    origin:       str            # city name
    destination:  str            # city name
    request_time: float          # simulation time in minutes

    id:           str            = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status:       TripStatus     = TripStatus.PENDING
    vehicle_id:   Optional[str]  = None

    # filled in as trip progresses
    assigned_time:   Optional[float] = None
    start_time:      Optional[float] = None
    end_time:        Optional[float] = None
    distance_km:     Optional[float] = None
    duration_min:    Optional[float] = None

    def assign(self, vehicle_id: str, sim_time: float):
        self.vehicle_id    = vehicle_id
        self.status        = TripStatus.ASSIGNED
        self.assigned_time = sim_time

    def start(self, sim_time: float):
        self.status     = TripStatus.IN_PROGRESS
        self.start_time = sim_time

    def complete(self, sim_time: float, distance_km: float, duration_min: float):
        self.status       = TripStatus.COMPLETED
        self.end_time     = sim_time
        self.distance_km  = distance_km
        self.duration_min = duration_min

    def mark_unmet(self):
        self.status = TripStatus.UNMET

    @property
    def wait_time(self) -> Optional[float]:
        """Minutes between request and vehicle pickup."""
        if self.start_time and self.request_time:
            return round(self.start_time - self.request_time, 2)
        return None

    def __repr__(self):
        return (f"Trip({self.id}, {self.origin}→{self.destination}, "
                f"status={self.status.value})")