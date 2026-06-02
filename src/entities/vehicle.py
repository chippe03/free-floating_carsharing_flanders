from enum import Enum
from dataclasses import dataclass
from typing import Optional

class VehicleStatus(Enum):
    IDLE        = "idle"
    IN_USE      = "in_use"
    RELOCATING  = "relocating"

@dataclass
class Vehicle:
    id:       str
    city:     str                          # current city

    status:      VehicleStatus = VehicleStatus.IDLE
    trip_id:     Optional[str] = None       # active trip or relocation id
    destination: Optional[str] = None       # only used during relocation to track where it's going

    # lifetime stats
    total_trips:        int   = 0
    total_relocations:  int   = 0
    total_km:           float = 0.0
    total_min:          float = 0.0

    def pick_up(self, trip_id: str):
        """Called when a customer starts a trip."""
        self.status  = VehicleStatus.IN_USE
        self.trip_id = trip_id

    def drop_off(self, destination_city: str, distance_km: float, duration_min: float):
        """Called when a customer ends a trip."""
        self.city        = destination_city
        self.status      = VehicleStatus.IDLE
        self.trip_id     = None
        self.total_trips += 1
        self.total_km    += distance_km
        self.total_min   += duration_min

    def start_relocation(self, destination_city: str, relocation_id: str):
        """Called when operator relocates this vehicle."""
        self.status      = VehicleStatus.RELOCATING
        self.trip_id     = relocation_id   # reuse field to track relocation id
        self.destination = destination_city

    def finish_relocation(self, distance_km: float, duration_min: float):
        """Called when relocation is complete."""
        self.status             = VehicleStatus.IDLE
        self.city               = self.destination
        self.trip_id            = None
        self.total_relocations += 1
        self.total_km          += distance_km
        self.total_min         += duration_min

    @property
    def is_available(self) -> bool:
        return self.status == VehicleStatus.IDLE

    def __repr__(self):
        return f"Vehicle({self.id}, city={self.city}, status={self.status.value})"