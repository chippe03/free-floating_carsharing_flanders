from dataclasses import dataclass, field
from typing import List
from src.entities.vehicle import Vehicle

@dataclass
class Zone:
    name:           str
    lat:            float
    lon:            float
    demand_weight:  float   # relative demand scale from cities.yaml
    min_vehicles:   int     # threshold before relocation triggers

    # live state (updated during simulation)
    vehicles:       List[Vehicle] = field(default_factory=list)

    # cumulative stats
    total_demand:          int = 0
    total_unmet:           int = 0
    total_incoming:        int = 0    # trips ending here
    total_outgoing:        int = 0    # trips starting here
    total_relocations_in:  int = 0
    total_relocations_out: int = 0

    # --- vehicle management ---

    def add_vehicle(self, vehicle: Vehicle):
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicles.remove(vehicle)

    def get_available_vehicles(self) -> List[Vehicle]:
        return [v for v in self.vehicles if v.is_available]

    def available_count(self) -> int:
        return len(self.get_available_vehicles())

    def total_count(self) -> int:
        return len(self.vehicles)

    # --- demand tracking ---

    def log_demand(self):
        self.total_demand += 1

    def log_unmet(self):
        self.total_unmet += 1

    def log_trip_start(self):
        self.total_outgoing += 1

    def log_trip_end(self):
        self.total_incoming += 1

    # --- relocation triggers ---

    @property
    def needs_relocation(self) -> bool:
        """True if available vehicles dropped below minimum threshold."""
        return self.available_count() < self.min_vehicles

    @property
    def surplus(self) -> int:
        """How many vehicles above the minimum threshold."""
        return self.available_count() - self.min_vehicles

    # --- stats ---

    @property
    def unmet_rate(self) -> float:
        if self.total_demand == 0:
            return 0.0
        return round(self.total_unmet / self.total_demand, 3)

    def summary(self) -> dict:
        return {
            "city":               self.name,
            "available_vehicles": self.available_count(),
            "total_vehicles":     self.total_count(),
            "needs_relocation":   self.needs_relocation,
            "surplus":            self.surplus,
            "total_demand":       self.total_demand,
            "total_unmet":        self.total_unmet,
            "unmet_rate":         f"{self.unmet_rate:.1%}",
        }

    def __repr__(self):
        return (f"Zone({self.name}, "
                f"available={self.available_count()}/{self.total_count()}, "
                f"needs_relocation={self.needs_relocation})")