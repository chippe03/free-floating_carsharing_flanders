from abc import ABC, abstractmethod
from src.simulation.state import SimulationState


class BaseRelocationPolicy(ABC):
    """
    Abstract base class for all relocation policies.
    Every policy must implement a single method: relocate().

    The engine calls relocate() on a fixed interval (e.g. every hour).
    The policy inspects the current state and decides which vehicles
    to move where. It returns a list of (vehicle_id, destination) pairs
    which the engine then executes.
    """

    def __init__(self, name: str, check_interval_minutes: int = 30):
        self.name = name
        self.persist_pending: bool = False
        self.check_interval_minutes = check_interval_minutes

        # lifetime stats for this policy
        self.total_relocations:   int   = 0
        self.total_km_relocated:  float = 0.0
        self.total_min_relocated: float = 0.0

    @abstractmethod
    def relocate(self, state: SimulationState) -> list[tuple[str, str]]:
        """
        Inspect the current simulation state and return relocation decisions.

        Returns a list of (vehicle_id, destination_city) tuples.
        Returning an empty list means no relocations this interval.
        """
        pass

    def log_relocation(self, distance_km: float, duration_min: float):
        """Called by the engine after each relocation is executed."""
        self.total_relocations  += 1
        self.total_km_relocated += distance_km
        self.total_min_relocated += duration_min

    def stats(self) -> dict:
        return {
            "policy":            self.name,
            "total_relocations": self.total_relocations,
            "total_km":          self.total_km_relocated,
            "avg_km":            (self.total_km_relocated / self.total_relocations) if self.total_relocations > 0 else 0.0,
            "total_min":         self.total_min_relocated,
            "avg_min":           (self.total_min_relocated / self.total_relocations) if self.total_relocations > 0 else 0.0,
        }

    def __repr__(self):
        return f"RelocationPolicy({self.name})"