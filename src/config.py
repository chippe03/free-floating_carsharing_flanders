import yaml
from pathlib import Path

CONFIG_DIR = Path("config")

def load(filename: str) -> dict:
    with open(CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)

cities      = load("cities.yaml")
simulation  = load("simulation.yaml")
tomtom      = load("tomtom.yaml")
financials  = load("financials.yaml")