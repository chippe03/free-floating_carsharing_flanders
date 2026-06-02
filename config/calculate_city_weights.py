import yaml
import math
from pathlib import Path

def calculate_weights():
    config_path = "config/cities.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cities = config["cities"]

    # ------------------------------------------------------------------ #
    #  Get total population excluding brussels                           #
    # ------------------------------------------------------------------ #

    brussels_pop = cities["brussels"]["population"]
    other_pop    = sum(
        data["population"]
        for city, data in cities.items()
        if city != "brussels"
    )
    total_pop = brussels_pop + other_pop

    # ------------------------------------------------------------------ #
    #  Compute raw weights                                               #
    # ------------------------------------------------------------------ #

    # brussels gets weight such that it accounts for 50% of all trips
    # remaining 50% split proportionally by population among other cities

    # normalize other cities to sum to 1.0 within their group
    weights = {}
    for city, data in cities.items():
        if city == "brussels":
            # brussels = 50% of all trips → weight = 1.0 (reference)
            # other cities together = 50% → each gets population share of 0.5
            weights[city] = other_pop / brussels_pop   # relative to others
        else:
            weights[city] = data["population"] / other_pop

    # normalize so brussels = 1.0 and others sum to 1.0
    # → brussels weight = 1.0, total other = 1.0 means brussels = 50%
    brussels_weight = 1.0
    for city in weights:
        if city != "brussels":
            weights[city] = weights[city]   # already normalized to sum 1.0

    # scale so brussels = 1.0
    scale = 1.0 / (other_pop / brussels_pop)
    for city in weights:
        if city != "brussels":
            weights[city] = round(weights[city] * scale * (other_pop / brussels_pop), 3)

    weights["brussels"] = 1.0

    # simpler direct calculation:
    # brussels = 50% of trips → assign weight proportional to its "effective population"
    # if brussels accounts for 50%, its effective weight = sum of all others
    effective_pop = {}
    for city, data in cities.items():
        if city == "brussels":
            effective_pop[city] = other_pop   # brussels = equivalent to all others combined
        else:
            effective_pop[city] = data["population"]

    total_effective = sum(effective_pop.values())

    # normalize to max = 1.0 for readability
    max_eff = max(effective_pop.values())
    demand_weights = {
        city: round((pop / max_eff) / 2, 4)
        for city, pop in effective_pop.items()
    }

    # ------------------------------------------------------------------ #
    #  Compute min_vehicles                                              #
    # ------------------------------------------------------------------ #
    
    # min_vehicles = round(200 * demand_weight)
    min_vehicles = {
        city: max(1, round(200 * w))
        for city, w in demand_weights.items()
    }

    # ------------------------------------------------------------------ #
    #  Write back to yaml                                                #
    # ------------------------------------------------------------------ #

    for city in cities:
        cities[city]["demand_weight"] = demand_weights[city]
        cities[city]["min_vehicles"]  = min_vehicles[city]

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nUpdated cities.yaml")
    return demand_weights, min_vehicles


if __name__ == "__main__":
    calculate_weights()