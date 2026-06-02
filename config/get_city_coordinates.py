import yaml, requests
from pathlib import Path

API_KEY = "SOEM9zZWPuwl3D2mlTILQh12JKZAP9HU"

def geocode(city_name):
    url = f"https://api.tomtom.com/search/2/geocode/{city_name}.json"
    params = {"key": API_KEY, "countrySet": "BE"}
    response = requests.get(url, params=params)
    result = response.json()["results"][0]["position"]
    return result["lat"], result["lon"]

def update_coordinates():
    with open(Path("config/cities.yaml")) as f:
        config = yaml.safe_load(f)

    cities = config["cities"]
    updated = 0

    for city_name, city_data in cities.items():
        # skip if coordinates already filled in
        if city_data.get("lat") and city_data.get("lon"):
            print(f"  {city_name:<12} already has coordinates, skipping")
            continue

        try:
            lat, lon = geocode(city_name)
            cities[city_name]["lat"] = lat
            cities[city_name]["lon"] = lon
            print(f"  {city_name:<12} → {lat}, {lon}")
            updated += 1
        except Exception as e:
            print(f"  {city_name:<12} → FAILED: {e}")

    with open(Path("config/cities.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\nUpdated {updated} cities in cities.yaml")

if __name__ == "__main__":
    update_coordinates()