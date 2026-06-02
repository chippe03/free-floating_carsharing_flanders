from math import floor

# https://www.vlaanderen.be/mobiliteit-en-openbare-werken/onderzoek-verplaatsingsgedrag-vlaanderen-ovg/onderzoek-verplaatsingsgedrag-vlaanderen-7-2023-2024/22-mei-2025-onderzoek-verplaatsingsgedrag-7-fietsgebruik-in-vlaanderen-op-hoogste-peil-ooit 
intercity_per_day_weekdays = 369000
intercity_per_day_weekends = 304000

# "Meer dan de helft van deze verplaatsingen zijn van of naar Brussel."
intercity_per_day_weekdays = floor(intercity_per_day_weekdays / 2)
intercity_per_day_weekends = floor(intercity_per_day_weekends / 2)

# "Tegelijk zijn verplaatsingen met deelsystemen wel nog maar goed voor 2,5%
# van alle verplaatsingen in Vlaanderen." --> weekdays: ~2% & weekends: ~3%
# 2% van de vlamingen gebruikt dagelijks een deelwagen
# 74% gebruikt nooit een deelwagen
carsharing_trips_per_day_weekdays = 0.02 * intercity_per_day_weekdays
carsharing_trips_per_day_weekends = 0.03 * intercity_per_day_weekends

# population of OECD cities except Brussels:
# https://www.ibz.rrn.fgov.be/sites/default/files/documents/fr/population/statistiques/population-bevolking-20260101.pdf 
antwerp     = 564561
ghent       = 273665
charleroi   = 206585
liege       = 198044
bruges      = 120393
kortrijk    = 81484
leuven      = 105233
mechelen    = 90282
bergen      = 97337
namur       = 115330
ostend      = 72942

total = antwerp + ghent + charleroi + liege + bruges + kortrijk + leuven + mechelen + bergen + namur + ostend

coverage = (antwerp + ghent + bruges + kortrijk + leuven + mechelen + ostend) / total
print(f"coverage: {coverage*100}%")
trips_in_network_weekdays = coverage * carsharing_trips_per_day_weekdays
trips_in_network_weekends = coverage * carsharing_trips_per_day_weekends

# Add Brussels
trips_in_network_weekdays_brussels = intercity_per_day_weekdays * 0.02
trips_in_network_weekends_brussels = intercity_per_day_weekends * 0.03

# base rate per hour
base_rate_weekday = floor((trips_in_network_weekdays + trips_in_network_weekdays_brussels) / 24)
base_rate_weekend = floor((trips_in_network_weekends + trips_in_network_weekends_brussels) / 24)

print(f"weekdays - intercity trips: {intercity_per_day_weekdays*coverage} - intercity carsharing: {carsharing_trips_per_day_weekdays} - base rate: {base_rate_weekday}")
print(f"weekends - intercity trips: {intercity_per_day_weekends*coverage} - intercity carsharing: {carsharing_trips_per_day_weekends} - base rate: {base_rate_weekend}")

print({intercity_per_day_weekdays})
print(f"Total trips weekdays: {intercity_per_day_weekdays+intercity_per_day_weekdays*coverage}")
print(f"Total trips weekends: {intercity_per_day_weekends+intercity_per_day_weekends*coverage}")

print(f"Car sharing trips weekdays: {(intercity_per_day_weekdays+intercity_per_day_weekdays*coverage)*0.02}")
print(f"Car sharing trips weekends: {(intercity_per_day_weekends+intercity_per_day_weekends*coverage)*0.03}")

print(f"Car sharing trips per hour weekdays: {((intercity_per_day_weekdays+intercity_per_day_weekdays*coverage)*0.02)/24}")
print(f"Car sharing trips per hour weekends: {((intercity_per_day_weekends+intercity_per_day_weekends*coverage)*0.03)/24}")