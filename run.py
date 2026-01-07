#---------------------------------------------------------------------------------------#
#  run.py:
#  this function goes through the process of cleaning all outage and weather data from a specific time period
#  and merging it into one large dataset.
#
# BEFORE RUNNING: Insert EAGLE-I datasets into Eagle-idatasets folder
#----------------------------------------------------------------------------------------------

# packages
import outage_cleaning
import weather_cleaning
import fetch_weather_data
import map_outage_weather
import os

state="Illinois"
county='Cook'
start=2018
end=2024
# APPROACH FOR EVENT DETECTION: Can use percent_customers, percentile, or flat
approach='percentile'
# Value dictates the event detection number. If flat, this is in units of num customers
# If threshold, this is a fraction. If percentile, this is a decimal representing said percentile
value=0.75


# ensure proper file paths exist
output_folder = f'data/{state}/{county}'
os.makedirs(output_folder, exist_ok=True)
output_folder = 'Results/'
os.makedirs(output_folder, exist_ok=True)
output_folder = f'weather_data/{state}'
os.makedirs(output_folder, exist_ok=True)

# fetch weather data

# add if statement that only fetches weather data if it's not already loaded in
a=f"weather_data/{state}/weather_{state}_{county}_{start}_{end}.parquet"
if os.path.isfile(a):
    print("State weather data already downloaded. Moving onto weather cleaning.")
else:
    print("State weather data not found. Proceeding to fetch weather from IEM servers.")
    fetch_weather_data.main(state, county, start, end)

# clean + map weather data
a=f"weather_data/{state}/cleaned_weather_data_{county}.parquet"
if os.path.isfile(a):
    print("Weather Data already cleaned. Moving onto outage cleaning.")
else:
    weather_cleaning.main(state, county, start, end)

# clean outage data
a = f"outage_data/{state}/{county}/Merged_Cleaned_data_{start}_{end}_{county}_{state}.parquet"
if os.path.isfile(a):
    print("Outage Data already cleaned. Moving onto merge process.")
else:
    outage_cleaning.main(state, county, start, end)

# merge outage and weather data
map_outage_weather.main(state, county, start, end, value, approach)
print("Done! Please see the combined outage-weather data "
      "in results folder.")