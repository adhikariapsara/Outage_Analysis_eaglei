# Outage_Analysis_eaglei

This repository contains outage and weather data processing scripts using the Eagle-I dataset, focused on outage analysis across all counties.

## Folder Structure
- **Eaglei_data/**: Raw outage data for all counties.
- **cleaned/**: Folder where cleaned data is stored after preprocessing.

## Scripts

- **`outage_cleaning.py`**  
  Cleans the Eagle-I outage data and stores the cleaned results in the `cleaned/` folder.

- **`weather_cleaning.py`**  
  Cleans the ASOS weather data and stores the output in the `cleaned/` folder.

- **`map_outage+weather.py`**  
  Maps the cleaned outage and weather datasets to extract event-based summaries used for downstream analysis.

## Purpose

This code is used to preprocess and align outage and weather data, generate clean event features, and prepare the dataset for modeling or analysis related to outage prediction and impact studies.
