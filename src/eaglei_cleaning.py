#!/usr/bin/env python3
"""
eaglei_cleaning.py - EAGLE-i Outage Data Processing Pipeline

This file contains functions to clean and preprocess EAGLE-i outage data.

Author: Arslan Ahmad
License: MIT
"""

# importing necessary libraries
from src.configManager import ConfigManager
from src.eaglei_modules.eagleiEventProcessing import EagleiStateProcessor
from src.eaglei_modules.constants import TIMESTAMP_COL, YEAR_COL
import pandas as pd
import numpy as np
import os

def main(state: str, 
         county: str, 
         start_year: int, 
         end_year: int, 
         cleaned_file: str,
         config: ConfigManager) -> None:
    """
    Main function to clean EAGLE-i outage data for a given state and county.

    Parameters:
    - state (str): The state abbreviation (e.g., 'CA' for California).
    - county (str): The county name.
    - start_year (int): The starting year for data cleaning.
    - end_year (int): The ending year for data cleaning.
    - cleaned_file (str): The path to save the cleaned data file.

    Returns:
    - None
    """

    # Initialize the state processor which load EAGLEi data for a specific state 
    # if already cleaned, else Load and Clean it (takes almost a minute)
    state_data = EagleiStateProcessor(state_name=state)

    # create a county processor object for Hampden County
    county_data = state_data.get_county_processor(county_name=county)

    # for the cleaning, we need to identify gaps (missing timestamps) in the data first
    # the defaults settings are to consider only those gaps where there were 
    # at least 20 customers before the gap and at least 2 customers after 
    # the gap, and the gap duration is less than 24 hours
    county_data.identify_gaps(min_customers_before_gap = config.get("data_cleaning_parameters.min_customers_before_gap", 20),
                              min_customers_after_gap = config.get("data_cleaning_parameters.min_customers_after_gap", 2), 
                              max_gap_minutes = config.get("data_cleaning_parameters.max_gap_minutes", 24))

    # now identify duplicates and fix them
    county_data.drop_duplicate_timestamps()
    
    # For convenience, we can automatically decide the rank threshold and fill the gaps above that threshold
    # The automatic rank threshold decision is based on analyzing the gap ranking metrics and selecting a threshold that balances data integrity and completeness
    # first check if auto decision is enabled in config
    if config.get("data_cleaning_parameters.use_auto_gap_rank_threshold", True):
        county_data.fill_gaps(auto_decide_rank_threshold=True)
    else:
        # fill the gaps with a specific rank threshold if auto decision is not enabled
        county_data.fill_gaps(auto_decide_rank_threshold=False,
                              rank_threshold_quantile = config.get("data_cleaning_parameters.gap_rank_threshold_quantile", 0.4))
    # Find total customers in the county and add as attribute
    customers_data_file_path = os.path.join(
        config.get("data_paths.outage_data_dir"),
        state,
        f"county_total_customers_in_{state.lower()}.parquet"
    )
    # check if the file exists
    if os.path.exists(customers_data_file_path):
        customers_df = pd.read_parquet(customers_data_file_path)
        total_customers = customers_df[customers_df['county'] == county]['total_customers'].values
    else:
        print('Error: Could not find total customers in county. This may interfere with results if percent_customers'
              'is selected as threshold method.')
        total_customers=0
    # Extract county-level events using a minimum customer threshold as specified in the config file
    county_data.extract_events_ac_thr(event_detection_type=config.get("data_cleaning_parameters.event_detection_approach","flat"),
                                      total_customers=total_customers,
                                      customer_threshold = config.get("data_cleaning_parameters.events_customer_threshold", 30))

    # save only the data for the specified years
    # first create a year column
    county_data.county_df_with_events_ac_thr[YEAR_COL] = county_data.county_df_with_events_ac_thr[TIMESTAMP_COL].dt.year
    # check if all the years in the range are present
    available_years = county_data.county_df_with_events_ac_thr[YEAR_COL].unique()
    missing_years = [year for year in range(start_year, end_year + 1) if year not in available_years]
    if missing_years:
        print(f"Warning: The following years are missing in the data: {missing_years}")
    # filter the data for the specified years
    county_data.county_df_with_events_ac_thr = county_data.county_df_with_events_ac_thr[
        (county_data.county_df_with_events_ac_thr[YEAR_COL] >= start_year) & 
        (county_data.county_df_with_events_ac_thr[YEAR_COL] <= end_year)
    ]

    # remove the year column before saving
    county_data.county_df_with_events_ac_thr.drop(columns=[YEAR_COL], inplace=True)
    
    # Save the cleaned data to the specified file
    county_data.county_df_with_events_ac_thr.to_parquet(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")