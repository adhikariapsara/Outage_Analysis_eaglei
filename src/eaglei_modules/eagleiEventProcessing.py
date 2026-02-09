"""
eagleiEventProcessing.py

This module contains functions for processing EAGLE-I Data.
It includes functions for processing outage events, extracting events,
plotting performance curves, and building outage graphs.


Author: Arslan Ahmad
Last Updated: December 2025
License: MIT
"""


# ------------------------- Import Libraries -------------------------

# import warnings
# warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

from __future__ import annotations
from typing import Dict, List, Tuple, Any

import re
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from datetime import timedelta
import networkx as nx
import geopandas as gpd
import json
from urllib.request import urlopen
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px

from . import constants
from pandas.api.types import is_datetime64_any_dtype


# ------------------------- Custom Formatters -------------------------

class CustomScalarFormatter(ScalarFormatter):
    def _set_format(self):
        # This line defines the format string. 
        # Here, '%.2f' means a float with exactly two decimal places.
        # self.format = "%1.2f"
        # Here, '%.8f' means a float with exactly eight significant digits.
        self.format = "%.8g"

# Instantiate the custom formatter
custom_label_formatter = CustomScalarFormatter(useMathText=True)


# ------------------------- Data Cleaning Functions -------------------------

def verify_eaglei_files(verbose: int = 1) -> list[str]:
    """
    Verify the presence of essential EAGLEi data files in the specified directory.

    Args:
        verbose (int): If 1, prints the number of years found.

    Returns:
        list[str]: List of verified EAGLEi outage data file paths.

    Raises:
        FileNotFoundError: If the EAGLEi data directory does not exist or no outage data files are found.
    """

    # construct the path to the EAGLEi data directory
    cwd = os.getcwd()
    dir_path = os.path.join(cwd, constants.EAGLEI_DATA_DIR)

    # check if the directory exists
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    # get list of all csv files that start with 'eaglei_outages_' followed by exactly 4 digits (year) in the EAGLEi data directory
    pattern = r'^eaglei_outages_\d{4}'
    eaglei_files = [f for f in os.listdir(dir_path) if re.match(pattern, f)]

    if len(eaglei_files) == 0:
        raise FileNotFoundError("No EAGLEi outage data files found.")

    # Print the number of years found
    available_years = set(f.split('_')[-1].split('.')[0] for f in eaglei_files)
    if verbose == 1:
        print(f"Found EAGLEi outage data files for {len(available_years)} years: {', '.join(sorted(available_years))}")

    # sort the files for consistency
    eaglei_files.sort()

    # create a list of full file paths
    eaglei_files = [os.path.join(dir_path, f) for f in eaglei_files]

    return eaglei_files


def load_eaglei_state_data(state_name: str, 
                           verbose: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to read and clean EAGLEi outage data for a specific state.

    There are differences in column names in the EAGLE-I files for different years.
    This function identify those and standardizes the data into a consistent format.
    And store the cleaned file in a parquet file which takes sgignificantly less space and is faster to read.

    Columns From 2014 to 2020 fips_code, county, state, sum, run_start_time  
    Columns From 2021 to 2022 fips_code, county, state, customers_out, run_start_time (sum is replaced with customers_out)  
    Columns in 2023 fips_code, county, state, sum, run_start_time  (again started using sum instead of customers_out)  
    Columns in 2024 fips_code, county, state, customers_out, run_start_time, total_customers (sum is replaced with customers_out)  
    
    Additionally, the total_customers column is placed in the individual EAGLE-I outage files
    instead of a seperate file. So this function seperate that information and stores it in a seperate file.

    Args:
        state_name (str): The name of the state to filter the data.
        verbose (int): If 1, prints progress and information messages.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - Cleaned DataFrame containing outage data for the specified state.
            - DataFrame containing total customers data for counties in the specified state.

    Raises:
        FileNotFoundError: If any of the specified EAGLE-I files do not exist.
        ValueError: If no data is found for the specified state.
    """

    # ==================== Reading and Filtering Data ====================
    cwd = os.getcwd()
    output_file_name = f"eaglei_cleaned_{state_name.lower()}.parquet"
    output_file_dir = os.path.join(cwd, constants.OUTAGE_DATA_DIR, state_name)
    output_file_path = os.path.join(output_file_dir, output_file_name)
    output_county_total_customers_file_name = f"county_total_customers_in_{state_name.lower()}.parquet"
    
    # check if the file already exists
    if os.path.isfile(output_file_path):
        print(f"Cleaned data for {state_name} already exists as {output_file_name}. Loading the existing file...")
        try:
            data_df = pd.read_parquet(output_file_path)
            county_total_customers = pd.read_parquet(os.path.join(output_file_dir, output_county_total_customers_file_name))
            return data_df, county_total_customers
        except Exception as e:
            print(f"Error loading existing cleaned data: {e}")
            print("Proceeding to re-clean the data...")

    eaglei_file_paths = verify_eaglei_files(verbose=0)

    data_df = pd.DataFrame()
    for file in tqdm(eaglei_file_paths, desc=f"Reading EAGLE-I files for state: {state_name}"):
        # check if the file exists
        if not os.path.isfile(file):
            raise FileNotFoundError(f"The file {file} does not exist.")
        
        # read the CSV file (loading the complete file for all states)
        temp_df = pd.read_csv(file)
        
        # check if the state name exists in the 'state' column
        if state_name in temp_df[constants.STATE_COL].values:
            # append the filtered data to the main DataFrame
            data_df = pd.concat([data_df, temp_df[temp_df[constants.STATE_COL] == state_name]], ignore_index=True)

    # check if the DataFrame is empty
    if data_df.empty:
        raise ValueError(f"No data found for state: {state_name}")
    
    print(f"Total records for {state_name} (before cleaning): {data_df.shape[0]}\n")

    # ==================== Consolidating Columns ====================

    # convert the run_start_time column to datetime
    data_df[constants.TIMESTAMP_COL] = pd.to_datetime(data_df[constants.TIMESTAMP_COL], errors='raise')
    
    # Add a year column to the DataFrame
    # This will help in grouping the data by year later on
    data_df[constants.YEAR_COL] = data_df[constants.TIMESTAMP_COL].dt.year
        
    # Verify the relationship between 'sum' and 'customers_out' columns
    print(f"=== Verifying Relationship between 'sum' and {constants.CUSTOMERS_COL} columns ===")

    # Check if when 'sum' is not NaN, 'customers_out' is NaN and vice versa
    sum_not_nan = data_df['sum'].notna()
    customers_out_not_nan = data_df[constants.CUSTOMERS_COL].notna()

    print(f"Rows with non-NaN 'sum': {sum_not_nan.sum()}")
    print(f"Rows with non-NaN {constants.CUSTOMERS_COL}: {customers_out_not_nan.sum()}")

    # Check the mutual exclusivity
    both_not_nan = sum_not_nan & customers_out_not_nan
    both_nan = (~sum_not_nan) & (~customers_out_not_nan)

    print(f"\nRows where both 'sum' and {constants.CUSTOMERS_COL} are not NaN: {both_not_nan.sum()}")
    print(f"Rows where both 'sum' and {constants.CUSTOMERS_COL} are NaN: {both_nan.sum()}")

    # Check by year to see the pattern
    print(f"\n=== Breakdown by year ===")
    year_breakdown = data_df.groupby(constants.YEAR_COL).agg({
        'sum': lambda x: x.notna().sum(),
        constants.CUSTOMERS_COL: lambda x: x.notna().sum()
    }).rename(columns={'sum': 'sum_non_nan_count', constants.CUSTOMERS_COL: 'customers_out_non_nan_count'})

    print(year_breakdown)

    # Verify the hypothesis: when sum is not NaN, customers_out should be NaN and vice versa
    if both_not_nan.sum() == 0:
        print(f"\nVERIFIED: 'sum' and {constants.CUSTOMERS_COL} are mutually exclusive (no rows have both values)")
    else:
        print(f"\nWARNING: Found {both_not_nan.sum()} rows where both 'sum' and {constants.CUSTOMERS_COL} have values")
        print("Sample rows with both values:")
        print(data_df[both_not_nan][['year', 'sum', constants.CUSTOMERS_COL]].head())
    if both_nan.sum() == 0:
        print(f"VERIFIED: No rows have both 'sum' and {constants.CUSTOMERS_COL} as NaN")
    else:
        print(f"WARNING: Found {both_nan.sum()} rows where both 'sum' and {constants.CUSTOMERS_COL} are NaN")

    # Create a temporary consolidated column that combines 'sum' and 'customers_out'
    print(f"\n=== Consolidating 'sum' and {constants.CUSTOMERS_COL} columns ===")

    # Use 'sum' values where available, otherwise use 'customers_out' values
    data_df['outages'] = data_df['sum'].fillna(data_df[constants.CUSTOMERS_COL])

    # Check for remaining NaN values before converting to integer
    nan_count = data_df['outages'].isna().sum()

    if nan_count > 0:
        print(f"Examining rows with NaN values in both 'sum' and 'customers_out':")
        nan_rows = data_df[data_df['outages'].isna()]
        print(f"Years with NaN values: {sorted(nan_rows['year'].unique())}")
        print(f"Sample rows with NaN values:")
        print(nan_rows[['year', 'county', 'sum', 'customers_out']].head())
        
        # Remove rows with NaN values (since they don't have outage data)
        print(f"\nRemoving {nan_count} rows with missing data...")
        data_df = data_df.dropna(subset=['outages'])

    # Convert to integer (since we're dealing with customer counts)
    data_df['outages'] = data_df['outages'].astype(int)

    # Verify the consolidation worked correctly
    print(f"Total rows after consolidation: {len(data_df)}")
    print(f"All rows have data: {data_df['outages'].notna().all()}")

    # remove the 'sum', 'customers_out' and 'year' columns as they are no longer needed
    data_df = data_df.drop(columns=['sum', 'customers_out', 'year'])
    
    # rename the 'outages' column to CUSTOMERS_COL
    data_df = data_df.rename(columns={'outages': constants.CUSTOMERS_COL})

    # ==================== Separating the total_customers column ====================
    
    county_total_customers = data_df.loc[:, ['fips_code', 'county', 'total_customers']].copy()
    county_total_customers.dropna(subset=['total_customers'], inplace=True)
    county_total_customers.drop_duplicates(subset=['county', 'total_customers'], inplace=True)
    county_total_customers['total_customers'] = county_total_customers['total_customers'].astype(int)
    county_total_customers.sort_values(by='total_customers', ascending=False, inplace=True)
    county_total_customers.reset_index(drop=True, inplace=True)
    
    # remove the total_customers column as it is no longer needed
    data_df = data_df.drop(columns=['total_customers'], inplace=False)

    # ==================== Clean the data ====================

    # Check if the data is sorted by timestamp_column
    if not data_df[constants.TIMESTAMP_COL].is_monotonic_increasing:
        if verbose > 0:
            print(f'Data was not sorted by {constants.TIMESTAMP_COL}, sorting the data...')
        data_df = data_df.sort_values(by=constants.TIMESTAMP_COL).reset_index(drop=True)

    # Standardize timestamps to 15-minute intervals
    invalid_times = data_df[~(data_df[constants.TIMESTAMP_COL].dt.minute.isin([0, 15, 30, 45]))]
    if invalid_times.shape[0] > 0:
        if verbose > 0:
            print('There are invalid times in the data:', invalid_times.shape[0])
            print('  Fixing the invalid times...')
        data_df[constants.TIMESTAMP_COL] = data_df[constants.TIMESTAMP_COL].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0)
        )
        if verbose > 0:
            print('  Invalid times fixed.')

    # Fix seconds to zero
    invalid_seconds = data_df[data_df[constants.TIMESTAMP_COL].dt.second != 0]
    if invalid_seconds.shape[0] > 0:
        if verbose > 0:
            print('There are invalid seconds in the data:', invalid_seconds.shape[0])
            print('  Fixing the invalid seconds...')
        data_df[constants.TIMESTAMP_COL] = data_df[constants.TIMESTAMP_COL].apply(lambda x: x.replace(second=0))
        if verbose > 0:
            print('  Invalid seconds fixed.')

    # Remove zero customer outages initially (we'll handle gaps separately)
    if data_df[data_df[constants.CUSTOMERS_COL] == 0].shape[0] > 0:
        if verbose > 0:
            print(f'There are {data_df[data_df[constants.CUSTOMERS_COL] == 0].shape[0]} records where {constants.CUSTOMERS_COL} == 0')
            print(f'  Removing the {constants.CUSTOMERS_COL} == 0 records...')
        data_df = data_df[data_df[constants.CUSTOMERS_COL] != 0].reset_index(drop=True)
        if verbose > 0:
            print(f'  Records removed with {constants.CUSTOMERS_COL} == 0.')

    # ==================== Export Cleaned Data ====================

    try:
        data_df.to_parquet(output_file_path, index=False)
        county_total_customers.to_parquet(os.path.join(output_file_dir, output_county_total_customers_file_name), index=False)
        print(f"\nCleaned data saved to {output_file_name}")
        print(f"and county total customers data saved to {output_county_total_customers_file_name}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
    
    return data_df, county_total_customers


def clean_eaglei_data(eaglei_df: pd.DataFrame, 
                      customer_column: str = constants.CUSTOMERS_COL,
                      timestamp_column: str = constants.TIMESTAMP_COL, 
                      verbose: int = 1) -> pd.DataFrame:
    """
    Cleans EAGLE-i outage data by standardizing timestamps, removing zero customer outages,
    and filling in gaps in the time series.
    
    Parameters:
    -----------
    eaglei_df : pd.DataFrame
        DataFrame containing EAGLE-i outage data
    customer_column : str, default=constants.CUSTOMERS_COL
        Name of the column containing customer outage counts
    timestamp_column : str, default=constants.TIMESTAMP_COL
        Name of the column containing timestamps
    verbose : int, default=1
        Verbosity level for logging
        - 0: No print outputs
        - 1: Basic print outputs (such as warnings and summaries)
        - 2: Detailed print outputs (including all intermediate steps)

    Returns:
    --------
    pd.DataFrame
        DataFrame with cleaned and standardized EAGLE-i outage data
    
    Notes:
    -----------
    This function performs the following operations:
    - Checks if the data is sorted by the timestamp column and sorts it if necessary.
    - Standardizes timestamps to 15-minute intervals.
    - Fixes invalid timestamps that do not align with 15-minute intervals.
    - Fixes seconds to zero in the timestamp column.
    - Removes records where the customer outage count is zero, except for gaps in the time series
      which will be handled separately.
    - Returns a cleaned DataFrame.
    """

    # Create a copy to avoid modifying the original dataframe
    df_cleaned = eaglei_df.copy()
    
    # Check if the data is sorted by timestamp_column
    if not df_cleaned[timestamp_column].is_monotonic_increasing:
        if verbose > 0:
            print(f'Data was not sorted by {timestamp_column}, sorting the data...')
        df_cleaned = df_cleaned.sort_values(by=timestamp_column).reset_index(drop=True)

    # Standardize timestamps to 15-minute intervals
    invalid_times = df_cleaned[~(df_cleaned[timestamp_column].dt.minute.isin([0, 15, 30, 45]))]
    if invalid_times.shape[0] > 0:
        if verbose > 0:
            print('There are invalid times in the data:', invalid_times.shape[0])
            print('  Fixing the invalid times...')
        df_cleaned[timestamp_column] = df_cleaned[timestamp_column].apply(
            lambda x: x.replace(minute=(x.minute // 15) * 15, second=0)
        )
        if verbose > 0:
            print('  Invalid times fixed.')

    # Fix seconds to zero
    invalid_seconds = df_cleaned[df_cleaned[timestamp_column].dt.second != 0]
    if invalid_seconds.shape[0] > 0:
        if verbose > 0:
            print('There are invalid seconds in the data:', invalid_seconds.shape[0])
            print('  Fixing the invalid seconds...')
        df_cleaned[timestamp_column] = df_cleaned[timestamp_column].apply(lambda x: x.replace(second=0))
        if verbose > 0:
            print('  Invalid seconds fixed.')

    # Remove zero customer outages initially (we'll handle gaps separately)
    if df_cleaned[df_cleaned[customer_column] == 0].shape[0] > 0:
        if verbose > 0:
            print(f'There are {df_cleaned[df_cleaned[customer_column] == 0].shape[0]} records where {customer_column} == 0')
            print(f'  Removing the {customer_column} == 0 records...')
        df_cleaned = df_cleaned[df_cleaned[customer_column] != 0].reset_index(drop=True)
        if verbose > 0:
            print(f'  Records removed with {customer_column} == 0.')

    return df_cleaned


def identify_and_rank_time_gaps(df: pd.DataFrame, 
                                timestamp_column: str = constants.TIMESTAMP_COL, 
                                customer_column: str = constants.CUSTOMERS_COL, 
                                max_gap_minutes: int = (24*60), 
                                min_customers_before_gap: int = 10, 
                                min_customers_after_gap: int = 2, 
                                ranking_method: str = 'customer_weighted',
                                verbose: int = 1):
    """
    Identify time gaps in a dataframe and rank them based on neighboring customer counts using sophisticated ranking algorithms.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing timestamp and customer data
    timestamp_column : str, default=constants.TIMESTAMP_COL
        Name of the timestamp column
    customer_column : str, default=constants.CUSTOMERS_COL
        Name of the customer count column
    max_gap_minutes : int, default=24*60 (24 hours)
        Maximum gap duration in minutes to consider (gaps longer than this are ignored)
    min_customers_before_gap : int, default=10
        Minimum number of customers before the gap to consider it significant
    min_customers_after_gap : int, default=2
        Minimum number of customers after the gap to consider it significant
    ranking_method : str, default='customer_weighted'
        Ranking method to use. Options:
        - 'weighted_composite': Weighted combination of multiple factors
        - 'customer_weighted': Focus on customer impact
        - 'time_weighted': Focus on time duration
        - 'hybrid_impact': Hybrid approach considering both impact and duration
    verbose : int, default=1
        Verbosity level for logging
        - 0: No print outputs
        - 1: Basic print outputs (such as warnings and summaries)
        - 2: Detailed print outputs (including all intermediate steps)

    Returns:
    --------
    pd.DataFrame
        DataFrame with gap information, neighboring customer counts, and rankings
    """
    
    # Sort the dataframe by timestamp
    df_sorted = df.sort_values(by=timestamp_column).reset_index(drop=True)
    
    # Calculate time differences between consecutive records
    time_diffs = df_sorted[timestamp_column].diff()
    time_diffs_minutes = time_diffs.dt.total_seconds() / 60
    
    # Identify gaps (where time difference > 15 minutes)
    gap_mask = time_diffs_minutes > 15
    gap_indices = gap_mask[gap_mask].index.tolist()
    
    # Filter gaps based on minimum duration
    filtered_gaps = [(i, time_diffs_minutes[i]) for i in gap_indices 
                       if time_diffs_minutes[i] <= max_gap_minutes]
    
    if not filtered_gaps:
        if verbose > 0:
            print(f"No gaps found with duration <= {max_gap_minutes} minutes")
        return pd.DataFrame()
    
    # Filter gaps based on minimum customers before the gap
    significant_gaps = [(i, duration) for i, duration in filtered_gaps
                        if i > 0 and df_sorted.iloc[i-1][customer_column] >= min_customers_before_gap]
    
    if not significant_gaps:
        if verbose > 0:
            print(f"No significant gaps found with at least {min_customers_before_gap} customers before the gap")
        return pd.DataFrame()
    
    # Collect gap information
    gap_data = []
    
    for gap_idx, gap_duration_minutes in significant_gaps:
        # Get neighboring records
        before_idx = gap_idx - 1  # Record before the gap
        after_idx = gap_idx       # Record after the gap
        
        # Extract information about the gap
        gap_info = {
            'gap_index': gap_idx,
            'gap_duration_minutes': gap_duration_minutes,
            'gap_duration_intervals': gap_duration_minutes / 15,  # Number of 15-min intervals
            'timestamp_before': df_sorted.iloc[before_idx][timestamp_column],
            'timestamp_after': df_sorted.iloc[after_idx][timestamp_column],
            'customers_before': df_sorted.iloc[before_idx][customer_column],
            'customers_after': df_sorted.iloc[after_idx][customer_column],
        }
        
        # Calculate various customer-based metrics
        gap_info.update(_calculate_gap_metrics(gap_info))
        
        gap_data.append(gap_info)
    
    # Convert to DataFrame
    gaps_df = pd.DataFrame(gap_data)

    # Apply minimum customers after the gap filter
    gaps_df = gaps_df[gaps_df['customers_after'] >= min_customers_after_gap].reset_index(drop=True)
    
    if len(gaps_df) == 0:
        if verbose > 0:
            print("No gaps found to rank.")
        return pd.DataFrame()
    else:
        # Apply ranking based on selected method
        gaps_df = _apply_gap_ranking(gaps_df, ranking_method)

        if verbose > 0:
            print(f"Found {len(gaps_df)} significant gaps (gap duration <= {max_gap_minutes} minutes, AND at least {min_customers_before_gap} customers before gap, AND at least {min_customers_after_gap} customers after gap)")
            print(f"Ranking method used: {ranking_method}")

        return gaps_df


def _calculate_gap_metrics(gap_info: Dict) -> Dict:
    """
    Calculate various metrics for a single gap based on neighboring customer counts.
    
    Parameters:
    -----------
    gap_info : Dict
        Dictionary containing information about the gap, including:
        - 'customers_before': Number of customers before the gap
        - 'customers_after': Number of customers after the gap  
    
    Returns:
    --------
    Dict
        Dictionary containing calculated metrics for the gap.
    """
    
    customers_before = gap_info['customers_before']
    customers_after = gap_info['customers_after']
    
    # Basic customer metrics
    customer_sum = customers_before + customers_after
    customer_max = max(customers_before, customers_after)
    customer_min = min(customers_before, customers_after)
    customer_avg = customer_sum / 2
    
    # Customer consistency (how similar the neighboring values are)
    if customer_max > 0:
        customer_consistency =  (customer_min / customer_max)
    else:
        customer_consistency = 1
    
    return {
        'customer_avg': customer_avg,
        'customer_consistency': customer_consistency
    }


def _apply_gap_ranking(gaps_df: pd.DataFrame, 
                       ranking_method: str = 'customer_weighted') -> pd.DataFrame:
    """
    Applying ranking algorithms to the gaps DataFrame.
    
    Parameters:
    -----------
    gaps_df : pd.DataFrame
        DataFrame containing gap information and metrics
    ranking_method : str, default='customer_weighted'
        Ranking method to use. Options:
        - 'customer_weighted': Focus on customer impact
        - 'time_weighted': Focus on time duration
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added ranking information
    """
    
    # Normalize metrics to 0-1 range for fair comparison
    if 'gap_duration_minutes' in gaps_df.columns:
        gaps_df['gap_duration_minutes_normalized'] = np.log1p(gaps_df['gap_duration_minutes']) / np.log1p(gaps_df['gap_duration_minutes'].max())
    else:
        gaps_df['gap_duration_minutes_normalized'] = 0

    # Since the customer_avg has some very large values, we will normalize it separately using a log scale
    if 'customer_avg' in gaps_df.columns:
        gaps_df['customer_avg_normalized'] = np.log1p(gaps_df['customer_avg']) / np.log1p(gaps_df['customer_avg'].max())
    else:
        gaps_df['customer_avg_normalized'] = 0
    
    if ranking_method == 'customer_weighted':
        # Focus primarily on customer impact
        gaps_df['rank'] = (
            (0.1 * gaps_df['customer_avg_normalized']) +
            (0.2 * gaps_df['customer_consistency']) + 
            (1 * (1 - gaps_df['gap_duration_minutes_normalized']) ** 1)  # Penalize longer gaps
        )
        
    elif ranking_method == 'time_weighted':
        # Focus primarily on time duration with customer weighting
        gaps_df['rank'] = (
            0.6 * gaps_df['gap_duration_minutes_normalized'] +
            0.4 * gaps_df['customer_avg_normalized']
        )
        
    else:
        raise ValueError(f"Unknown ranking method: {ranking_method}")
    
    # Add ranking position
    gaps_df['rank_position'] = gaps_df['rank'].rank(ascending=False, method='dense').astype(int)

    # Sort by rank (higher rank = higher priority)
    gaps_df = gaps_df.sort_values('rank', ascending=False).reset_index(drop=True)
    
    return gaps_df


def analyze_gap_rankings(gaps_df: pd.DataFrame, 
                         top_n: int = 10, 
                         verbose: int = 1) -> None:
    """
    Analyze and display the top-ranked gaps with detailed information.
    
    Parameters:
    -----------
    gaps_df : pd.DataFrame
        DataFrame with ranked gaps from identify_and_rank_time_gaps
    top_n : int, default=10
        Number of top gaps to display
    verbose : int, default=1
        Verbosity level for logging
        - 0: No print outputs
        - 1: Basic print outputs
        - 2: Detailed print outputs (including all intermediate steps)
    
    Returns:
    --------
    None
    """
    
    if len(gaps_df) == 0:
        print("No gaps to analyze")
        return
    
    if verbose > 1:
        print(f"\n=== TOP {top_n} RANKED GAPS ===")
        print(f"{'Rank':<5} {'Duration':<10} {'Before':<8} {'After':<8} {'Avg (Norm.)':<12} {'Consistency':<12} {'Timestamp Before':<20}")
        print("-" * 85)
    
        for idx, row in gaps_df.head(top_n).iterrows():
            print(f"{row['rank_position']:<5} "
                f"{row['gap_duration_minutes']:<10.0f} "
                f"{row['customers_before']:<8.0f} "
                f"{row['customers_after']:<8.0f} "
                f"{row['customer_avg_normalized']:<12.3f} "
                f"{row['customer_consistency']:<12.3f} "
                f"{row['timestamp_before'].strftime('%Y-%m-%d %H:%M'):<20}")
    
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total gaps analyzed: {len(gaps_df)}")
        print(f"Average gap duration: {gaps_df['gap_duration_minutes'].mean():.1f} minutes")
        print(f"Maximum gap duration: {gaps_df['gap_duration_minutes'].max():.1f} minutes")
        print(f"Average customers before gap: {gaps_df['customers_before'].mean():.1f}")
        print(f"Average customers after gap: {gaps_df['customers_after'].mean():.1f}")
    
    # Gap duration distribution
    duration_bins = [0, 30, 60, 120, 240, 480, float('inf')]
    duration_labels = ['<30min', '30-60min', '1-2hrs', '2-4hrs', '4-8hrs', '>8hrs']
    
    gaps_df['duration_category'] = pd.cut(gaps_df['gap_duration_minutes'], 
                                         bins=duration_bins, 
                                         labels=duration_labels, 
                                         include_lowest=True)
    
    duration_counts = gaps_df['duration_category'].value_counts().sort_index()

    if verbose > 1:
        print(f"\n=== GAP DURATION DISTRIBUTION ===")
        for category, count in duration_counts.items():
            print(f"{category}: {count} gaps")
        
    return None


def visualize_gap_analysis(df: pd.DataFrame, rank_quantile: float | None = None) -> None:
    """
    Create comprehensive visualizations for gap analysis results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ranked gaps from identify_and_rank_time_gaps
    rank_quantile : float, optional
        Quantile to determine rank threshold for highlighting (e.g., 0.5 for median)
        If None, median (0.5) is used as default.

    Returns:
    --------
    None
    """
    
    if len(df) == 0:
        print("No gaps to visualize")
        return
    
    if rank_quantile is None:
        print("No rank quantile provided, using median (0.5) as default")
        rank_quantile = 0.5

    # Determine rank threshold based on quantile
    rank_threshold = df['rank'].quantile(rank_quantile)
    
    gaps_df = df.copy()
    gaps_df['Above Threshold'] = gaps_df['rank'] > rank_threshold
    gaps_df.rename(columns={'rank': 'Rank Score'}, inplace=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f'Time Gap Analysis and Ranking - Selected Rank Threshold = {rank_threshold:.2f} (Quantile = {rank_quantile:.2f})', fontsize=16)

    # 1. Gap duration vs Customer Consistency
    sns.scatterplot(ax = axes[0, 0],
                    data=gaps_df, 
                    x='gap_duration_minutes_normalized', 
                    y='customer_consistency', 
                    hue='Rank Score',
                    palette='viridis_r',
                    style='Above Threshold',
                    markers={True: 'o', False: 'P'},
                    alpha=0.8, 
                    s=80)
    axes[0, 0].set_xlabel('Gap Duration (Normalized)')
    axes[0, 0].set_ylabel('Customer Consistency')
    axes[0, 0].set_title('Gap Duration (Normalized) vs Customer Consistency')
    axes[0, 0].legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')
    
    # 2. Gap duration vs Customer average
    sns.scatterplot(ax = axes[1, 0],
                    data=gaps_df, 
                    x='gap_duration_minutes_normalized', 
                    y='customer_avg_normalized', 
                    hue='Rank Score',
                    palette='viridis_r',
                    style='Above Threshold',
                    markers={True: 'o', False: 'P'},
                    alpha=0.8, 
                    s=80)
    axes[1, 0].set_xlabel('Gap Duration (Normalized)')
    axes[1, 0].set_ylabel('Average of Customers (Before & After the Gap)')
    axes[1, 0].set_title('Gap Duration (Normalized) vs Customer Average (Normalized)')
    axes[1, 0].legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')

    # 3. Gap duration histogram
    axes[0, 1].hist(gaps_df['gap_duration_minutes'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Gap Duration (minutes)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Gap Duration Distribution')

    # 4. Rank Score histogram
    axes[1, 1].hist(gaps_df['Rank Score'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Rank Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Rank Score Distribution')
    plt.tight_layout()
    plt.show()


    # Create a figure and a 3D axes object
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gaps_df['gap_duration_minutes_normalized'], gaps_df['customer_consistency'], gaps_df['customer_avg_normalized'], c=gaps_df['Rank Score'], s=50, alpha=0.8, cmap='viridis_r')
    ax.set_xlabel('Gap Duration (Normalized)')
    ax.set_ylabel('Customer Consistency')
    ax.set_zlabel('Average of Customers (Normalized)')
    ax.set_title('3D Scatter Plot')
    # show the legend
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1, aspect=20, orientation='vertical', shrink=0.8)
    cbar.set_label('Rank Score')
    # Set the view angle for better visualization
    ax.view_init(elev=30, azim=-45)
    # Display the plot
    plt.show()


def fill_data_gaps_eaglei(eaglei_df: pd.DataFrame, 
                          gaps_df: pd.DataFrame, 
                          timestamp_column: str = constants.TIMESTAMP_COL, 
                          rank_threshold: float = 0.35, 
                          verbose: int = 1) -> pd.DataFrame:
    """
    Fill data gaps in EAGLE-i outage data with preceding values.
    
    This function addresses missing data records during large outage events by filling gaps
    with the immediately preceding customer outage values. It uses the identified gaps
    from the `gaps_df` DataFrame, which contains the indices of gaps, timestamps before and after the gaps, and their ranks.
    
    Parameters:
    -----------
    eaglei_df : pd.DataFrame
        DataFrame containing EAGLE-i outage data
    gaps_df : pd.DataFrame
        DataFrame containing identified gaps with columns:
        - 'gap_index': Index of the gap in eaglei_df
        - 'timestamp_before': Timestamp before the gap
        - 'timestamp_after': Timestamp after the gap
        - 'rank': Rank of the gap based on its characteristics
    timestamp_column : str, default=constants.TIMESTAMP_COL
        Name of the column containing timestamps
    rank_threshold : float, default=0.35
        Threshold for gap ranking to determine which gaps to fill
    verbose : int, default=1
        Verbosity level for logging
        - 0: No print outputs
        - 1: Basic print outputs (such as warnings and summaries)
        - 2: Detailed print outputs (including all intermediate steps)

    Returns:
    --------
    pd.DataFrame
        DataFrame with filled data gaps, maintaining original structure but with additional records
    """
    
    if eaglei_df['county'].nunique() > 1:
        print("Warning: The input DataFrame contains data from multiple counties. This function is designed to work with a single county's data.")
        return eaglei_df  # Return original DataFrame if multiple counties are present

    gaps_to_fill = gaps_df[gaps_df['rank'] > rank_threshold].copy()

    # Create a list to store filled records
    filled_records = []

    for row in gaps_to_fill.itertuples():
        if (row.gap_index == 0) or (row.gap_index >= len(eaglei_df)):
            if verbose > 0:
                print(f"Skipping gap at index {row.gap_index} due to invalid index.")
            continue
        current_record = eaglei_df.iloc[row.gap_index - 1]  # Get the record before the gap
        gap_start_time = (row.timestamp_before) + pd.Timedelta(minutes=15)  # Start filling from the next 15-minute interval
        gap_end_time = (row.timestamp_after) - pd.Timedelta(minutes=15)  # End filling at the previous 15-minute interval
        # create a time range for the gap
        gap_time_range = pd.date_range(start=gap_start_time, end=gap_end_time, freq='15min')
        # Fill the gap with preceding values
        for gap_time in gap_time_range:
            gap_record = current_record.copy()
            gap_record[timestamp_column] = gap_time
            gap_record['filled_gap'] = True  # Mark as a filled record
            filled_records.append(gap_record)
        
    # Convert filled records to DataFrame
    filled_df = pd.DataFrame(filled_records)

    # Check if any of the timestamp values in filled_df are already present in eaglei_df
    if eaglei_df[eaglei_df[timestamp_column].isin(filled_df[timestamp_column])].shape[0] > 0:
        print("Warning: Some filled timestamps already exist in the original data. This may lead to duplicate records.")
        print("Aborting filling process to avoid duplicates.")
        return eaglei_df  # Return original DataFrame if duplicates found

    # Add original records that were not filled
    original_records = eaglei_df[~eaglei_df[timestamp_column].isin(filled_df[timestamp_column])]
    original_records['filled_gap'] = None  # Mark as original records

    gaps_NOT_to_fill = gaps_df[gaps_df['rank'] <= rank_threshold]
    original_records.loc[((original_records[timestamp_column].isin(gaps_NOT_to_fill['timestamp_before'])) | 
                     (original_records[timestamp_column].isin(gaps_NOT_to_fill['timestamp_after']))), 'filled_gap'] = False  # Mark as not filled

    filled_df = pd.concat([filled_df, original_records], ignore_index=True)
    # Sort by timestamp and reset index
    filled_df = filled_df.sort_values(by=timestamp_column).reset_index(drop=True)

    if verbose > 0:
        print(f"\nData gap filling completed:")
        print(f"  New Gap-filled records created: {filled_df['filled_gap'].sum()}")

    # Check if the data is sorted by timestamp_column
    if not filled_df[timestamp_column].is_monotonic_increasing:
        filled_df = filled_df.sort_values(by=timestamp_column).reset_index(drop=True)

    return filled_df


def plot_eaglei_filled_gaps(df: pd.DataFrame, 
                            customer_column: str = constants.CUSTOMERS_COL, 
                            timestamp_column: str = constants.TIMESTAMP_COL):
    """
    Plots the EAGLE-I outages data with a step plot and scatter points for original, filled, and missing data.
    This function is used to visualize the outages in the EAGLE-I dataset, before converting them to events and 
    extracting the outage, restore and perofmance curves.
    Main utility of this function is to visualize the gaps in the EAGLE-I data, and how the gap filling
    algorithm fill those gaps.

    Parameters:
    - df: DataFrame containing the EAGLE-I outages data.
    - customer_column: Name of the column containing customer data (default is constants.CUSTOMERS_COL).
    - timestamp_column: Name of the column containing timestamps (default is constants.TIMESTAMP_COL).

    Returns:
    - None
    """
    
    if len(df) == 0:
        print("DataFrame is empty. No data to plot.")
        return None
    if len(df) > 5000:
        print("DataFrame has more than 1000 records. Plot will be cluttered.")
        return None

    # Make a copy to avoid modifying original data
    df_plot = df.copy()
    
    # Ensure run_start_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_plot[timestamp_column]):
        print(f"Converting {timestamp_column} to datetime format...")
        df_plot[timestamp_column] = pd.to_datetime(df_plot[timestamp_column])
    
    # Sort by time if not already sorted
    if not df_plot[timestamp_column].is_monotonic_increasing:
        print(f"Sorting {timestamp_column} in ascending order...")
        df_plot = df_plot.sort_values(by=timestamp_column).reset_index(drop=True)

    # Create a 15-minute frequency time series using the minimum and maximum timestamps values in the timestamp column
    min_time = df_plot[timestamp_column].min()
    max_time = df_plot[timestamp_column].max()
    timeSeries = pd.date_range(start=min_time, end=max_time, freq='15min')
    # Create a new DataFrame with the time series
    df_time_series = pd.DataFrame(timeSeries, columns=[timestamp_column])
    # Merge the time series with the original data
    df_plot = pd.merge(df_time_series, df_plot, on=timestamp_column, how='left')
    # Fill NaN values in the customer column with 0
    df_plot[customer_column] = df_plot[customer_column].fillna(0)

    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    if 'filled_gap' in df_plot.columns:
        # Plot only the original data
        original_data = df_plot[(df_plot[customer_column] > 0) & (df_plot['filled_gap'] != True)]
        if not original_data.empty:
            # Plot original data
            plt.scatter(original_data[timestamp_column], original_data[customer_column], 
                    alpha=1.0, s=30, label='Original data', color='#1f77b4')
            
        # Plot filled data
        filled_only = df_plot[df_plot['filled_gap'] == True]
        if not filled_only.empty:
            plt.scatter(filled_only[timestamp_column], filled_only[customer_column], 
                       alpha=1.0, s=30, label='Gap-filled data', color="#CE07C4", marker='D')
    else:
        # Plot only the original data
        original_data = df_plot[df_plot[customer_column] > 0]
        if not original_data.empty:
            # Plot original data
            plt.scatter(original_data[timestamp_column], original_data[customer_column], 
                    alpha=1.0, s=30, label='Original data', color='#1f77b4')
            
    # Plot missing data
    missing_data = df_plot[df_plot[customer_column] == 0]
    if not missing_data.empty:
        # Plot missing data
        plt.scatter(missing_data[timestamp_column], missing_data[customer_column], 
                   alpha=1.0, s=30, label='Missing data', color="#ce0909", marker='x')
    
    # Plot step plot
    plt.step(df_plot[timestamp_column], df_plot[customer_column], where='post', color='#ff7f0e', alpha=1.0, label='Performance Curve', linewidth=1.5)
    
    plt.xlabel('Time')
    plt.ylabel('Customers Out')
    if df_plot['county'].nunique() > 1:
        plt.title(f'EAGLE-I Data from {df_plot[timestamp_column].min().date():%Y-%m-%d %H:%M} to {df_plot[timestamp_column].max().date():%Y-%m-%d %H:%M} for Multiple Counties')
    else:
        plt.title(f'EAGLE-I Data from {df_plot[timestamp_column].min().date():%Y-%m-%d %H:%M} to {df_plot[timestamp_column].max().date():%Y-%m-%d %H:%M} for {df_plot["county"].iloc[0]} County, {df_plot["state"].iloc[0]}')
    plt.legend()
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, alpha=0.1)
    # space the vertical grid lines at 15-minute intervals
    # plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=120))  # Set major ticks at 120-minute intervals
    # format the x-axis labels to show date and time
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()  # Rotate x-axis labels for better readability
    plt.show()
    
    return None


# ------------------------- Event Extraction -------------------------


def extract_events_eaglei_ac(outage_df: pd.DataFrame, 
                             time_delta: int = 15, 
                             timestamp_column: str = constants.TIMESTAMP_COL) -> pd.DataFrame:
    """
    Extract outage events from the EAGLE-I dataset using the AC method.
    This is the very basic version of the event extraction, which simply groups the outages into events based on a missing 15-minute interval.
    The data has to be cleaned and the missing gaps should be filled before using this function.
    
    Parameters:
    - outage_df: The input DataFrame containing outage data.
    - time_delta: The time interval (in minutes) to consider for grouping outages.
    - timestamp_column: The name of the column containing timestamp information.

    Returns:
    - pd.DataFrame: DataFrame with an additional column 'event_number_ac' indicating event numbers.
    """
    # check if timestamp column exists
    if timestamp_column not in outage_df.columns:
        raise ValueError(f'{timestamp_column} column is missing in the DataFrame')
    
    # check if data is sorted by run_start_time
    if not outage_df[timestamp_column].is_monotonic_increasing:
        raise ValueError(f'Data is not sorted by {timestamp_column}. Sort the data first!')
    
    # check if filled_gap column exists
    if 'filled_gap' not in outage_df.columns:
        raise ValueError('filled_gap column is missing in the DataFrame. Clean the data first!')
    
    # reset the index
    outage_df_copy = outage_df.reset_index(drop=True, inplace=False)
    event_ids = []
    current_event_id = 1
    
    # iterate through the rows of the outage data
    for i in range(0, len(outage_df_copy)-1):
        event_ids.append(current_event_id)
        if outage_df_copy.loc[i+1, timestamp_column] != (outage_df_copy.loc[i, timestamp_column] + timedelta(minutes=time_delta)):
            current_event_id += 1
    
    # add the last event id
    event_ids.append(current_event_id)

    # add the event numbers to the dataframe
    outage_df_copy['event_number_ac'] = event_ids

    return outage_df_copy


def extract_events_eaglei_ac_threshold(outage_df: pd.DataFrame,
                                        event_detection_type: str = "flat",
                                        total_customers: int = 0,
                                        customer_threshold: float = 10,
                                        time_delta: str = "15min",
                                        timestamp_column: str = constants.TIMESTAMP_COL,
                                        customer_column: str = constants.CUSTOMERS_COL,
                                        active_only: bool = False,
                                        crossing_mode: str = "both") -> pd.DataFrame:
    """
    Detects events in the EAGLE-i DataFrame based on customer outage thresholds and time intervals.
    An event is defined as a continuous period where the number of customer outages MEETS or EXCEEDS
    a specified threshold, with gaps in time series data handled appropriately.

    Parameters:
    - outage_df (pd.DataFrame): DataFrame containing time series data with customer outages.
    - event_detection_type (str): Method for selecting number of customers on outage threshold.
    - customer_threshold (int): Threshold for customer outages to consider an event active.
    - time_delta (str): Time interval for checking continuity (e.g., "15min").
    - timestamp_column (str): Name of the column containing timestamps.
    - customer_column (str): Name of the column containing customer outage counts.
    - active_only (bool): If True, only label events where customer outages exceed the threshold.
    - crossing_mode (str): Mode for detecting status changes ("both" or "down").

    Returns:
    - pd.DataFrame: DataFrame with an additional column indicating event numbers.
    """
    # Check if customer_column exists
    if customer_column not in outage_df.columns:
        raise ValueError(f'{customer_column} column is missing in the DataFrame')

    # Check if timestamp_column exists
    if timestamp_column not in outage_df.columns:
        raise ValueError(f'{timestamp_column} column is missing in the DataFrame')

    # Validate crossing_mode
    if crossing_mode not in {"both", "down"}:
        raise ValueError("crossing_mode must be either 'both' or 'down'")
    
    # Check if timestamp column exists
    if 'filled_gap' not in outage_df.columns:
        raise ValueError('filled_gap column is missing in the DataFrame. Clean the data first!')

    df = outage_df.reset_index(drop=True, inplace=False)

    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    if not outage_df[timestamp_column].is_monotonic_increasing:
        print(f'Data is not sorted by {timestamp_column}. Sorting the data first')
        df = df.sort_values(timestamp_column).reset_index(drop=True)

    # Determine detection method - which changes thresholding approach
    if event_detection_type=='percentile':
        customer_threshold=df['customers_out'].quantile(customer_threshold)
        print(f'Using percentile based event detection: Events defined '
              f'as those greater than {customer_threshold} customers.')
    elif event_detection_type=='percent_customers':
        if total_customers != 0:
            customer_threshold=round(customer_threshold*int(total_customers))
            print(f'Defining events as a percent of customers in the county out: Events defined '
                  f'as those greater than {customer_threshold} customers.')
        else:
            print("Error using percent customers detection method: Defining events using the flat method.")
            customer_threshold=30
            print(f'Events defined as those greater than {customer_threshold} customers.')
    elif event_detection_type=='flat':
        print(f'Defining events as a flat value: Events defined '
              f'as those greater than {customer_threshold} customers.')
    else:
        print('Invalid event detection method. Using a flat 30 customers threshold for event definition.')
        customer_threshold=30

    event_col = f'event_number_ac_threshold_{event_detection_type}_{customer_threshold}'

    n = len(df)
    if n == 0:
        return df.assign(**{event_col: pd.Series(dtype="Int64" if active_only else "int")})

    freq_td = pd.Timedelta(time_delta)

    actual_next = df[timestamp_column].shift(-1)
    expected_next = df[timestamp_column] + freq_td
    gap = (actual_next != expected_next).fillna(True)

    status = (df[customer_column] >= customer_threshold)
    status_next = status.shift(-1)

    if crossing_mode == "both":
        status_change = (status != status_next).fillna(True)
    elif crossing_mode == "down":
        status_change = ((status == True) & (status_next == False)).fillna(True)

    end_of_event = gap | status_change
    end_of_event.iloc[-1] = True

    starts = end_of_event.shift(1, fill_value=True)
    event_ids = starts.cumsum().astype(int)

    df[event_col] = event_ids

    if active_only:
        # df[event_col] = df[event_col].where(status, other=pd.NA).astype("Int64")
        df[event_col] = df[event_col].where(status, other=-1).astype("Int64")

    return df


def count_crossings(df: pd.DataFrame,
                    value_col: str = constants.CUSTOMERS_COL,
                    timestamp_col: str = constants.TIMESTAMP_COL,
                    threshold: int = 0,
                    crossing: str = "down") -> int:
    """
    Count the number of threshold crossings in a time series.
    Handles missing 15-minute intervals by filling with 0.

    Crossing types:
      - "down": current >= threshold and next < threshold
      - "up":   current < threshold and next >= threshold
      - "both": counts both upward and downward crossings

    Note that the number of "up" crossings may not equal
    the number of "down" crossings if the series starts
    or ends above or below the threshold.
    But other than that the number of "up" and "down"
    crossings should be equal and the "both" count should be double.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain timestamp and value columns.
    value_col : str
        Name of column with numeric values.
    timestamp_col : str
        Name of column with timestamps.
    threshold : int
        Threshold for crossings.
    crossing : {"down", "up", "both"}
        Type of crossing to count.

    Returns
    -------
    int
        Number of crossings.
    """
    if crossing not in {"down", "up", "both"}:
        raise ValueError("crossing must be one of {'down', 'up', 'both'}")

    if value_col not in df.columns or timestamp_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{timestamp_col}' and '{value_col}'")

    # Ensure datetime
    ts = pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df[timestamp_col] = ts
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # Check sampling interval
    diffs = df[timestamp_col].diff().dropna()
    if not (diffs.mode().iloc[0] == pd.Timedelta("15min")):
        raise ValueError("Data is not sampled at consistent 15-minute intervals")

    # Reindex to continuous 15min grid, fill missing with 0
    full_index = pd.date_range(start=df[timestamp_col].iloc[0],
                               end=df[timestamp_col].iloc[-1],
                               freq="15min")
    df = df.set_index(timestamp_col).reindex(full_index)
    df[value_col] = df[value_col].fillna(0)
    df = df.reset_index().rename(columns={"index": timestamp_col})

    # Crossing detection
    series = df[value_col].astype(float)
    current = series[:-1].values
    nxt = series[1:].values

    if crossing == "down":
        crossings = (current >= threshold) & (nxt < threshold)
    elif crossing == "up":
        crossings = (current < threshold) & (nxt >= threshold)
    elif crossing == "both":
        crossings = ((current >= threshold) & (nxt < threshold)) | \
                    ((current < threshold) & (nxt >= threshold))

    return int(crossings.sum())


# ------------------------- Event Processes And Statistics -------------------------


def get_eaglei_processes(outage_df: pd.DataFrame, 
                         event_number: int, 
                         event_method: str = 'ac', 
                         timestamp_column: str = constants.TIMESTAMP_COL, 
                         customer_column: str = constants.CUSTOMERS_COL) -> Tuple[List, List, List]:
    """
    Function to get the outage, restore, and performance processes for a given event number

    Parameters:
        outage_df: the outage data frame
        event_number: the event number to get the processes for
        event_method: the method used to extract the event number (default is 'ac')
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)
    Returns:
        A tuple containing three lists:
            - outages: list of tuples (timestamp, customers_out) for outages
            - restores: list of tuples (timestamp, customers_restored) for restorations
            - performance_data: list of tuples (timestamp, customers_out) for performance curve
    """

    event_column = f'event_number_{event_method}'
    event_data = outage_df[outage_df[event_column] == event_number].copy()

    # Check if the timestamp_column has any duplicate values
    if event_data[timestamp_column].duplicated().any():
        raise ValueError(f"Warning: The timestamp column '{timestamp_column}' has duplicate values.\n"
                         f"This may lead to incorrect event processing.\n"
                         f"This might be due to multiple counties being present in the data.\n"
                         f"If this is the case, try using the plot_eaglei_multicounty_performance_curve function.")
    
    event_data = event_data.sort_values(by=timestamp_column).reset_index(drop=True)
    event_start_time = event_data[timestamp_column].min()
    tau = timedelta(minutes=15)  # 15-minute intervals 

    # Create performance data as a list of tuples (timestamp, customers_out)
    # Start with the event start time minus tau to include the first time step
    # and end with the maximum timestamp plus tau to include the last time step
    # This ensures we have a complete time series for the event
    performance_data = [tuple(v) for v in event_data[[timestamp_column, customer_column]].values]
    performance_data.insert(0, (event_start_time - tau, 0))  # Add 0 for the first time step
    performance_data.append((event_data[timestamp_column].max() + tau, 0))  # Add 0 for the last time step

    # Calculate the differences in customer outages between consecutive time steps
    # This will give us the change in outages over each 15-minute interval
    diffs = np.diff( [v[1] for v in performance_data])
    # Outages are positive changes
    outages = [(event_start_time + (i*tau), v) for i,v in enumerate(diffs) if v > 0]  
    # Restorations are negative changes
    restores = [(event_start_time + (i*tau), -v) for i,v in enumerate(diffs) if v < 0] 


    # Remove duplicate enties (based on customers out) in performance_data
    # This is important to ensure that we have a unique time series for the event
    indexes_to_remove = set()
    for i in range(1, len(performance_data)):
        if performance_data[i][1] == performance_data[i-1][1]:
            indexes_to_remove.add(i)
    performance_data = [v for i, v in enumerate(performance_data) if i not in indexes_to_remove]

    return outages, restores, performance_data


def _get_eaglei_event_stats_single_event(eaglei_df: pd.DataFrame, 
                                         event_number: int, 
                                         event_method: str = 'ac', 
                                         timestamp_column: str = constants.TIMESTAMP_COL, 
                                         customer_column: str = constants.CUSTOMERS_COL) -> Dict:
    """
    Function to get the statistics of a given event number
    
    Parameters:
        eaglei_df: the outage data frame
        event_number: the event number to get the statistics for
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)
        
    Returns:
        A dictionary with the following keys:
            - event_number: the event number
            - start_time: the start time of the event
            - end_time: the end time of the event
            - duration: the duration of the event in minutes
            - max_customers_out: the maximum number of customers out during the event
            - total_customers_out: the total number of customers out during the event
            - num_outages: the number of outages during the event
            - num_restores: the number of restores during the event
    """
    outages, restores, performance_process = get_eaglei_processes(eaglei_df, event_number, event_method, timestamp_column, customer_column)

    if len(performance_process) == 0:
        print(f'No performance process found for event number {event_number}')
        return {'event_number': event_number,
                'start_time': None,
                'end_time': None,
                'duration_hours': 0,
                'max_customers_out': 0,
                'total_customers_out': 0,
                'num_outages': 0,
                'num_restores': 0,
                'customer_hours': 0
               }
    
    start_time = performance_process[1][0]
    end_time = performance_process[-1][0]
    duration = (end_time - start_time).total_seconds() / 3600  # in hours
    max_customers_out = max([v[1] for v in performance_process])
    total_customers_out = sum([v[1] for v in outages])
    num_outages = len(outages)
    num_restores = len(restores)
    # customer_hours = sum([v[1] * 0.25 for v in performance_process])  # each interval is 15 minutes = 0.25 hours
    # calculate customer hours by multiplying the number of customers out by the duration of each interval in hours (calculating using successive time steps)
    customer_hours = 0
    for i in range(1, len(performance_process)):
        interval_duration = (performance_process[i][0] - performance_process[i-1][0]).total_seconds() / 3600  # in hours
        customer_hours += performance_process[i-1][1] * interval_duration

    
    return {
        'event_number': event_number,
        'start_time': start_time,
        'end_time': end_time,
        'duration_hours': duration,
        'max_customers_out': max_customers_out,
        'total_customers_out': int(total_customers_out),
        'num_outages': num_outages,
        'num_restores': num_restores,
        'customer_hours': customer_hours
    }


def get_eaglei_event_stats(eaglei_df: pd.DataFrame, 
                           event_numbers: Any, 
                           event_method: str = 'ac', 
                           timestamp_column: str = constants.TIMESTAMP_COL, 
                           customer_column: str = constants.CUSTOMERS_COL) -> pd.DataFrame | Dict:
    """
    Function to get event statistics for one or more event numbers
    
    Parameters:
        eaglei_df: the EAGLEi data frame
        event_numbers: a single event number or a list of event numbers
        event_method: the method used to extract the events (default is 'ac')
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)
        
    Returns:
        A DataFrame if multiple event numbers are provided, or a dictionary if a single event number is provided
    """

    if len(event_numbers) == 1:
        return _get_eaglei_event_stats_single_event(eaglei_df, event_numbers[0], event_method, timestamp_column, customer_column)
    else:
        # apply the function to all event numbers and create a DataFrame
        event_stats = []
        for event_number in event_numbers:
            stats = _get_eaglei_event_stats_single_event(eaglei_df, event_number, event_method, timestamp_column, customer_column)
            # only add the stats if start_time is not None (i.e., event exists)
            if stats['start_time'] is not None:
                event_stats.append(stats)
        event_stats_df = pd.DataFrame(event_stats)
        return event_stats_df


# ------------------------- Plotting Functions -------------------------


def plot_eaglei_event_curves(outage_df: pd.DataFrame, 
                             event_number: int, 
                             event_method: str = 'ac', 
                             timestamp_column: str = constants.TIMESTAMP_COL, 
                             customer_column: str = constants.CUSTOMERS_COL) -> None:
    """
    Function to plot the outage and restore processes for a given event number
    
    Parameters:
        outage_df: the outage data frame
        event_number: the event number to plot
        event_method: the method used to extract the events (default is 'ac')
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)

    Returns:
        None
    """
    outages, restores, performance_process = get_eaglei_processes(outage_df, event_number, event_method, timestamp_column, customer_column)

    outage_process = [(outages[i][0], v) for i, v in enumerate(np.cumsum([o[1] for o in outages]))]
    restore_process = [(restores[i][0], v) for i, v in enumerate(np.cumsum([r[1] for r in restores]))]
    
    # Add 0 for the first time step to the outage and restore processes
    outage_process.insert(0, (outage_process[0][0], 0))  
    restore_process.insert(0, (outage_process[0][0], 0))
    
    # Add the last time step to the outage process
    outage_process.append((restore_process[-1][0], restore_process[-1][1]))  
    
    # Remove the first time step of the performance process
    performance_process = performance_process[1:]  # Remove the first time step (0, 0)
    performance_process.insert(0, (performance_process[0][0], 0))  

    # create a step plot of the outages
    plt.figure(figsize=(10,7))
    plt.step([row[0] for row in outage_process], [row[1] for row in outage_process], where='post', label='Outage Curve', color=constants.COLOR_OUTAGE_CURVE)
    plt.step([row[0] for row in restore_process], [row[1] for row in restore_process], where='post', label='Restore Curve', color=constants.COLOR_RESTORE_CURVE)
    plt.step([row[0] for row in performance_process], [-row[1] for row in performance_process], where='post', label='Performance Curve', color=constants.COLOR_PERFORMANCE_CURVE)
    plt.ylabel('Number of Customers')
    plt.xlabel('Time')
    plt.title('Outage and Restore Processes for EAGLE-i Event Number: ' + str(event_number) + ' with ' + str(len(outage_process)-2) +' outages (' + event_method.upper() + ')')
    plt.legend()
    plt.axhline(y=0, color='black', linewidth=0.5)  # show a horizontal line at 0

    # Format x-axis for better readability
    xtick_formatter = mdates.DateFormatter('%m-%d-%y\n%H:%M')
    plt.gca().xaxis.set_major_formatter(xtick_formatter)
    # Get the first and last x-values
    first_x_tick = min([row[0] for row in performance_process])
    last_x_tick = max([row[0] for row in performance_process])
    # Calculate eight intermediate tick positions (e.g., the midpoint)
    time_diff_secs = (last_x_tick - first_x_tick).total_seconds()
    intermediate_x_ticks = [first_x_tick + pd.Timedelta(seconds=time_diff_secs * i / 8) for i in range(1, 8)]
    # Set the x-ticks to only these three positions
    plt.gca().set_xticks([first_x_tick, *intermediate_x_ticks, last_x_tick])
    plt.show()


def plot_eaglei_log_performance(outage_df: pd.DataFrame, 
                                event_number: int, 
                                event_method: str = 'ac', 
                                timestamp_column: str = constants.TIMESTAMP_COL, 
                                customer_column: str = constants.CUSTOMERS_COL) -> None:
    """
    Function to plot the outage and restore processes for a given event number
    
    Parameters:
        outage_df: the outage data frame
        event_number: the event number to plot
        event_method: the method used to extract the events (default is 'ac')
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)
    
    Returns:
        None
    """
    outages, _, performance_process = get_eaglei_processes(outage_df, event_number, event_method, timestamp_column, customer_column)
    
    # Remove the first time step of the performance process
    performance_process = performance_process[1:]  # Remove the first time step (0, 0)
    performance_process.insert(0, (performance_process[0][0], 0))  

    # create a step plot of the outages
    plt.figure(figsize=(18,7))
    plt.step([row[0] for row in performance_process], [row[1] for row in performance_process], where='post', label='Performance Curve', color=constants.COLOR_PERFORMANCE_CURVE, linewidth=1.5, zorder=3)
    plt.ylabel('Number of Customers (log scale)')
    plt.xlabel('Time')
    plt.title('Outage and Restore Processes for EAGLE-i Event Number: ' + str(event_number) + ' with ' + str(len(outages)) +' outages (' + event_method.upper() + ')')
    plt.legend()
    plt.yscale('log')
    plt.axhline(y=1, color='black', linewidth=0.5, zorder=0, alpha=0.5, linestyle='--')  # show a horizontal line at 1
    plt.axhline(y=2, color='black', linewidth=0.5, zorder=0, alpha=0.5, linestyle='--')  # show a horizontal line at 2
    plt.axhline(y=3, color='black', linewidth=0.5, zorder=0, alpha=0.5, linestyle='--')  # show a horizontal line at 3
    # Format x-axis for better readability
    xtick_locator = mdates.AutoDateLocator()  # Automatically adjust ticks
    xtick_formatter = mdates.DateFormatter('%m-%d-%y\n%H:%M')
    plt.gca().xaxis.set_major_locator(xtick_locator)
    plt.gca().xaxis.set_major_formatter(xtick_formatter)
    # Show x-axis ticks every hour
    plt.gca().xaxis.set_tick_params(left=True, labelleft=True)
    plt.gca().yaxis.set_major_formatter(custom_label_formatter)  # Use custom formatter for y-axis
    
    plt.show()


def plot_multiple_eaglei_performance_curves(outage_df: pd.DataFrame, 
                                            event_numbers: List[int], 
                                            event_method: str = 'ac', 
                                            timestamp_column: str = constants.TIMESTAMP_COL, 
                                            customer_column: str = constants.CUSTOMERS_COL) -> None:
    """
    Function to plot the outage and restore processes for multiple event numbers in subplots of 5x10
    
    Parameters:
        outage_df: the outage data frame
        event_numbers: the list of event numbers to plot
        event_method: the method used to extract the events (default is 'ac')
        timestamp_column: the name of the timestamp column (default is constants.TIMESTAMP_COL)
        customer_column: the name of the customer column (default is constants.CUSTOMERS_COL)
    
    Returns:
        None
    """
    num_events = len(event_numbers)
    num_cols = 5
    num_rows = (num_events + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows))
    axs = axs.flatten()

    for idx, event_number in enumerate(event_numbers):
        outages, restores, performance_process = get_eaglei_processes(outage_df, event_number, event_method, timestamp_column, customer_column)

        if len(performance_process) == 0:
            continue

        outage_process = [(outages[i][0], v) for i, v in enumerate(np.cumsum([o[1] for o in outages]))]
        restore_process = [(restores[i][0], v) for i, v in enumerate(np.cumsum([r[1] for r in restores]))]

        # Add 0 for the first time step to the outage and restore processes
        outage_process.insert(0, (outage_process[0][0], 0))  
        restore_process.insert(0, (outage_process[0][0], 0))

        # Add the last time step to the outage process
        outage_process.append((restore_process[-1][0], restore_process[-1][1]))  

        # Remove the first time step of the performance process
        performance_process = performance_process[1:]  # Remove the first time step (0, 0)
        performance_process.insert(0, (performance_process[0][0], 0))  

        ax = axs[idx]
        ax.step([row[0] for row in outage_process], [row[1] for row in outage_process], where='post', label='Outage Curve', color=constants.COLOR_OUTAGE_CURVE)
        ax.step([row[0] for row in restore_process], [row[1] for row in restore_process], where='post', label='Restore Curve', color=constants.COLOR_RESTORE_CURVE)
        ax.step([row[0] for row in performance_process], [-row[1] for row in performance_process], where='post', label='Performance Curve', color=constants.COLOR_PERFORMANCE_CURVE)
        ax.set_ylabel('Number of Customers')
        ax.set_xlabel('Time')
        ax.set_title(f'Event Number: {event_number} ({len(outage_process)-2} outages)', fontsize=12)
        # ax.legend()
        ax.axhline(y=0, color='black', linewidth=0.5)  # show a horizontal line at 0

        # Format x-axis for better readability
        xtick_formatter = mdates.DateFormatter('%m-%d-%y\n%H:%M')
        ax.xaxis.set_major_formatter(xtick_formatter)
        # Get the first and last x-values
        first_x_tick = min([row[0] for row in performance_process])
        last_x_tick = max([row[0] for row in performance_process])
        # Calculate two intermediate tick positions between first and last
        time_diff_secs = (last_x_tick - first_x_tick).total_seconds()
        intermediate_x_ticks = [first_x_tick + pd.Timedelta(seconds=time_diff_secs * i / 3) for i in range(1, 3)]
        # Set the x-ticks to only these three positions
        ax.set_xticks([first_x_tick, *intermediate_x_ticks, last_x_tick])

    # Remove any unused subplots
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
    # return fig


# ------------------------- Identifying Issues in County Data -------------------------


def _detect_missing_data_gaps(df: pd.DataFrame, 
                              timestamp_col: str = constants.TIMESTAMP_COL, 
                              freq: str = "15min") -> pd.DataFrame:
    """
    Detects missing data gaps in the timestamp column of the dataframe.
    Returns a dataframe with start and end timestamps of the missing gaps and their duration.
    
    Parameters:
    - df: The input dataframe containing the timestamp column.
    - timestamp_col: The name of the timestamp column in the dataframe.
    - freq: The frequency of the EAGLE-I timestamps (default is "15min").
    
    Returns:
    - pd.DataFrame: A dataframe with columns 'start', 'end', and 'duration' indicating the missing data gaps.
    """
    
    # create a new dataframe using the timestamp column values excluding the last value
    missing_df = df.iloc[0:-1][timestamp_col].to_frame(name='start').copy()
    # add another column which is the next timestamp value
    missing_df['end'] = df.iloc[1:][timestamp_col].values
    # calculate the difference between the two timestamp columns
    missing_df['duration'] = missing_df['end'] - missing_df['start']
    # filter the dataframe to only include rows where the difference is greater than the typical frequency
    missing_df = missing_df[missing_df['duration'] > pd.Timedelta(freq)].reset_index(drop=True)
    return missing_df


def _detect_flatline_periods(df_sorted: pd.DataFrame, 
                             timestamp_col: str = constants.TIMESTAMP_COL, 
                             value_col: str = constants.CUSTOMERS_COL, 
                             min_value: int = 1, 
                             var_window: str = "12h") -> pd.DataFrame:
    """
    Detects flatline anomalies where the series is stuck above min_value.
    
    Parameters:
    - df_sorted: The input dataframe sorted by timestamp.
    - timestamp_col: The name of the timestamp column in the dataframe.
    - value_col: The name of the value column to check for flatlines.
    - min_value: The minimum value threshold to consider for flatlines.
    - var_window: The rolling window size for variance calculation (default is "12h").
    
    Returns:
    - pd.DataFrame: A dataframe containing detected flatline anomalies with start time, end time, duration, and flat value.
    """
    # Rolling variance
    df_sorted["rolling_var"] = df_sorted.rolling(window=var_window, min_periods=1, on=timestamp_col)[value_col].var()
    # To avoid NaNs at the start of the data and at the start of each year, we can back-fill them
    df_sorted["rolling_var"] = df_sorted["rolling_var"].bfill()

    # Identify low-variance segments above threshold value
    flat_segments = (df_sorted["rolling_var"] < 1e-6) & (df_sorted[value_col] > min_value)

    # Group consecutive runs
    flat_groups = (flat_segments != flat_segments.shift()).cumsum()
    flat_durations = df_sorted.groupby(flat_groups).apply(lambda g: (flat_segments[g.index].all(), g.index))

    anomalies = []
    for is_flat, idx in flat_durations:
        if is_flat:
            start_time = df_sorted.loc[idx[0], timestamp_col]
            end_time = df_sorted.loc[idx[-1], timestamp_col]
            flat_value = df_sorted.loc[idx[0], value_col]
            if (end_time - start_time) > pd.Timedelta("1hr"):   # sanity check to avoid very short periods
                # trace back the time in the timestamp column to find the actual start time
                # by looking for the first index where the value is different from the flat value
                for i in range(idx[0]-1, -1, -1):
                    if df_sorted.loc[i, value_col] != flat_value:
                        start_time = df_sorted.loc[i+1, timestamp_col]
                        break
                end_time = df_sorted.loc[idx[-1], timestamp_col]
                duration = end_time - start_time
                # if duration >= pd.Timedelta(duration_thresh):
                anomalies.append({"start_time": start_time, "end_time": end_time,
                                    "duration": duration, "flat_value": flat_value})
    
    # create a new dataframe to show the anomalies
    anomaly_df = pd.DataFrame(anomalies)
    anomaly_df = anomaly_df[['start_time', 'end_time', 'duration', 'flat_value']]
    anomaly_df = anomaly_df.sort_values(by='duration', ascending=False).reset_index(drop=True)
    # print(f"Detected {len(anomaly_df)} flatline anomalies.")
    return anomaly_df


def _max_consecutive_true_duration(mask: Any, 
                                   index: Any) -> pd.Timedelta:
    """
    Return max duration (Timedelta) of consecutive True segments in mask (same length as index).
    
    Parameters:
    - mask: Boolean array indicating True/False values.
    - index: DatetimeIndex corresponding to the mask.
    
    Returns:
    - pd.Timedelta: Maximum duration of consecutive True segments.
    """
    max_dur = pd.Timedelta(0)
    start = None
    for m, ts in zip(mask, index):
        if m:
            if start is None:
                start = ts
            end = ts
        else:
            if start is not None:
                dur = end - start
                if dur > max_dur:
                    max_dur = dur
                start = None
    # tail
    if start is not None:
        dur = end - start
        if dur > max_dur:
            max_dur = dur
    return max_dur


def _detect_stuck_periods(df_sorted: pd.DataFrame,
                          value_col: str = "customers_out",
                          timestamp_col: str = "run_start_time",
                          min_value: int = 1,
                          window_width: str = "24h",
                          duration_thresh: str = "14D",
                          floor_frac_thresh: float = 0.001,     # fraction of points near floor that indicates clipping
                          run_length_thresh: str = "1h",     # long consecutive time at floor
                          flatline_df=None) -> pd.DataFrame:
    """
    Detect candidate clipped (left-censored) periods.
    Returns list of (start, end, diagnostics_dict).
    Diagnostics include floor_val, floor_fraction, max_run_time, next_val, gap.
    
    Parameters
    ----------
    df_sorted : pd.DataFrame
        Input dataframe sorted by timestamp.
    value_col : str
        Column name for the values to analyze.
    timestamp_col : str
        Column name for the timestamps.
    min_value : int
        Minimum value threshold to consider for stuck detection.
    window_width : str
        Rolling window width for minimum calculation (e.g., "24h").
    duration_thresh : str
        Minimum duration threshold for stuck periods (e.g., "14D").
    floor_frac_thresh : float
        Fraction of points at floor to consider as clipping.
    run_length_thresh : str
        Minimum consecutive run length at floor to consider as clipping (e.g., "1h").
    flatline_df : pd.DataFrame, optional
        DataFrame of flatline periods to exclude from analysis.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing detected stuck anomalies with diagnostics.
    """
    df = df_sorted.copy()
    # exclude the flatline periods from the data
    if flatline_df is not None and not flatline_df.empty:
        for _, row in flatline_df.iterrows():
            df = df[~df[timestamp_col].between(row['start_time'], row['end_time'])]
        df = df.reset_index(drop=True)

    df.set_index(timestamp_col, inplace=True)
    # ensure datetime index sorted
    s = df[value_col].copy()
    s = s.sort_index()
    rolling_min = s.rolling(window_width).min()

    # candidate windows where rolling_min > min_value
    clipped_candidate = rolling_min > min_value

    # identify runs of True in clipped_candidate
    groups = (clipped_candidate != clipped_candidate.shift(fill_value=False)).cumsum()
    anomalies = []

    # convert string thresholds to Timedelta
    duration_thresh_td = pd.Timedelta(duration_thresh)
    run_length_thresh_td = pd.Timedelta(run_length_thresh)

    for g, group_df in s.groupby(groups):
        # only consider runs where clipped_candidate is True
        if not clipped_candidate.loc[group_df.index[0]]:
            continue

        start = group_df.index[0]
        end = group_df.index[-1]
        duration = end - start
        if duration < duration_thresh_td:
            continue

        sub = group_df  # pd.Series

        floor_val = float(sub.min())

        floor_mask = sub == floor_val
        floor_fraction = floor_mask.sum() / len(sub)

        # next distinct value above floor (if any)
        larger_values = sub[~floor_mask]
        if len(larger_values) > 0:
            next_val = larger_values.min()
            gap = next_val - floor_val
        else:
            next_val = None
            gap = np.inf

        gap_ok = (gap >= 2)

        # max consecutive time at floor
        max_run_time = _max_consecutive_true_duration(floor_mask.values, sub.index)

        # decide clipped vs high baseline:
        is_clipped = (
            (floor_fraction >= floor_frac_thresh and gap_ok)  # many hits at floor + gap
            or (max_run_time >= run_length_thresh_td)  # long consecutive stuck runs
        )

        diag = dict(
            start_time=start, 
            end_time=end, 
            duration=duration,
            stuck_value=floor_val,
            floor_fraction=floor_fraction,
            next_val=next_val, gap=gap,
            max_run_time=max_run_time
        )

        if is_clipped:
            anomalies.append(diag)

    if len(anomalies) == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "duration", "stuck_value", 
                                     "floor_fraction", "next_val", 
                                     "gap", "max_run_time"])
    else:
        # convert to DataFrame
        anomalies = pd.DataFrame(anomalies)
        # sort by duration descending
        anomalies = anomalies.sort_values(by="duration", ascending=False).reset_index(drop=True)
        return anomalies


def detect_eaglei_data_issues(df: pd.DataFrame,
                              value_col: str = "customers_out",
                              timestamp_col: str = "run_start_time",
                              baseline: int = 1,
                              freq: str = "15min",
                              min_stuck_duration: str = "14D",
                              min_gap_duration: str = "3D",
                              min_flatline_duration: str = "3D") -> Dict:
    """
    Analyze the dataset and suggest reasonable values for 
    min_gap_duration and min_stuck_duration for the detect_eaglei_data_issues function.

    Parameters
    ----------
    df : pd.DataFrame
        Input eaglei data containing timestamps and values.
    value_col : str
        Column containing outage/customer values.
    timestamp_col : str
        Column containing timestamps.
    baseline : float
        Baseline for "stuck" detection.
    freq : str
        Expected reporting frequency (default 15min).
    min_stuck_duration : str
        Minimum duration for stuck detection (default 14D).
    min_gap_duration : str
        Minimum duration for missing data gap detection (default 3D).
    min_flatline_duration : str
        Minimum duration for flatline detection (default 3D).
    
    Returns
    -------
    Dict
        Dictionary containing detected missing periods, stuck periods, and flatline periods.
    """

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)


    # --- 1. Detect missing data gaps ---
    missing_df = _detect_missing_data_gaps(df, timestamp_col, freq)

    # if not missing_df.empty:
    #     gap_threshold = missing_df['duration'].quantile(gap_quantile)
    #     gap_threshold = pd.Timedelta(int(gap_threshold.total_seconds()), unit='s')
    # else:
    #     gap_threshold = pd.Timedelta(freq)

    significant_missing_gaps = missing_df[missing_df['duration'] >= pd.Timedelta(min_gap_duration)]

    # --- 2. Detect Flatline periods (constant runs above baseline) ---
    all_flatline_periods_df = _detect_flatline_periods(df, timestamp_col, value_col, 
                                                       min_value = baseline,
                                                       var_window = "12h")
    
    significant_flatline_periods = all_flatline_periods_df[all_flatline_periods_df['duration'] >= pd.Timedelta(min_flatline_duration)]

    # --- 3. Detect Stuck durations (constant runs above baseline) ---
    all_stuck_periods_df = _detect_stuck_periods(df, timestamp_col=timestamp_col, 
                                                 value_col=value_col, 
                                                 min_value=baseline,
                                                 flatline_df=significant_flatline_periods,
                                                 duration_thresh=min_stuck_duration)
    
    # check if any of the detected stuck periods overlap with any of the missing periods
    # if so, modify the stuck period to exclude the missing periods
    if not significant_missing_gaps.empty and not all_stuck_periods_df.empty:
        adjusted_stuck_periods = []
        for _, stuck_row in all_stuck_periods_df.iterrows():
            stuck_start = stuck_row['start_time']
            stuck_end = stuck_row['end_time']
            overlapping_missing = significant_missing_gaps[(significant_missing_gaps['start'] < stuck_end) & (significant_missing_gaps['end'] > stuck_start)]
            if not overlapping_missing.empty:
                # there are overlapping missing periods, adjust the stuck period
                current_start = stuck_start
                for _, miss_row in overlapping_missing.iterrows():
                    miss_start = miss_row['start']
                    miss_end = miss_row['end']
                    if miss_start > current_start:
                        adjusted_stuck_periods.append({
                            "start_time": current_start,
                            "end_time": miss_start,
                            "duration": miss_start - current_start,
                            "stuck_value": stuck_row['stuck_value']
                        })
                    current_start = max(current_start, miss_end)
                if current_start < stuck_end:
                    adjusted_stuck_periods.append({
                        "start_time": current_start,
                        "end_time": stuck_end,
                        "duration": stuck_end - current_start,
                        "stuck_value": stuck_row['stuck_value']
                    })
            else:
                # no overlap, keep the original stuck period (only the start_time, end_time, duration, and stuck_value columns)
                adjusted_stuck_periods.append(stuck_row[['start_time', 'end_time', 'duration', 'stuck_value']].to_dict())
        all_stuck_periods_df = pd.DataFrame(adjusted_stuck_periods)
        all_stuck_periods_df = all_stuck_periods_df[['start_time', 'end_time', 'duration', 'stuck_value']]
        all_stuck_periods_df = all_stuck_periods_df.sort_values(by='duration', ascending=False).reset_index(drop=True)
    else:
        all_stuck_periods_df = all_stuck_periods_df[['start_time', 'end_time', 'duration', 'stuck_value']]

    return {
        "missing_periods": significant_missing_gaps,
        "stuck_periods": all_stuck_periods_df,
        "flatline_periods": significant_flatline_periods
    }


def plot_eaglei_timeline(df: pd.DataFrame, 
                         timestamp_col: str = "run_start_time", 
                         value_col: str = "customers_out", 
                         perform_zero_fill: bool = True, 
                         date_range: Tuple | None = None, 
                         log_y_scale: bool = True,
                         overlay_issues: bool = False, 
                         show_moving_average: bool = False,
                        #  ma_window_length: int = int(60*24*60/15),
                         ma_window_length: str = "60D",
                         show_median: bool = False,
                         plot_type: str = "step") -> None:
    """
    Function to plot EAGLEi time series data with options for zero-filling, date range filtering,
    log scale, moving average, median, and overlaying detected data issues.
    
    Parameters:
        df: The input EAGLEi dataframe.
        timestamp_col: The name of the timestamp column.
        value_col: The name of the value column.
        perform_zero_fill: Whether to perform zero-filling for missing timestamps.
        date_range: A tuple of (start_date, end_date) to filter the data.
        log_y_scale: Whether to use logarithmic scale for the y-axis.
        overlay_issues: Whether to overlay detected data issues on the plot.
        show_moving_average: Whether to show moving average on the plot.
        ma_window_length: The window length for moving average (e.g., "60D" for 60 days).
        show_median: Whether to show median line on the plot.
        plot_type: Type of plot - "line", "scatter", or "step".
    
    Returns:
        None
    """
    
    if plot_type not in ["line", "scatter", "step"]:
        raise ValueError("plot_type must be one of 'line', 'scatter', or 'step'")

    # check if timestamp_col and value_col are in the dataframe
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_col} not found in the dataframe.")
    
    if value_col not in df.columns:
        raise ValueError(f"Value column {value_col} not found in the dataframe.")
    
    # check if timestamp_col is in datetime format
    if not is_datetime64_any_dtype(df[timestamp_col]):
        raise ValueError(f"Timestamp column {timestamp_col} is not in datetime format.")
    
    counties = df['county'].unique()
    
    if date_range is not None:
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            raise ValueError("date_range must be a tuple of (start_date, end_date).")
        start_date, end_date = date_range
        df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
        if df.empty:
            raise ValueError("No data available in the specified date range.")
        # print(f"Data filtered to date range {start_date} to {end_date}. Data points: {df.shape[0]}")
    
    
    df_copy = df.copy()

    if perform_zero_fill:
        # reindex the dataframe by timestamp_col with a frequency of 15 minutes and fill missing values with 0
        df_copy = df_copy.set_index(timestamp_col).asfreq('15min').fillna(0).reset_index()

    _, ax = plt.subplots(figsize=(12, 5))

    if plot_type == "line":
        ax.plot(df_copy[timestamp_col], df_copy[value_col], label="Customers", color='darkorange', linewidth=1.0, zorder=1)
    elif plot_type == "scatter":
        ax.scatter(df_copy[timestamp_col], df_copy[value_col], label="Customers", color='darkorange', 
                   zorder=1, s=5, alpha=1.0, linewidth=1.0, marker='+')
    elif plot_type == "step":
        # Step plot of values
        ax.step(df_copy[timestamp_col], df_copy[value_col], where="post", label="Customers", color='darkorange', linewidth=1.0, zorder=1)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Customers Interrupted")
    
    counties = df['county'].unique()
    if len(counties) == 1:
        ax.set_title(f"EAGLEi Time Series - {counties[0]} County", y=1.09)
    elif len(counties) <= 5:
        ax.set_title("EAGLEi Time Series - Counties: " + ", ".join(counties), y=1.09)
    else:
        ax.set_title(f"EAGLEi Time Series - {len(counties)} Counties", y=1.09)

    # Overlay detected issues if requested
    if overlay_issues:
        issues = detect_eaglei_data_issues(df, 
                                           value_col=value_col, 
                                           timestamp_col=timestamp_col, 
                                           baseline=1, freq="15min")
        
        if not issues['missing_periods'].empty:
            min_dur = issues['missing_periods']['duration'].min().total_seconds()/3600/24
            max_dur = issues['missing_periods']['duration'].max().total_seconds()/3600/24
            gap_threshold_days_label = f"{len(issues['missing_periods'])} Data Gaps\n({min_dur:.1f} - {max_dur:.1f} days)"

            for _, row in issues['missing_periods'].iterrows():
                ax.axvspan(xmin=row['start'], xmax=row['end'], ymax=0.03, color='blue', alpha=0.9, zorder=2,
                           label=gap_threshold_days_label if gap_threshold_days_label not in ax.get_legend_handles_labels()[1] else "")
        
        if not issues['flatline_periods'].empty:
            min_dur = issues['flatline_periods']['duration'].min().total_seconds()/3600/24
            max_dur = issues['flatline_periods']['duration'].max().total_seconds()/3600/24
            flatline_periods_label = f"{len(issues['flatline_periods'])} Flatline Periods\n({min_dur:.1f} - {max_dur:.1f} days)"

            for _, row in issues['flatline_periods'].iterrows():
                ax.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymax=0.03, color='violet', alpha=0.9, zorder=3,
                           label=flatline_periods_label if flatline_periods_label not in ax.get_legend_handles_labels()[1] else "")
        
        if not issues['stuck_periods'].empty:
            min_dur = issues['stuck_periods']['duration'].min().total_seconds()/3600/24
            max_dur = issues['stuck_periods']['duration'].max().total_seconds()/3600/24
            stuck_threshold_days_label = f"{len(issues['stuck_periods'])} Stuck Periods\n({min_dur:.1f} - {max_dur:.1f} days)"
            # stuck_threshold_days_label = f"{len(issues['stuck_periods'])} Stuck Periods\n({issues['stuck_threshold'].total_seconds()/3600/24:.1f} days)"
            
            for _, row in issues['stuck_periods'].iterrows():
                ax.axvspan(xmin=row['start_time'], xmax=row['end_time'], ymax=0.03, color='limegreen', alpha=0.9, zorder=3,
                           label=stuck_threshold_days_label if stuck_threshold_days_label not in ax.get_legend_handles_labels()[1] else "")
        ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.15))

    # Overlay moving average if requested
    ma_window_len = pd.Timedelta(ma_window_length)
    if show_moving_average:
        df_copy = df_copy.sort_values(timestamp_col).reset_index(drop=True)
        if not perform_zero_fill:
            df_copy = df_copy.set_index(timestamp_col).asfreq('15min').fillna(0).reset_index()
        # Rolling average
        df_copy["rolling_avg"] = df_copy.rolling(window=ma_window_len, min_periods=1, on=timestamp_col)[value_col].mean()
        # convert the moving average to integer
        df_copy["rolling_avg"] = df_copy["rolling_avg"].astype(int)
        # convert the moving average values of 0 to 1 to avoid log(0) issues
        df_copy["rolling_avg"] = df_copy["rolling_avg"].replace(0, 1)
        ax.plot(df_copy[timestamp_col], df_copy["rolling_avg"],
                color="black", linewidth=1.0, alpha=1.0, zorder=4,
                label=f"Moving Average\n({(ma_window_len.total_seconds() / 3600 / 24):.0f} days window)")
        if show_median:
            df_copy["rolling_median"] = df_copy.rolling(window=ma_window_len, min_periods=1, on=timestamp_col)[value_col].median()
            df_copy["rolling_median"] = df_copy["rolling_median"].astype(int)
            df_copy["rolling_median"] = df_copy["rolling_median"].replace(0, 1)
            ax.plot(df_copy[timestamp_col], df_copy["rolling_median"],
                    color="purple", linewidth=1.0, alpha=1.0, zorder=5,
                    label=f"_Moving Median\n({(ma_window_len.total_seconds() / 3600 / 24):.0f} days window)")
        if overlay_issues:
            ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.15))
        else:
            ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15))

    # Log scale if requested
    if log_y_scale:
        ax.set_yscale('log')
        # ax.yaxis.set_major_formatter(custom_label_formatter)
        # set a customer y-axis formatter which shows the actual number instead of scientific notation
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:,.0f}'.format(y) if y >= 1 else '{:,.1f}'.format(y)))
        ax.set_ylabel("Customers Interrupted (log scale)")
    plt.tight_layout()
    plt.show()
    # return df_sorted


def create_stuck_periods_chart(eaglei_state_df: pd.DataFrame, 
                               logx_axis: bool = False, 
                               logy_axis: bool = False,
                               filter_min_duration_days: int = 14):
    """
    Create a scatter plot of all stuck periods detected in the EAGLE-i data for each county.
    
    Parameters:
    -----------
    eaglei_state_df : pd.DataFrame
        DataFrame containing EAGLE-i data for a specific state.
    logx_axis : bool
        Whether to use logarithmic scale for the x-axis.
    logy_axis : bool
        Whether to use logarithmic scale for the y-axis.
    filter_min_duration_days : int
        Minimum duration (in days) to filter stuck periods for plotting.

    Returns:
    --------
    None
    """
    all_stuck_periods = {}
    for c in eaglei_state_df['county'].unique():
        stuck_periods = detect_eaglei_data_issues(eaglei_state_df[eaglei_state_df['county'] == c])['stuck_periods']
        if not stuck_periods.empty:
            all_stuck_periods[c] = stuck_periods
    # create a dataframe from the dictionary
    all_stuck_periods_df = pd.concat(all_stuck_periods).reset_index(level=1, drop=True).reset_index()
    all_stuck_periods_df.columns = ['county', 'start', 'end', 'duration', 'stuck_value']
    all_stuck_periods_df['duration_days'] = all_stuck_periods_df['duration'].dt.total_seconds() / 3600 / 24
    all_stuck_periods_df = all_stuck_periods_df.sort_values(by='duration_days', ascending=False)
    all_stuck_periods_df['stuck_value'] = all_stuck_periods_df['stuck_value'].astype(int)
    # filter for stuck periods longer than 14 days
    all_stuck_periods_df = all_stuck_periods_df[all_stuck_periods_df['duration_days'] >= filter_min_duration_days]  # filter for stuck periods longer than 14 days
    if all_stuck_periods_df.empty:
        print("No stuck periods detected.")
        return

    state_name = eaglei_state_df[constants.STATE_COL].iloc[0] if constants.STATE_COL in eaglei_state_df.columns else 'Unknown'
    sns.displot(x='stuck_value', y='duration_days', 
                data=all_stuck_periods_df, 
                log_scale=(logx_axis, logy_axis), 
                height=6, aspect=1)  #16/9
    plt.title(f'All Stuck Periods in each county (EAGLE-i outages) - {state_name}\n(Only showing stuck periods longer than {filter_min_duration_days} days)')
    if logx_axis:
        plt.xlabel('Stuck Value (Customers Out) - log scale')
        plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
    else:
        plt.xlabel('Stuck Value (Customers Out)')
    
    if logy_axis:
        plt.ylabel('Duration (Days) - log scale')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y):,}'))
    else:
        plt.ylabel('Duration (Days)')
    
    plt.tight_layout()
    plt.show()


# ------------------------- County Adjacency Graphs -------------------------


def load_counties_shapefile(shapefile_path: str = os.path.join(os.getcwd(), constants.MISC_DIR, 'geojson-counties-fips.json'),
                            shapefile_url: str = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') -> dict | None:
    """
    Loads county shapefile data from a GeoJSON URL or local file.
    
    Parameters:
    -----------
    shapefile_url : str
        URL to the GeoJSON file containing county boundaries
    shapefile_path : str
        Local path to the GeoJSON file containing county boundaries

    Returns:
    --------
    counties_shape_data : dict
        GeoJSON FeatureCollection containing county boundaries
    """
    # First try to load from local file
    if os.path.exists(shapefile_path):
        try:
            with open(shapefile_path, 'r') as f:
                counties_shape_data = json.load(f)
            return counties_shape_data
        except Exception as e:
            print(f"Error loading county shapefile from local file: {e}")
    # If local file not found or failed, try to load from URL
    try:
        with urlopen(shapefile_url) as response:
            counties_shape_data = json.load(response)
    except Exception as e:
        print(f"Error loading county shapefile: {e}")
        return None
    
    return counties_shape_data


def create_county_adjacency_graph(state_fips_prefix: str | None = None) -> nx.Graph:
    """
    Creates a NetworkX graph where nodes are counties in a state and edges represent 
    neighboring counties with weights corresponding to boundary overlap length.
    
    Parameters:
    -----------
    counties_shape_data : dict
        GeoJSON FeatureCollection containing county boundaries
        
    Returns:
    --------
    G : networkx.Graph
        Graph where nodes are county names and edge weights are boundary overlap lengths
    """

    if state_fips_prefix is None:
        raise ValueError("State FIPS code must be provided to filter counties.")

    # Load county shapefile data
    counties_shape_data = load_counties_shapefile()
    if counties_shape_data is None:
        raise RuntimeError("Failed to load county shapefile data.")
    
    # Filter for counties in the state
    filtered_counties = {
        "type": "FeatureCollection",
        "features": [
            f for f in counties_shape_data['features'] if f['properties']['STATE'] == state_fips_prefix
        ]
    }
        
    # Convert GeoJSON to GeoDataFrame for easier spatial operations
    gdf = gpd.GeoDataFrame.from_features(filtered_counties['features'])
    gdf = gdf.set_crs('EPSG:4326')  # Set coordinate reference system
    
    # Project to a suitable CRS for USA Mainland (NAD83 / Conus Albers)
    # This ensures accurate distance/area calculations
    gdf = gdf.to_crs('EPSG:5070') 
    
    # Create the graph
    G = nx.Graph()
    
    # Add all counties as nodes
    for _, row in gdf.iterrows():
        county_name = row['NAME']
        G.add_node(county_name, 
                   fips_code=row['GEO_ID'],
                   geometry=row['geometry'],
                   census_area=row['CENSUSAREA'])
    
    # Check all pairs of counties for adjacency and calculate overlap
    for i, county1 in gdf.iterrows():
        for j, county2 in gdf.iterrows():
            if i >= j:  # Avoid duplicate pairs and self-comparison
                continue
                
            geom1 = county1['geometry']
            geom2 = county2['geometry']
            
            # Check if counties are adjacent (share a boundary)
            if geom1.touches(geom2):
                # Calculate the length of shared boundary
                intersection = geom1.intersection(geom2)
                
                # The intersection of two adjacent polygons should be a line (or lines)
                if hasattr(intersection, 'length'):
                    overlap_length = intersection.length
                else:
                    # Handle case where intersection might be a collection of geometries
                    try:
                        overlap_length = sum(geom.length for geom in intersection.geoms 
                                           if hasattr(geom, 'length'))
                    except:
                        overlap_length = 0
                
                # Add edge with overlap length as weight (convert to kilometers)
                if overlap_length > 0:
                    G.add_edge(county1['NAME'], 
                             county2['NAME'], 
                             weight=overlap_length / 1000,  # Convert to kilometers
                             overlap_length_km=overlap_length / 1000,
                             custom_added=False)
    
    return G


def create_multi_state_county_adjacency_graph(county_fips: List[str]) -> nx.Graph:
    """
    Creates a NetworkX graph where nodes are counties in multiple states and edges represent 
    neighboring counties with weights corresponding to boundary overlap length.
    
    Parameters:
    -----------
    county_fips : List[str]
        List of county FIPS codes to filter counties
        
    Returns:
    --------
    G : networkx.Graph
        Graph where nodes are county names and edge weights are boundary overlap lengths
    """

    if not county_fips:
        raise ValueError("At least one county FIPS code must be provided to filter counties.")
    
    # Load county shapefile data
    counties_shape_data = load_counties_shapefile()
    if counties_shape_data is None:
        raise RuntimeError("Failed to load county shapefile data.")
    
    # Filter for counties in the specified counties
    filtered_counties = {
        "type": "FeatureCollection",
        "features": [
            f for f in counties_shape_data['features'] if (f['properties']['STATE']+f['properties']['COUNTY']) in county_fips
        ]
    }
        
    # Convert GeoJSON to GeoDataFrame for easier spatial operations
    gdf = gpd.GeoDataFrame.from_features(filtered_counties['features'])
    gdf = gdf.set_crs('EPSG:4326')  # Set coordinate reference system
    # Add another column 'NAME_TO_USE' which is state name + 'NAME' to avoid duplicate county names across states
    # Since 'STATE' is a FIPS code, we can map it to state abbreviation using contants.STATE_FIPS_DICT, but we need to reverse the dictionary first
    reversed_state_fips_dict = {v: k.title() for k, v in constants.STATE_FIPS_DICT.items()}
    gdf['STATE_NAME'] = gdf['STATE'].map(reversed_state_fips_dict)
    gdf['NAME_TO_USE'] = gdf['STATE_NAME'] + gdf['NAME']

    # Project to a suitable CRS for USA Mainland (NAD83 / Conus Albers)
    # This ensures accurate distance/area calculations
    gdf = gdf.to_crs('EPSG:5070') 
    
    # Create the graph
    G = nx.Graph()
    
    # Add all counties as nodes
    for _, row in gdf.iterrows():
        county_name = row['NAME_TO_USE']
        G.add_node(county_name, 
                   fips_code=row['COUNTY'],
                   geometry=row['geometry'],
                   census_area=row['CENSUSAREA'])
    
    # Check all pairs of counties for adjacency and calculate overlap
    for i, county1 in gdf.iterrows():
        for j, county2 in gdf.iterrows():
            if i >= j:  # Avoid duplicate pairs and self-comparison
                continue
                
            geom1 = county1['geometry']
            geom2 = county2['geometry']
            
            # Check if counties are adjacent (share a boundary)
            if geom1.touches(geom2):
                # Calculate the length of shared boundary
                intersection = geom1.intersection(geom2)
                # The intersection of two adjacent polygons should be a line (or lines)
                if hasattr(intersection, 'length'):
                    overlap_length = intersection.length
                else:
                    # Handle case where intersection might be a collection of geometries
                    try:
                        overlap_length = sum(geom.length for geom in intersection.geoms 
                                           if hasattr(geom, 'length'))
                    except:
                        overlap_length = 0

                # Add an edge with the overlap length as weight
                G.add_edge(county1['NAME_TO_USE'], county2['NAME_TO_USE'], weight=overlap_length, overlap_length_km=overlap_length / 1000, custom_added=False)
    
    return G


def analyze_county_graph(G: nx.Graph) -> None:
    """
    Analyze and display basic statistics about the county adjacency graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        County adjacency graph

    Returns:
    --------
    None
    """
    print(f"County Adjacency Graph Statistics:")
    print(f"Number of counties (nodes): {G.number_of_nodes()}")
    print(f"Number of adjacencies (edges): {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Find counties with most/least neighbors
    degrees = dict(G.degree())
    max_degree_county = max(degrees, key=lambda k: degrees[k])
    min_degree_county = min(degrees, key=lambda k: degrees[k])

    print(f"\nCounty with most neighbors: {max_degree_county} ({degrees[max_degree_county]} neighbors)")
    print(f"County with least neighbors: {min_degree_county} ({degrees[min_degree_county]} neighbors)")
    
    # Find longest shared boundary
    edge_weights = nx.get_edge_attributes(G, 'weight')
    if edge_weights:
        max_weight_edge = max(edge_weights, key=edge_weights.get)
        print(f"\nLongest shared boundary: {max_weight_edge[0]} - {max_weight_edge[1]}")
        print(f"Boundary length: {edge_weights[max_weight_edge]:.2f} km")
    
    return None


def visualize_county_graph(G: nx.Graph, pos: Any = None, figsize: Tuple = (9, 6), save: bool = False, save_path: str = None) -> None:
    """
    Visualize the county adjacency graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        County adjacency graph
    pos : dict, optional
        Node positions for visualization
    figsize : tuple
        Figure size for the plot
    
    Returns:
    --------
    None
    """
    
    plt.figure(figsize=figsize)
    
    # Use circular layout if no positions provided
    if pos is None:
        # pos = nx.spring_layout(G, k=3, iterations=50)
        pos = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.9)
    
    # Draw edges with thickness proportional to weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [w / max_weight * 5 for w in weights]  # Scale to max width of 5
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.9, edge_color='black')

    # Draw edges for special cases (zero weight edges) in red dashed lines
    zero_weight_edges = [(u, v) for u, v in edges if G[u][v]['weight'] == 0]
    if zero_weight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=zero_weight_edges, width=0.5, alpha=0.9, edge_color='red', style='dashed')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title("County Adjacency Graph\n(Edge thickness = Boundary overlap length)")
    plt.axis('off')
    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def shortest_path_between_counties(G, county_a, county_b):
    """
    Find the shortest path between two counties in the adjacency graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        County adjacency graph
    county_a : str
        Starting county name
    county_b : str
        Ending county name
    
    Returns:
    --------
    None
    """
    # Find the shortest path between two counties
    try:
        path = nx.shortest_path(G, county_a, county_b)
        print(f"Shortest path from {county_a} to {county_b}: {' -> '.join(path)}")
        
        # Calculate path length based on boundary overlaps
        path_weight = nx.shortest_path_length(G, county_a, county_b, weight='weight')
        print(f"Total boundary overlap distance: {path_weight:.2f} km")
    except nx.NetworkXNoPath:
        print("No path found between the counties")
    except nx.NodeNotFound as e:
        print(f"County not found: {e}")


# ------------------------- Spatiotemporal Events Extraction -------------------------

def find_time_overlapping_groups(df_with_all_counties: pd.DataFrame, 
                                event_col: str ='event_number_ac_threshold_30', 
                                new_event_col: str ='event_number_temporal',
                                timestamp_col: str = constants.TIMESTAMP_COL,
                                county_col: str = constants.COUNTY_COL) -> pd.DataFrame:
    """
    Identify and label overlapping events across multiple counties based only on time.
    
    Parameters:
    -----------
    df_with_all_counties : pd.DataFrame
        DataFrame containing event data for multiple counties of a single state.
    event_col : str
        Column name of the column containing county-level events' event numbers for each county.
    new_event_col : str
        Column name for the new temporal event numbers to be created.
    timestamp_col : str
        Column name of the timestamp column.
    county_col : str
        Column name of the county column.
    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional column for temporal event numbers.
    """

    # create new columns using each county's event_number_ac_threshold_30 values
    df_pivot = df_with_all_counties.pivot_table(index=timestamp_col, columns=county_col, values=event_col, fill_value=0)
    # modify the frequency of the index to 15 minutes and fill missing timestamps with 0
    df_pivot = df_pivot.asfreq('15min').fillna(0)
    # change all column types to integers
    df_pivot = df_pivot.astype(int)
    # replace all the non-zero values with 1
    df_pivot[df_pivot > 0] = 1
    # create a new column with the sum of each row
    # this column basically tells you the total number of overlaping events in all the counties at each timestamp
    df_pivot[new_event_col] = df_pivot.sum(axis=1)
    # replace values greater than 0 with 1
    # df_pivot[new_event_col] = df_pivot[new_event_col].apply(lambda x: 1 if x > 0 else 0)
    

    temp = df_pivot[new_event_col].values
    event_numbers = []
    i = 0
    j = 1
    while (i < len(temp)):
        if temp[i] >= 1:
            start = i
            while (i < len(temp) and temp[i] >= 1):
                i+=1
            event_numbers.extend([j] * (i - start))
            j+=1
        else:
            event_numbers.append(0)
            i+=1

    # append the event number of the df
    df_pivot[new_event_col] = event_numbers 
    # return df_pivot

    # remove the 0 values
    df_pivot = df_pivot[df_pivot[new_event_col]>0][new_event_col]

    df_with_all_counties_copy = df_with_all_counties.copy()

    # Map the temporal event ids from df_pivot to the original dataframe using run_start_time
    df_with_all_counties_copy[new_event_col] = df_with_all_counties_copy[timestamp_col].map(df_pivot)

    return df_with_all_counties_copy


def get_neighbors_at_level(graph: nx.Graph, county: str, level: int) -> set:
    """
    Get all neighbors of a county up to a specified level.
    
    Parameters:
    -----------
    graph : networkx.Graph
        County adjacency graph
    county : str
        County name
    level : int
        Maximum neighbor level (1 = immediate neighbors, 2 = neighbors + their neighbors, etc.)
        
    Returns:
    --------
    set
        Set of county names within the specified neighbor level
    """
    if county not in graph:
        return {county}
    
    neighbors = {county}  # Include the county itself
    current_level = {county}
    
    for _ in range(level):
        next_level = set()
        for node in current_level:
            next_level.update(graph.neighbors(node))
        current_level = next_level - neighbors # Only new neighbors for next iteration
        neighbors.update(next_level)
        if not current_level:  # No more neighbors to explore
            break
    
    return neighbors


def _create_event_map(events_df, 
                      event_id, 
                      event_col, 
                      counties_geojson = None, 
                      county_wide_events_col: str = 'event_number_ac_threshold_30', 
                      plotting_axis=None, 
                      state_fips_code=None):
    """
    Plot a specific event on a map showing which counties are involved and record counts.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Events dataframe (either time-only or spatio-temporal)
    event_id : int
        Event ID to plot
    event_col : str
        Type of event: 'time_only' or 'spatiotemporal'
    counties_geojson : dict, optional
        GeoJSON FeatureCollection containing county boundaries
    county_wide_events_col : str
        Column name for county-wide event numbers
    plotting_axis : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a ValueError is raised.
    state_fips_code : str, optional
        FIPS code of the state to filter counties. If None, it will be inferred from events_df.
    """
    

    if plotting_axis is None:
        raise ValueError("plotting_axis must be provided, or use the plot_event_map function which creates its own figure and axis.")

    if state_fips_code is None:
        # check if state column exists in events_df
        if constants.STATE_COL in events_df.columns:
            unique_states = events_df[constants.STATE_COL].unique()
            if len(unique_states) == 1:
                state_name = unique_states[0].lower()
                state_fips_code = constants.STATE_FIPS_DICT.get(state_name, None)
                if state_fips_code is None:
                    raise ValueError(f"State name '{state_name}' not found in state_fips_dict.")
            else:
                raise ValueError("Multiple states found in events_df. Please provide a specific state_fips_code.")
        else:
            raise ValueError(f"state_fips_code must be provided if '{constants.STATE_COL}' column is not in events_df.")
    else:
        state_name = [k for k, v in constants.STATE_FIPS_DICT.items() if v == state_fips_code][0].title()
    
    
    # Filter data for the specific event
    event_data = events_df[events_df[event_col] == event_id]

    # check if county names are prefixed with state name and remove the prefix if exists
    if event_data[constants.COUNTY_COL].str.startswith(state_name).any():
        event_data[constants.COUNTY_COL] = event_data[constants.COUNTY_COL].str.replace(f"{state_name}", "", regex=False)
    
    if len(event_data) == 0:
        print(f"No data found for event {event_id}")
        return
    
    def count_eaglei_outages(series):
        """Custom aggregation function to calculate the number of EAGLE-i outages"""
        return np.sum(np.diff(series.unique())>0) + 1

    # Count records per county for this event
    county_counts = event_data.groupby(constants.COUNTY_COL).agg({
        constants.CUSTOMERS_COL: ['count', count_eaglei_outages, 'sum'],
        constants.TIMESTAMP_COL: ['min', 'max'],
        county_wide_events_col: 'nunique'
    }).reset_index()

    county_counts.columns = ['county', 'record_count', 'eaglei_outages_count', 'total_customers_out', 'start_time', 'end_time', 'num_county_wide_events']

    if counties_geojson is None:
        counties_geojson = load_counties_shapefile()
    features = []
    for feature in counties_geojson['features']:
        if feature['properties']['STATE'] == state_fips_code:
            features.append(feature)
    
    gdf = gpd.GeoDataFrame.from_features(features)
    gdf = gdf.set_crs('EPSG:4326')

    
    # Merge with event data
    gdf_with_counts = gdf.merge(
        county_counts,
        left_on='NAME',
        right_on='county',
        how='left'
    )
    
    # Fill NaN values with 0 for counties not in this event
    gdf_with_counts['record_count'] = gdf_with_counts['record_count'].fillna(0)
    
    # Plot all counties with boundaries
    gdf_with_counts.boundary.plot(ax=plotting_axis, linewidth=0.8, color='black')
    
    # Create color map for event counties
    event_counties = gdf_with_counts[gdf_with_counts['record_count'] > 0]
    
    if len(event_counties) > 0:
        # Plot event counties with color intensity based on record count
        t_ax=event_counties.plot(
            column='total_customers_out',
            ax=plotting_axis,
            cmap='Reds',
            legend=True,
            legend_kwds={'label': 'Total Customers Interrupted (in all county-wide events)', 'shrink': 0.6},
            vmin=event_counties['total_customers_out'].min(),
            vmax=event_counties['total_customers_out'].max()
        )

        # Add county labels for event counties
        for _, row in event_counties.iterrows():
            centroid = row['geometry'].centroid
            plotting_axis.annotate(
                f"{row['NAME']}\n({int(row['num_county_wide_events'])})",
                (centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        t_fig = t_ax.figure
        cb_ax = t_fig.axes[1] 
        cb_ax.tick_params(labelsize=12)     # Increase colorbar tick label size
        cb_ax.yaxis.label.set_size(12)      # Increase colorbar label size
    else:
        plotting_axis.text(0.5, 0.5, 'No counties involved in this event', 
            verticalalignment='center', horizontalalignment='center',
            transform=plotting_axis.transAxes,
            color='red', fontsize=14, alpha=0.7, fontweight='bold')
    
    # Add a text in one corner stating that the map is for illustrative purposes only
    plotting_axis.text(0, 0, 'Label boxes indicate number of county-wide events in that county', 
            verticalalignment='bottom', horizontalalignment='left',
            transform=plotting_axis.transAxes,
            color='black', fontsize=8, alpha=0.7, fontstyle='italic')
    
    # Set title and styling
    duration = (county_counts['end_time'].max() - county_counts['start_time'].min())
    plotting_axis.set_title(
        f'Event # {event_id}, {len(event_counties)} Counties, {county_counts["record_count"].sum():.0f} Total Records'
        f', {county_counts["eaglei_outages_count"].sum():.0f} Total EAGLEi outages\n'
        f'Start Time: {county_counts['start_time'].min()}, End Time: {county_counts['end_time'].max()}, Duration: {duration}',
        fontsize=12,
        pad=10
    )

    # Restrict the map to state extent
    plotting_axis.set_xlim(gdf.total_bounds[0] - 0.05, gdf.total_bounds[2] + 0.05)
    plotting_axis.set_ylim(gdf.total_bounds[1] - 0.05, gdf.total_bounds[3] + 0.05)
    
    # Remove axis labels and grid lines for cleaner look
    plotting_axis.set_xlabel('')
    plotting_axis.set_ylabel('')
    plotting_axis.grid(False)

    # Remove axis ticks for cleaner look
    plotting_axis.set_xticks([])
    plotting_axis.set_yticks([])

    # Remove axis spines for cleaner look
    for spine in plotting_axis.spines.values():
        spine.set_visible(False)


def plot_event_on_map(events_df, 
                      event_id, 
                      event_col, 
                      counties_geojson = None, 
                      county_wide_events_col='event_number_ac_threshold_30', 
                      state_fips_code=None):
    """
    Wrapper function to plot a specific event on a map.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Events dataframe (either time-only or spatio-temporal)
    event_id : int
        Event ID to plot
    event_col : str
        Column name for event IDs in events_df
    counties_geojson : dict
        GeoJSON FeatureCollection containing Massachusetts county boundaries
    county_wide_events_col : str
        Column name for county-wide events in events_df
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated map figure
    """
    figsize = (12, 10)
    fig, ax = plt.subplots(figsize=figsize)
    _create_event_map(events_df, event_id, event_col, counties_geojson, county_wide_events_col, plotting_axis=ax, state_fips_code=state_fips_code)
    plt.tight_layout()
    plt.show(fig)


def _create_event_timeline(events_df, 
                           event_id, 
                           event_col_to_identify, 
                           event_col='event_number_ac_threshold_30', 
                           max_counties=14, 
                           plotting_axis=None):
    """
    Plot a timeline showing event start and end times by county within a specified time range.
    
    Parameters:
    -----------
    time_events_df : pandas.DataFrame
        DataFrame containing event data with columns: county, run_start_time, event_number
    start_time : str or datetime
        Start time for the timeline (e.g., '2018-03-01' or '2018-03-01 12:00:00')
    end_time : str or datetime  
        End time for the timeline (e.g., '2018-03-10' or '2018-03-10 12:00:00')
    event_col : str, default='event_number_ac_threshold_30'
        Column name containing event numbers
    figsize : tuple, default=(15, 10)
        Figure size (width, height)
    max_counties : int, optional
        Maximum number of counties to display (displays counties with most events if limited)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The timeline plot figure
    """
    if plotting_axis is None:
        raise ValueError("plotting_axis must be provided, or use the plot_event_timeline function which creates its own figure and axis.")

    if max_counties > 14:
        print(f"Warning: Limiting the number of counties to {max_counties} for better visualization.")
        max_counties = 14

    df_filtered = events_df[events_df[event_col_to_identify] == event_id].copy()
    if len(df_filtered) == 0:
        print(f"No events found with event ID {event_id}")
        return None
    
    # Ensure run_start_time is datetime
    # df_filtered[constants.TIMESTAMP_COL] = pd.to_datetime(df_filtered[constants.TIMESTAMP_COL])
    
    start_time = df_filtered[constants.TIMESTAMP_COL].min()
    end_time = df_filtered[constants.TIMESTAMP_COL].max()
    
    # Calculate event start and end times for each county-event combination
    event_ranges = []
    
    for county in df_filtered[constants.COUNTY_COL].unique():
        county_data = df_filtered[df_filtered[constants.COUNTY_COL] == county]
        
        for event_num in county_data[event_col].unique():
            if pd.isna(event_num):
                continue
                
            event_data = county_data[county_data[event_col] == event_num]
            event_start = event_data[constants.TIMESTAMP_COL].min()
            event_end = event_data[constants.TIMESTAMP_COL].max()
            
            # If event has only one timestamp, add 15 minutes for visualization
            if event_start == event_end:
                event_end = event_start + pd.Timedelta(minutes=15)
            
            max_customers = event_data[constants.CUSTOMERS_COL].max()
            
            event_ranges.append({
                'county': county,
                'event_num': int(event_num),
                'start_time': event_start,
                'end_time': event_end,
                'duration_hours': (event_end - event_start).total_seconds() / 3600,
                'max_customers_out': max_customers
            })
    
    event_ranges_df = pd.DataFrame(event_ranges)
    
    if len(event_ranges_df) == 0:
        print("No complete events found in the time range")
        return None
    total_counties_in_event = event_ranges_df['county'].nunique()
    if total_counties_in_event > max_counties:
        # print(f"Warning: There are {total_counties_in_event} counties involved in event {event_id}. ")
        
        # Selecting top counties to plot based on the maximum number of events in each county
        # county_event_counts = event_ranges_df['county'].value_counts()
        # top_counties = county_event_counts.head(max_counties).index.tolist()
        # event_ranges_df = event_ranges_df[event_ranges_df['county'].isin(top_counties)]

        # Selecting top counties to plot based on the maximum customers affected in each county
        county_max_customers = event_ranges_df.groupby('county')['max_customers_out'].max().sort_values(ascending=False)
        top_counties = county_max_customers.head(max_counties).index.tolist()
        event_ranges_df = event_ranges_df[event_ranges_df['county'].isin(top_counties)]
        
        # print(f"Displaying top {len(top_counties)} counties with most events")
    
    # Sort counties by first event start time for better visualization
    county_first_event = event_ranges_df.groupby('county')['start_time'].min().sort_values()
    counties_ordered = county_first_event.index.tolist()
    
    # Calculate y-positions for events, ensuring no overlap within counties
    y_positions = {}
    event_y_positions = {}
    county_row_height = 0.20  # Space allocated per county
    event_height = 0.1  # Height of each event bar
    
    for i, county in enumerate(counties_ordered):
        base_y = i * county_row_height
        county_events = event_ranges_df[event_ranges_df['county'] == county].copy()
        county_events = county_events.sort_values('start_time')
        
        # For each county, stack events vertically if they would overlap
        # Since events within a county don't overlap in time, we can plot them all at the same level
        # But we'll add a small vertical offset to make multiple events more visible
        
        for _, (_, event_row) in enumerate(county_events.iterrows()):
            event_key = (county, event_row['event_num'])
            # Use the base position for the county since events don't overlap in time
            event_y_positions[event_key] = base_y
        
        y_positions[county] = base_y
    
    
    # Color map for different event intensities
    max_customers_global = event_ranges_df['max_customers_out'].max()
    
    # Plot each event as a horizontal bar
    for _, row in event_ranges_df.iterrows():
        county = row['county']
        event_key = (county, row['event_num'])
        y_pos = event_y_positions[event_key]
        
        
        # Plot the event duration as a horizontal bar
        plotting_axis.barh(y_pos, 
                (row['end_time'] - row['start_time']).total_seconds() / (3600*24),  # Duration in days
                left=mdates.date2num(row['start_time']),
                height=event_height,
                color = 'lightgray',
                alpha=0.4,
                edgecolor='black',
                linewidth=0.5,
                zorder=1)
        
        # Plot the individual timestamps as small vertical lines with color intensity equal to the customer's affected in each timestamp
        county_event_data = df_filtered[(df_filtered[constants.COUNTY_COL] == county) & (df_filtered[event_col] == row['event_num'])]
        for ts, cust in zip(county_event_data[constants.TIMESTAMP_COL], county_event_data[constants.CUSTOMERS_COL]):
            plotting_axis.plot([mdates.date2num(ts), mdates.date2num(ts)], 
                    [y_pos - event_height/2.0, y_pos + event_height/2.0], 
                    color=plt.colormaps['Reds'](0.3 + 0.8 * (cust / max_customers_global)),  # Color based on customer intensity
                    alpha=1.0, 
                    linewidth=1.5,
                    zorder=2)
        
        # Add event number as text
        mid_time = row['start_time'] + (row['end_time'] - row['start_time']) / 2
        plotting_axis.text(mdates.date2num(mid_time), y_pos+0.085, 
                f"{int(row['event_num'])}", 
                ha='center', va='center', 
                fontsize=8, fontweight='normal', color='gray', zorder=3)
    
    # Customize the plot
    plotting_axis.set_yticks([y_positions[county] for county in counties_ordered])
    plotting_axis.set_yticklabels(counties_ordered)
    plotting_axis.set_ylabel('Counties', fontsize=12, fontweight='bold')
    plotting_axis.set_xlabel('Time', fontsize=12, fontweight='bold')
    plotting_axis.set_title(f'Event # {event_id} Timeline by County\nStart Time: {start_time.strftime("%Y-%m-%d %H:%M")}, End Time: {end_time.strftime("%Y-%m-%d %H:%M")} '
                 f'Duration: {(end_time - start_time).total_seconds() / 86400:.2f} days, {len(event_ranges_df)} County-wide Events, {total_counties_in_event} Counties', 
                 fontsize=12, fontweight='bold', pad=20)
    
    # Format x-axis
    plotting_axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    total_duration_hours = (end_time - start_time).total_seconds() / 3600
    interval_value = max(1, total_duration_hours/30)        # maximum 30 values on the x-axis
    plotting_axis.xaxis.set_major_locator(mdates.HourLocator(interval= int(interval_value)))
    plt.setp(plotting_axis.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set x-axis limits
    plotting_axis.set_xlim(mdates.date2num(start_time - pd.Timedelta(minutes=10)), 
                mdates.date2num(end_time + pd.Timedelta(minutes=10)))

    # Set y-axis limits
    plotting_axis.set_ylim(-0.1, 2.7)

    # Add colorbar to show customer intensity scale
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps['Reds'], 
                               norm=plt.Normalize(vmin=0, vmax=max_customers_global))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plotting_axis, shrink=0.8, pad=0.02)
    cbar.set_label('Customers Interrupted', fontsize=12, fontweight='bold')

    # Add a text in one corner
    if total_counties_in_event > max_counties:
        plotting_axis.text(1.09, 1.1, f'(Displaying top {max_counties} counties\nwith most customers affected)', 
            verticalalignment='top', horizontalalignment='right',
            transform=plotting_axis.transAxes,
            color='black', fontsize=8, alpha=0.7, fontstyle='italic')


def plot_event_timeline(events_df, event_id, event_col_to_identify, event_col='event_number_ac_threshold_30', figsize=(15, 6), max_counties=14):
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    _create_event_timeline(events_df, event_id, event_col_to_identify, event_col, max_counties, ax)
    plt.tight_layout()
    plt.show(fig)


def segregate_by_space(events_df, 
                       temporal_event_number, 
                       temporal_event_col='event_number_temporal', 
                       county_event_col='event_number_ac_threshold_30', 
                       neighbour_level=1, 
                       graph_of_counties=None,
                       timestamp_column=constants.TIMESTAMP_COL,
                       customer_column=constants.CUSTOMERS_COL,
                       verbose=0,
                       time_overlap_method='outage_process_overlap') -> pd.DataFrame:

    spatiotemporal_event_col_name = 'event_number_spatiotemporal'

    # Filter events for the specified temporal event number
    event_data = events_df[events_df[temporal_event_col] == temporal_event_number].copy()

    # Get the time ranges for each county-event combination
    county_events_in_event = event_data.groupby([constants.COUNTY_COL,county_event_col]).agg({timestamp_column:['min','max']}).sort_values((timestamp_column,'min'))
    county_events_in_event = county_events_in_event.reset_index(drop=False, col_level=1)
    county_events_in_event = county_events_in_event.droplevel(0, axis=1)

    # Pre-compute neighbor sets for all counties in the event
    if graph_of_counties is None:
        raise ValueError("Adjanency Graph of Counties must be provided")
    else:
        county_neighbors = {}
        for county in events_df.county.unique():
            county_neighbors[county] = get_neighbors_at_level(graph_of_counties, county, level=neighbour_level)

    def _find_outage_process_end_time(county_event_id):
        '''
        Find the time when the outage process ends for a given county event.
        '''
        event_data_subset = event_data[event_data[county_event_col] == county_event_id]
        # check if all values in 'customers_out' are the same
        if event_data_subset[customer_column].nunique() == 1:
            return event_data_subset[timestamp_column].min()
        diffs = event_data_subset[customer_column].diff()
        diffs.iloc[0] = 1
        idx = diffs[diffs>0].index.values[-1]
        return event_data_subset.loc[idx,timestamp_column]
    
    county_events_in_event['outage_process_end_at'] = county_events_in_event[county_event_col].apply(_find_outage_process_end_time)

    # define a function to check the spatiotemporal conditions
    def _check_spatiotemporal_conditions(event_i, event_j):

        # Check if counties are neighbours
        if event_j[constants.COUNTY_COL] not in county_neighbors[event_i[constants.COUNTY_COL]]:
            return False

        # Check for temporal overlap
        if time_overlap_method == 'outage_process_overlap':
            if (event_i['min'] <= event_j['max']) and (event_j['min'] <= event_i['outage_process_end_at']):
                return True
        elif time_overlap_method == 'standard_overlap':
            if (event_i['min'] <= event_j['max']) and (event_i['max'] >= event_j['min']):
                return True
            
        return False

    # Create a graph and add all the events as nodes using their ids as per county_event_col
    event_graph = nx.Graph()
    event_graph.add_nodes_from(county_events_in_event[county_event_col].values)
    
    # Add edges between nodes if their counties are neighbours and their time ranges overlap
    for i, row_i in county_events_in_event.iterrows():
        for j, row_j in county_events_in_event.iterrows():
            if i >= j:
                continue

            if _check_spatiotemporal_conditions(row_i, row_j):
                event_graph.add_edge(row_i[county_event_col], row_j[county_event_col])
    
    # Find connected components in the graph
    connected_components = list(nx.connected_components(event_graph))
    if verbose > 0:
        print(f"Found {len(connected_components)} spatial groups within temporal event {temporal_event_number}")
    
    # Assign spatiotemporal event numbers based on connected components
    spatiotemporal_event_mapping = {}
    for spatiotemporal_event_number, component in enumerate(connected_components, start=1):
        for county_event_id in component:
            spatiotemporal_event_mapping[county_event_id] = spatiotemporal_event_number
            event_data.loc[event_data[county_event_col] == county_event_id, spatiotemporal_event_col_name] = spatiotemporal_event_number
    
    # change the column to integer type
    event_data[spatiotemporal_event_col_name] = event_data[spatiotemporal_event_col_name].astype(int)

    return event_data





def apply_spatiotemporal_grouping(combined, graph_of_counties, 
                                   temporal_event_col='event_number_multi_county',
                                   county_event_col='event_number_eaglei',
                                   neighbour_level=1,
                                   time_overlap_method='outage_process_overlap',
                                   verbose=1):
    """
    Apply spatiotemporal grouping to all temporal events in the dataset.
    
    Parameters:
    -----------
    combined : xarray.Dataset
        Your combined dataset
    graph_of_counties : networkx.Graph
        Adjacency graph of counties
    temporal_event_col : str
        Column name for temporal events (e.g., 'event_number_multi_county')
    county_event_col : str
        Column name for county events (e.g., 'event_number_eaglei')
    neighbour_level : int
        Neighbor level for spatial grouping
    time_overlap_method : str
        'outage_process_overlap' or 'standard_overlap'
    verbose : int
        Verbosity level
    """
    
    # Step 1: Convert xarray to DataFrame (excluding 0 and -1)
    print("Step 1: Converting xarray to DataFrame...")
    df = prepare_dataframe_for_spatiotemporal(combined, temporal_event_col, county_event_col)
    
    # Step 2: Apply segregate_by_space to all temporal events
    print(f"\nStep 2: Applying spatiotemporal grouping to {df[temporal_event_col].nunique()} temporal events...")
    
    # Get unique temporal events
    temporal_events = df[temporal_event_col].unique()
    temporal_events = temporal_events[temporal_events > 0]  # Exclude 0 if present
    
    # Store results
    all_results = []
    
    for idx, temporal_event_num in enumerate(temporal_events):
        if verbose > 0 and idx % 100 == 0:
            print(f"  Processing temporal event {idx+1}/{len(temporal_events)}")
        
        # Apply segregate_by_space to this temporal event
        result_df = segregate_by_space(
            events_df=df,
            temporal_event_number=temporal_event_num,
            temporal_event_col=temporal_event_col,
            county_event_col=county_event_col,
            neighbour_level=neighbour_level,
            graph_of_counties=graph_of_counties,
            timestamp_column='time',
            customer_column='customers_out',
            verbose=0,
            time_overlap_method=time_overlap_method
        )
        
        all_results.append(result_df)
    
    # Combine all results
    df_with_spatiotemporal = pd.concat(all_results, ignore_index=True)
    
    print(f"\nStep 3: Assigning globally unique spatiotemporal event IDs...")
    # Make spatiotemporal event numbers globally unique
    df_with_spatiotemporal = make_spatiotemporal_globally_unique(
        df_with_spatiotemporal, 
        temporal_event_col
    )
    
    # Step 4: Merge back into xarray
    print("Step 4: Merging results back into xarray dataset...")
    combined = _merge_results_back(
        combined, 
        df_with_spatiotemporal, 
        'event_number_mc_spatiotemporal'
    )
    
    return combined, df_with_spatiotemporal


def prepare_dataframe_for_spatiotemporal(combined, temporal_event_col, county_event_col):
    """
    Convert xarray to DataFrame with required columns, excluding 0 and -1.
    """
    # Extract data
    temporal_events = combined[temporal_event_col].values  # (county, time)
    county_events = combined[county_event_col].values  # (county, time)
    customers_out = combined['customers_out'].values  # (county, time)
    times = combined['time'].values
    counties = combined['county'].values
    
    # Build DataFrame
    data_rows = []
    for c, county in enumerate(counties):
        for t, time in enumerate(times):
            temporal_val = temporal_events[c, t]
            county_val = county_events[c, t]
            
            # Only include rows where both temporal and county events are > 0
            if temporal_val > 0 and county_val > 0:
                data_rows.append({
                    'time': time,
                    'county': county,
                    'customers_out': customers_out[c, t],
                    temporal_event_col: temporal_val,
                    county_event_col: county_val
                })
    
    df = pd.DataFrame(data_rows)
    print(f"  Created DataFrame with {len(df)} rows")
    print(f"  Unique temporal events: {df[temporal_event_col].nunique()}")
    
    return df


def make_spatiotemporal_globally_unique(df, temporal_event_col):
    """
    Make spatiotemporal event numbers globally unique across all temporal events.
    
    The segregate_by_space function assigns spatiotemporal numbers starting from 1
    for each temporal event. This function makes them globally unique.
    """
    next_global_id = 1
    spatiotemporal_mapping = {}
    
    # Group by temporal event
    for temporal_event_num in df[temporal_event_col].unique():
        mask = df[temporal_event_col] == temporal_event_num
        
        # Get unique spatiotemporal events within this temporal event
        local_spatiotemporal_events = df.loc[mask, 'event_number_spatiotemporal'].unique()
        
        for local_id in local_spatiotemporal_events:
            # Create unique key: (temporal_event, local_spatiotemporal_id)
            key = (temporal_event_num, local_id)
            
            if key not in spatiotemporal_mapping:
                spatiotemporal_mapping[key] = next_global_id
                next_global_id += 1
    
    # Apply global IDs
    df['event_number_mc_spatiotemporal'] = df.apply(
        lambda row: spatiotemporal_mapping[(row[temporal_event_col], row['event_number_spatiotemporal'])],
        axis=1
    )
    
    print(f"  Assigned {next_global_id - 1} globally unique spatiotemporal events")
    
    return df


def _merge_results_back(combined, df_result, new_event_column_name='event_number_mc_spatiotemporal'):
    """
    Merge the new event numbers from pandas DataFrame back into xarray dataset.
    """
    # Create a new array initialized with 0
    n_counties = len(combined['county'])
    n_times = len(combined['time'])
    
    new_event_array = np.zeros((n_counties, n_times), dtype=int)
    
    # Create lookup dictionaries for fast indexing
    county_to_idx = {county: idx for idx, county in enumerate(combined['county'].values)}
    time_to_idx = {pd.Timestamp(time): idx for idx, time in enumerate(combined['time'].values)}
    
    # Fill in the new event numbers
    for _, row in df_result.iterrows():
        county_idx = county_to_idx[row['county']]
        time_idx = time_to_idx[pd.Timestamp(row['time'])]
        new_event_array[county_idx, time_idx] = row[new_event_column_name]
    
    # Add to dataset
    combined[new_event_column_name] = (('county', 'time'), new_event_array)
    
    print(f"Added '{new_event_column_name}' to dataset")
    print(f"Unique events: {len(np.unique(new_event_array[new_event_array > 0]))}")
    
    return combined


def apply_spatiotemporal_grouping_optimized(combined, graph_of_counties,
                                             temporal_event_col='event_number_multi_county',
                                             county_event_col='event_number_eaglei',
                                             neighbour_level=1,
                                             time_overlap_method='outage_process_overlap',
                                             verbose=1):
    """
    Optimized version that processes all temporal events together.
    This is more efficient as it:
    1. Pre-computes neighbor relationships once
    2. Vectorizes time overlap calculations where possible
    3. Builds the entire graph in one pass
    """
    
    print("Converting xarray to DataFrame...")
    df = prepare_dataframe_for_spatiotemporal(combined, temporal_event_col, county_event_col)
    
    print("\nPre-computing county neighbors...")
    # Pre-compute all county neighbors
    county_neighbors = {}
    for county in df['county'].unique():
        county_neighbors[county] = get_neighbors_at_level(graph_of_counties, county, level=neighbour_level)
    
    print("Computing time ranges and outage process end times...")
    # Compute time ranges and outage process end times for all county events
    county_event_info = df.groupby(['county', county_event_col]).agg({
        'time': ['min', 'max'],
        'customers_out': list
    }).reset_index()
    
    county_event_info.columns = ['county', county_event_col, 'time_min', 'time_max', 'customers_list']
    
    # Compute outage process end time
    def compute_outage_process_end(row):
        customers = row['customers_list']
        times_for_event = df[(df['county'] == row['county']) & 
                             (df[county_event_col] == row[county_event_col])]['time'].values
        
        if len(set(customers)) == 1:
            return times_for_event[0]
        
        # Find last increase
        diffs = np.diff(customers)
        increase_indices = np.where(diffs > 0)[0]
        
        if len(increase_indices) > 0:
            last_increase_idx = increase_indices[-1] + 1
            return times_for_event[last_increase_idx]
        else:
            return times_for_event[0]
    
    county_event_info['outage_process_end'] = county_event_info.apply(compute_outage_process_end, axis=1)
    
    # Add temporal event info
    df_with_temporal = df.merge(
        county_event_info[['county', county_event_col, 'time_min', 'time_max', 'outage_process_end']],
        on=['county', county_event_col],
        how='left'
    )
    
    print("Building spatiotemporal graph...")
    # Build graph for all events
    event_graph = nx.Graph()
    event_graph.add_nodes_from(df[county_event_col].unique())
    
    # Process each temporal event
    temporal_events = df[temporal_event_col].unique()
    temporal_events = temporal_events[temporal_events > 0]
    
    spatiotemporal_assignments = {}
    next_global_id = 1
    
    for idx, temporal_event_num in enumerate(temporal_events):
        if verbose > 0 and idx % 100 == 0:
            print(f"  Processing temporal event {idx+1}/{len(temporal_events)}")
        
        # Get county events in this temporal event
        temporal_mask = df_with_temporal[temporal_event_col] == temporal_event_num
        events_in_temporal = df_with_temporal[temporal_mask]
        
        # Get unique county events
        unique_county_events = events_in_temporal[[
            'county', county_event_col, 'time_min', 'time_max', 'outage_process_end'
        ]].drop_duplicates()
        
        # Create subgraph for this temporal event
        subgraph = nx.Graph()
        subgraph.add_nodes_from(unique_county_events[county_event_col].values)
        
        # Add edges based on spatial and temporal conditions
        for i, row_i in unique_county_events.iterrows():
            for j, row_j in unique_county_events.iterrows():
                if i >= j:
                    continue
                
                # Check spatial condition
                if row_j['county'] not in county_neighbors[row_i['county']]:
                    continue
                
                # Check temporal condition
                if time_overlap_method == 'outage_process_overlap':
                    overlaps = (row_i['time_min'] <= row_j['time_max']) and \
                              (row_j['time_min'] <= row_i['outage_process_end'])
                else:  # standard_overlap
                    overlaps = (row_i['time_min'] <= row_j['time_max']) and \
                              (row_i['time_max'] >= row_j['time_min'])
                
                if overlaps:
                    subgraph.add_edge(row_i[county_event_col], row_j[county_event_col])
        
        # Find connected components
        connected_components = list(nx.connected_components(subgraph))
        
        # Assign global spatiotemporal IDs
        for component in connected_components:
            for county_event_id in component:
                spatiotemporal_assignments[county_event_id] = next_global_id
            next_global_id += 1
    
    # Apply assignments to DataFrame
    df['event_number_mc_spatiotemporal'] = df[county_event_col].map(spatiotemporal_assignments)
    
    print(f"\nAssigned {next_global_id - 1} globally unique spatiotemporal events")
    
    # Merge back into xarray
    print("Merging results back into xarray dataset...")
    combined = _merge_results_back(combined, df, 'event_number_mc_spatiotemporal')
    
    return combined, df


# def apply_spatiotemporal_grouping_optimized2(combined, graph_of_counties,
#                                                 temporal_event_col='event_number_multi_county',
#                                                 county_event_col='event_number_eaglei',
#                                                 neighbour_level=1,
#                                                 time_overlap_method='outage_process_overlap',
#                                                 verbose=1):
#     """
#     Optimized version - Another variant
#     """
    
#     print("Converting xarray to DataFrame...")
#     df = prepare_dataframe_for_spatiotemporal(combined, temporal_event_col, county_event_col)
    
#     # Sort by time to ensure temporal ordering
#     df = df.sort_values('time').reset_index(drop=True)
    
#     print("\nPre-computing county neighbors...")
#     county_neighbors = {}
#     for county in df['county'].unique():
#         county_neighbors[county] = get_neighbors_at_level(graph_of_counties, county, level=neighbour_level)
    
#     # Process each temporal event separately (matching original)
#     temporal_events = df[temporal_event_col].unique()
#     temporal_events = temporal_events[temporal_events > 0]
    
#     all_spatiotemporal_assignments = {}
#     next_global_id = 1
    
#     for idx, temporal_event_num in enumerate(temporal_events):
#         if verbose > 0 and idx % 100 == 0:
#             print(f"  Processing temporal event {idx+1}/{len(temporal_events)}")
        
#         # Filter for this temporal event
#         event_data = df[df[temporal_event_col] == temporal_event_num].copy()
        
#         # Compute time ranges exactly like original
#         county_events_in_event = event_data.groupby(['county', county_event_col]).agg({
#             'time': ['min', 'max']
#         }).sort_values(('time', 'min'))
        
#         county_events_in_event = county_events_in_event.reset_index(drop=False)
#         county_events_in_event.columns = ['county', county_event_col, 'time_min', 'time_max']
        
#         # Compute outage process end time for each county event
#         def compute_outage_process_end(row):
#             county_event_data = event_data[
#                 (event_data['county'] == row['county']) & 
#                 (event_data[county_event_col] == row[county_event_col])
#             ].sort_values('time')
            
#             customers = county_event_data['customers_out'].values
#             times = county_event_data['time'].values
            
#             # Match original logic exactly
#             if len(set(customers)) == 1:
#                 return times[0]
            
#             # Compute diffs
#             diffs = np.diff(customers)
#             diffs = np.insert(diffs, 0, 1)  # Add 1 at the beginning like original
            
#             # Find last positive diff
#             positive_indices = np.where(diffs > 0)[0]
#             if len(positive_indices) > 0:
#                 idx = positive_indices[-1]
#                 return times[idx]
#             else:
#                 return times[0]
        
#         county_events_in_event['outage_process_end'] = county_events_in_event.apply(
#             compute_outage_process_end, axis=1
#         )
        
#         # Build graph for this temporal event
#         subgraph = nx.Graph()
#         subgraph.add_nodes_from(county_events_in_event[county_event_col].values)
        
#         # Check pairs exactly like original
#         for i in range(len(county_events_in_event)):
#             for j in range(len(county_events_in_event)):
#                 if i >= j:
#                     continue
                
#                 row_i = county_events_in_event.iloc[i]
#                 row_j = county_events_in_event.iloc[j]
                
#                 # Skip same county (addressing the issue from earlier review)
#                 if row_i['county'] == row_j['county']:
#                     continue
                
#                 # Check spatial condition
#                 if row_j['county'] not in county_neighbors[row_i['county']]:
#                     continue
                
#                 # Check temporal condition - EXACTLY as original
#                 if time_overlap_method == 'outage_process_overlap':
#                     # Original: (event_i['min'] <= event_j['max']) and (event_j['min'] <= event_i['outage_process_end_at'])
#                     overlaps = (row_i['time_min'] <= row_j['time_max']) and \
#                               (row_j['time_min'] <= row_i['outage_process_end'])
#                 else:  # standard_overlap
#                     overlaps = (row_i['time_min'] <= row_j['time_max']) and \
#                               (row_i['time_max'] >= row_j['time_min'])
                
#                 if overlaps:
#                     subgraph.add_edge(row_i[county_event_col], row_j[county_event_col])
        
#         # Find connected components
#         connected_components = list(nx.connected_components(subgraph))
        
#         # Assign global IDs
#         for component in connected_components:
#             for county_event_id in component:
#                 all_spatiotemporal_assignments[county_event_id] = next_global_id
#             next_global_id += 1
    
#     # Apply assignments to DataFrame
#     df['event_number_mc_spatiotemporal'] = df[county_event_col].map(all_spatiotemporal_assignments)
    
#     print(f"\nAssigned {next_global_id - 1} globally unique spatiotemporal events")
    
#     # Merge back into xarray
#     print("Merging results back into xarray dataset...")
#     combined = _merge_results_back(combined, df, 'event_number_mc_spatiotemporal')
    
#     return combined, df






def _create_eaglei_multicounty_performance_curve(events_df: pd.DataFrame, 
                                                 event_number: int, 
                                                 event_method: str ='spatiotemporal', 
                                                 timestamp_column: str =constants.TIMESTAMP_COL, 
                                                 customer_column: str =constants.CUSTOMERS_COL) -> go.Figure:
    """
    Plot performance curve of an event that involves multiple counties.
    That include temporally overlapping events and spatio-temporal events.

    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame containing event data with columns: county, run_start_time, customers_out, event_number
    event_number : int
        Event number to plot
    event_method : str, default='spatiotemporal'
        Method used for event detection (for labeling purposes)
    timestamp_column : str, default=constants.TIMESTAMP_COL
        Column name for the timestamp
    customer_column : str, default=constants.CUSTOMERS_COL
        Column name for the customer count

    Returns:
    -------
    None
    """

    # Determine the correct event column based on the method
    event_col = 'event_number_' + event_method

    # Check if the event column exists in the DataFrame
    if event_col not in events_df.columns:
        raise ValueError(f"Event column '{event_col}' not found in DataFrame")
    
    # Check if the timestamp and customer columns exist
    if timestamp_column not in events_df.columns or customer_column not in events_df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' or customer column '{customer_column}' not found in DataFrame")

    # Filter data for the specified event number
    event_data = events_df[events_df[event_col] == event_number].copy()
    if event_data.empty:
        raise ValueError(f"No data found for event number {event_number}")
    
    # Check how many unique counties are involved in this event
    unique_counties = event_data[constants.COUNTY_COL].nunique()
    if unique_counties < 2:
        raise ValueError(f"Event number {event_number} involves only {unique_counties} county. Use plot_eaglei_event_curves() instead.")
    
    # Convert timestamp column to datetime if not already (check first)
    if not np.issubdtype(event_data[timestamp_column].dtype, np.datetime64):
        event_data[timestamp_column] = pd.to_datetime(event_data[timestamp_column])

    # Sort data by timestamp
    event_data = event_data.sort_values(by=timestamp_column)

    # Counties order
    counties_order = event_data.groupby(constants.COUNTY_COL)[customer_column].sum().sort_values(ascending=False).index.tolist()

    # Calculate data for performance curve
    performance_data = event_data.groupby(timestamp_column).agg({customer_column: 'sum'}).reset_index()
    # add a row at the end with timestamp + 15 minutes and 0 customers_out to extend the performance curve
    last_row = pd.DataFrame([{timestamp_column: performance_data[timestamp_column].max() + pd.Timedelta(minutes=15), customer_column: 0}])
    performance_data = pd.concat([performance_data, last_row], ignore_index=True)
    # add a row at the begining with timestamp and 0 customers_out to extend the performance curve
    first_row = pd.DataFrame([{timestamp_column: performance_data[timestamp_column].min() - pd.Timedelta(minutes=0), customer_column: 0}])
    performance_data = pd.concat([first_row, performance_data], ignore_index=True)

    # Calculate time range for x-axis limits
    event_start_time = performance_data[timestamp_column].min()
    event_end_time = performance_data[timestamp_column].max()
    time_offset_for_xaxis = ((event_end_time-event_start_time).total_seconds()/60)*.05

    # Create an empty figure
    fig = go.Figure()

    # Add performance curve as a line (step plot equivalent)
    fig.add_trace(go.Scatter(x=performance_data[timestamp_column], y=performance_data[customer_column],
                    mode='lines',
                    line_shape='hv',  # horizontal-vertical step lines
                    name='Performance Curve',
                    marker_color=constants.COLOR_PERFORMANCE_CURVE,
                    line=dict(width=1)
                    ))

    # Use a particular color palette
    color_palette = px.colors.qualitative.Prism  # 24 colors, similar to tab20

    # if there are more than 10 counties then only use first 10 counties and group the rest as 'Other'
    if len(counties_order) > 10:
        counties_to_plot = counties_order[:10]
        event_data[constants.COUNTY_COL] = event_data[constants.COUNTY_COL].apply(lambda x: x if x in counties_to_plot else 'Other counties')
        counties_order = counties_to_plot + ['Other counties']

    # Add a bar trace for each county
    for i, c in enumerate(counties_order):
        county_data = event_data[event_data[constants.COUNTY_COL] == c]
        fig.add_trace(go.Bar(
            x=county_data[timestamp_column],
            y=county_data[customer_column],
            name=c,
            offset=-0.5,     # left align the bars to start from the timestamp (like align='edge' in matplotlib)
            marker_color=color_palette[i % len(color_palette)],  # Cycle through color palette
            opacity=1.0,
        ))
    
    # Update layout for stacked bars with no gaps and no borders
    fig.update_layout(barmode='stack', bargap=0, xaxis={'categoryorder':'array', 'categoryarray':counties_order})
    fig.update_traces(marker_line_width=0)
    
    # Set figure size
    fig.update_layout(width=1200, height=600)
    
    # Set titles and labels
    fig.update_layout(
        title=dict(
            text=f"Performance Curve for Event {event_number} ({event_method})",
            x=0.5,
            pad=dict(t=0, b=0),
            xanchor='center',
            font=dict(size=18, family='Arial', color='black', weight='bold')
        ),
        xaxis_title=dict(
            text="Time",
            font=dict(size=16, family='Arial', color='black', weight='bold')
        ),
        yaxis_title=dict(
            text="Total Customers Affected",
            font=dict(size=16, family='Arial', color='black', weight='bold')
        ),
        # increase size of the tick labels
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=14)
        ),
    )

    # Update x and y axis properties
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            type='date',
            # tickformat='%d %b %y\n%H:%M',
            range=[event_start_time - pd.Timedelta(minutes=time_offset_for_xaxis),
                   event_end_time + pd.Timedelta(minutes=time_offset_for_xaxis)],
            # tickangle=-45
        ),
        yaxis=dict(
            type='linear',
            tickformat=',',
            gridcolor='lightgray',
            # range=[0, performance_data[customer_column].max() * 1.1]
        ),
        template='plotly_white',
        margin=dict(t=50, b=50, r=150)  # Extra margin for legend
    )

    # Return the figure
    return fig


def plot_eaglei_multicounty_performance_curve(events_df, event_number, event_method='spatiotemporal', timestamp_column=constants.TIMESTAMP_COL, customer_column=constants.CUSTOMERS_COL):
    # Create the figure
    fig = _create_eaglei_multicounty_performance_curve(events_df, event_number, event_method=event_method, timestamp_column=timestamp_column, customer_column=customer_column)
    # Display the figure
    fig.show()


def save_spatiotemporal_event_curves(events_df, 
                          temporal_event_id, 
                          temporal_event_col='event_number_temporal',
                          spatiotemporal_event_col='event_number_spatiotemporal', 
                          pdf_path=None):
    """
    Save performance curves of spatiotemporal events within a specified temporal event to a single-page PDF.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        DataFrame containing event data with spatiotemporal event numbers
    temporal_event_id : int
        Temporal event ID to plot
    temporal_event_col : str
        Column name for temporal event numbers
    spatiotemporal_event_col : str
        Column name for spatiotemporal event numbers
    pdf_path : str, optional
        Path to save the PDF file. If None, saves to 'RESULTS_DIR/curves_of_event_{temporal_event_id}.pdf'
    """
    # Filter data for the specified temporal event
    event_data = events_df[events_df[temporal_event_col] == temporal_event_id]
    
    # Get unique spatiotemporal event IDs within this temporal event
    spatiotemporal_event_ids = event_data[spatiotemporal_event_col].unique()
    
    num_spatiotemporal_events = len(spatiotemporal_event_ids)

    if num_spatiotemporal_events == 1:
        print("Only one spatiotemporal event found; no need to plot multiple subplots.")
        return
    
    if pdf_path == None:
        cwd = os.getcwd()
        pdf_path = os.path.join(cwd , constants.RESULTS_DIR, f'curves_of_event_{temporal_event_id}.pdf')

    with PdfPages(pdf_path) as pdf:

        # Create a plot with 2 rows and 1 column
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(20, 15))
        axes = axes.flatten()

        event_data_agg = event_data.groupby(constants.TIMESTAMP_COL).agg({constants.CUSTOMERS_COL:'sum', temporal_event_col:'min'}).reset_index().sort_values(constants.TIMESTAMP_COL, ascending=True)
        outages_full, restores_full, performance_process_full = get_eaglei_processes(event_data_agg, temporal_event_id, event_method='temporal')
        outage_process_full = [(outages_full[i][0], v) for i, v in enumerate(np.cumsum([o[1] for o in outages_full]))]
        restore_process_full = [(restores_full[i][0], v) for i, v in enumerate(np.cumsum([r[1] for r in restores_full]))]
        # Add 0 for the first time step to the outage and restore processes
        outage_process_full.insert(0, (outage_process_full[0][0], 0))  
        restore_process_full.insert(0, (outage_process_full[0][0], 0))
        # Add the last time step to the outage process
        outage_process_full.append((restore_process_full[-1][0], restore_process_full[-1][1]))  
        # Remove the first time step of the performance process
        performance_process_full = performance_process_full[1:]  # Remove the first time step (0, 0)
        performance_process_full.insert(0, (performance_process_full[0][0], 0))
        # create a step plot of the outages
        axes[0].step([row[0] for row in outage_process_full], [row[1] for row in outage_process_full], where='post', label='Outage Curve', color=constants.COLOR_OUTAGE_CURVE)
        axes[0].step([row[0] for row in restore_process_full], [row[1] for row in restore_process_full], where='post', label='Restore Curve', color=constants.COLOR_RESTORE_CURVE)
        axes[0].step([row[0] for row in performance_process_full], [-row[1] for row in performance_process_full], where='post', label='Performance Curve', color=constants.COLOR_PERFORMANCE_CURVE)
        axes[0].set_ylabel('Number of Customers')
        # axes[0].set_xlabel('Time')
        axes[0].set_title(f'Complete Temporal Event # {temporal_event_id}', fontsize=12)
        axes[0].axhline(y=0, color='black', linewidth=0.5)  # show a horizontal line at 0
        # Format x-axis for better readability
        xtick_formatter = mdates.DateFormatter('%m-%d-%y\n%H:%M')
        axes[0].xaxis.set_major_formatter(xtick_formatter)
        # Get the first and last x-values
        first_x_tick = min([row[0] for row in performance_process_full])
        last_x_tick = max([row[0] for row in performance_process_full])
        # Calculate three intermediate tick positions (e.g., the midpoint)
        time_diff_secs = (last_x_tick - first_x_tick).total_seconds()
        intermediate_x_ticks = [first_x_tick + pd.Timedelta(seconds=time_diff_secs * i / 15) for i in range(1, 15)]
        # Set the x-ticks to only these three positions
        axes[0].set_xticks([first_x_tick, *intermediate_x_ticks, last_x_tick])

        # Create single plot for the complete spatiotemporal event
        event_data_agg = event_data.groupby(constants.TIMESTAMP_COL).agg({constants.CUSTOMERS_COL:'sum', temporal_event_col:'min'}).reset_index().sort_values(constants.TIMESTAMP_COL, ascending=True)
        outages_full, restores_full, performance_process_full = get_eaglei_processes(event_data_agg, temporal_event_id, event_method='temporal')
        outage_process_full = [(outages_full[i][0], v) for i, v in enumerate(np.cumsum([o[1] for o in outages_full]))]
        restore_process_full = [(restores_full[i][0], v) for i, v in enumerate(np.cumsum([r[1] for r in restores_full]))]
        # Add 0 for the first time step to the outage and restore processes
        outage_process_full.insert(0, (outage_process_full[0][0], 0))  
        restore_process_full.insert(0, (outage_process_full[0][0], 0))
        # Add the last time step to the outage process
        outage_process_full.append((restore_process_full[-1][0], restore_process_full[-1][1]))  
        # Remove the first time step of the performance process
        performance_process_full = performance_process_full[1:]  # Remove the first time step (0, 0)
        performance_process_full.insert(0, (performance_process_full[0][0], 0))

        for spatiotemporal_event_id in spatiotemporal_event_ids:
            spatiotemporal_event_data = event_data[event_data[spatiotemporal_event_col] == spatiotemporal_event_id]
            spatiotemporal_event_data = spatiotemporal_event_data.groupby(constants.TIMESTAMP_COL).agg({constants.CUSTOMERS_COL:'sum', spatiotemporal_event_col:'min'}).reset_index().sort_values(constants.TIMESTAMP_COL, ascending=True)
            outages, restores, performance_process = get_eaglei_processes(spatiotemporal_event_data, spatiotemporal_event_id, event_method='spatiotemporal')

            outage_process = [(outages[i][0], v) for i, v in enumerate(np.cumsum([o[1] for o in outages]))]
            restore_process = [(restores[i][0], v) for i, v in enumerate(np.cumsum([r[1] for r in restores]))]
            
            # Add 0 for the first time step to the outage and restore processes
            outage_process.insert(0, (outage_process[0][0], 0))  
            restore_process.insert(0, (outage_process[0][0], 0))
            
            # Add the last time step to the outage process
            outage_process.append((restore_process[-1][0], restore_process[-1][1]))  
            
            # Remove the first time step of the performance process
            performance_process = performance_process[1:]  # Remove the first time step (0, 0)
            performance_process.insert(0, (performance_process[0][0], 0))  

            # create a step plot of the performance process
            axes[1].step([row[0] for row in performance_process], [row[1] for row in performance_process], where='post', label=f'Performance Curve {spatiotemporal_event_id}')

        axes[1].set_ylabel('Number of Customers (Log Scale)')
        axes[1].set_xlabel('Time')
        axes[1].set_title(f'Performance Curves of all Spatiotemporal Events of the Temporal Event # {temporal_event_id}', fontsize=12)

        axes[1].set_yscale('log')

        # Format x-axis for better readability
        xtick_formatter = mdates.DateFormatter('%m-%d-%y\n%H:%M')
        axes[1].xaxis.set_major_formatter(xtick_formatter)
        
        plt.tight_layout()
        # plt.show()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved plots to {pdf_path}")


def _ordinal(n: int) -> str:
    """
    Return the ordinal representation of an integer (e.g. 1 -> "1st", 2 -> "2nd", 11 -> "11th").

    Args:
        n: integer to convert

    Returns:
        str: ordinal string

    Raises:
        TypeError: if n is not an int

    Examples:
        >>> ordinal(1)
        '1st'
        >>> ordinal(2)
        '2nd'
        >>> ordinal(9)
        '9th'
        >>> ordinal(11)
        '11th'
        >>> ordinal(43)
        '43rd'
        >>> ordinal(-1)
        '-1st'
    """
    if not isinstance(n, int):
        raise TypeError("ordinal() expects an int")

    abs_n = abs(n)
    # Special-case 11-13 -> 'th'
    if 10 <= (abs_n % 100) <= 20:
        suffix = "th"
    else:
        last = abs_n % 10
        if last == 1:
            suffix = "st"
        elif last == 2:
            suffix = "nd"
        elif last == 3:
            suffix = "rd"
        else:
            suffix = "th"

    return f"{n}{suffix}"


def save_plots_to_pdf(events_df, 
                      event_id, 
                      graph_of_counties,
                      overlap_method='outage_process_overlap', 
                      county_event_col='event_number_ac_threshold_30',
                      pdf_path=None):
    """
    Save plots of specified events to a PDF file.
    
    Parameters:
    -----------
    events_df : pd.DataFrame
        Events dataframe (either time-only or spatio-temporal)
    event_id : int
        event ID to plot
    method : str
        Type of event: 'time_only' or 'spatiotemporal'
    customer_threshold : int
        Customer threshold used for event detection
    pdf_path : str
        Path to save the PDF file
    """

    if pdf_path == None:
        cwd = os.getcwd()
        pdf_path = os.path.join(cwd , constants.RESULTS_DIR, f'plots_of_event_{event_id}.pdf')

    spatiotemporal_events = segregate_by_space(events_df, 
                                               temporal_event_number=event_id, 
                                               time_overlap_method=overlap_method,
                                               county_event_col=county_event_col,
                                               graph_of_counties=graph_of_counties)
    new_events_ids = spatiotemporal_events.event_number_spatiotemporal.value_counts().index.values  # sorted by event size
    print(f"{len(new_events_ids)} spatiotemporal events created")

    with PdfPages(pdf_path) as pdf:
        # create a figure with two subplots so that the top subplot contains the map and the bottom subplot contains the time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1.4, 1]})

        _create_event_map(events_df, 
                          event_id, 
                          event_col='event_number_temporal', 
                          plotting_axis=ax1,
                          county_wide_events_col=county_event_col)
        # Add a text in the top left corner indicating which event this is
        ax1.text(0, 1, 'Original Event (based only on temporal overlaps)', 
                verticalalignment='top', horizontalalignment='left',
                transform=ax1.transAxes,
                color='red', fontsize=10, alpha=1.0, fontstyle='normal', fontweight='bold')
        
        _create_event_timeline(events_df, 
                               event_id, 
                               event_col_to_identify='event_number_temporal', 
                               event_col=county_event_col,
                               plotting_axis=ax2)
        plt.tight_layout()
        # plt.show(fig)       # display the figure of the original event
        pdf.savefig(fig)
        plt.close(fig)

        # save all the created spatiotemporal events
        for i, id in enumerate(new_events_ids):
            # create a figure with two subplots so that the top subplot contains the map and the bottom subplot contains the time series
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1.4, 1]})
            _create_event_map(events_df=spatiotemporal_events, 
                              event_id=id, 
                              county_wide_events_col=county_event_col,
                              event_col='event_number_spatiotemporal', 
                              plotting_axis=ax1)
            # Add a text in the top left corner indicating which event this is
            ax1.text(0, 1, f'Spatiotemporal Event: {_ordinal(i+1)} largest, out of {len(new_events_ids)} total', 
                    verticalalignment='top', horizontalalignment='left',
                    transform=ax1.transAxes,
                    color='red', fontsize=10, alpha=1.0, fontstyle='normal', fontweight='bold')
            _create_event_timeline(events_df = spatiotemporal_events, 
                                   event_id = id,
                                   event_col_to_identify = 'event_number_spatiotemporal', 
                                   event_col = county_event_col,
                                   plotting_axis = ax2)
            plt.tight_layout()
            # plt.show(fig)       # display the figure of the original event
            pdf.savefig(fig)
            plt.close(fig)

        # save the performance curve of the temporal event
        fig1 = _create_eaglei_multicounty_performance_curve(events_df, event_id, event_method='temporal')
        # fig1.show()
        temp_file_name = os.path.join(cwd , constants.RESULTS_DIR, "temp.png")
        fig1.write_image(file=temp_file_name, format="png", width=1200, height=600, scale=1.0)     # convert the plotly figure to a static image
        # fig1.write_html(file=temp_file_name.replace('.pdf', '.html'))   # save as html as well
        
        # read the image back using matplotlib to get its dimensions
        img = plt.imread(temp_file_name, format='png')
        # Get image dimensions
        img_dimensions = img.shape
        img_height = img_dimensions[0]
        img_width = img_dimensions[1]
        # Set desired DPI
        display_dpi = 100.0
        # Calculate figure size in inches
        fig_size = (img_width / display_dpi, img_height / display_dpi)
        # create a matplotlib figure to display the image
        fig = plt.figure(figsize=fig_size)
        plt.imshow(img, interpolation='nearest')    # inerpolation='nearest' to avoid blurriness
        plt.axis('off') # Turn off axes for a cleaner look
        pdf.savefig(fig, dpi=300)
        plt.close(fig) # Close the Matplotlib figure
        # Clean up the temporary image file
        os.remove(temp_file_name)
    
    print(f"Saved plots to {pdf_path}")

    # save_spatiotemporal_event_curves(spatiotemporal_events, temporal_event_id=event_id)


# ------------------------- Public API for EagleiStateProcessor -------------------------

# A class which can load data for a specific state
class EagleiStateProcessor:
    def __init__(self, 
                 state_name: str, 
                 verbose: int = 1):
        
        self.verbose = verbose

        # Load EAGLE-i data for the state
        self.eaglei_df, self.county_wide_customers = load_eaglei_state_data(state_name, verbose=verbose)

        # Load the adjacency graph for counties in the state
        self.state_fips_code = constants.STATE_FIPS_DICT.get(state_name.lower(), None)
        self.county_adjacency_graph = create_county_adjacency_graph(state_fips_prefix=self.state_fips_code)

        # Initializing other attributes
        self.state_name = state_name
        self.counties = self.eaglei_df['county'].unique().tolist()
        self.county_processors = {}
        self.all_counties_events_df = pd.DataFrame()
        self.ac_customers_threshold = 1

    def add_edge_to_county_adjacency_graph(self, from_county: str, to_county: str, edge_weight: float = 0.0, border_length: float = 0.0):
        '''
        Add an additional edge to the county adjacency graph.
        Each edge is a tuple of two county names (county1, county2).
        
        Parameters:
        -----------
        from_county : str
            Name of the first county
        to_county : str
            Name of the second county
        edge_weight : float
            Weight of the edge (default: 0.0)
        border_length : float
            Length of the shared border in kilometers (default: 0.0)
        
        Returns:
        --------
        None
        '''
        if from_county not in self.county_adjacency_graph:
            print(f"Error: from_county {from_county} not found in the county adjacency graph.")
        elif to_county not in self.county_adjacency_graph:
            print(f"Error: to_county {to_county} not found in the county adjacency graph.")
        else:
            self.county_adjacency_graph.add_edge(from_county, to_county, weight=edge_weight, overlap_length_km=border_length, custom_added=True)
            if self.verbose > 0:
                print(f"Added an additional edge from {from_county} to {to_county} with weight {edge_weight} and border length {border_length} to the county adjacency graph.")
    
    def auto_process_all_counties(self,
                                  min_customers_before_gap: int = 10,
                                  min_customers_after_gap: int = 2,
                                  max_gap_minutes: int = 24*60, # 1 day
                                  detection_method: str = "flat",
                                  total_customers: int=0,
                                  ac_customers_threshold: int = 30
                                  ):
        '''
        Automatically process all counties in the state to identify gaps, fill gaps, and extract events.
        
        Parameters:
        -----------
        min_customers_before_gap : int
            Minimum number of customers before a gap to consider it valid (default: 10)
        min_customers_after_gap : int
            Minimum number of customers after a gap to consider it valid (default: 2)
        max_gap_minutes : int
            Maximum gap duration in minutes to consider for filling (default: 1440 minutes = 1 day)
        event_detection_type : str
            Method for determining customer threshold (default: flat)
        ac_customers_threshold : int
            Customer threshold for event extraction (default: 30)
        
        Returns:
        --------
        None
        '''

        for county in self.counties:
            if self.verbose > 0:
                print(f"\nProcessing County: {county}")
            county_processor = EagleiCountyProcessor(
                eaglei_df=self.eaglei_df,
                county_name=county,
                verbose=self.verbose
            )
            county_processor.identify_gaps(
                min_customers_before_gap=min_customers_before_gap,
                min_customers_after_gap=min_customers_after_gap,
                max_gap_minutes=max_gap_minutes
            )
            county_processor.fill_gaps(
                auto_decide_rank_threshold=True
            )
            county_processor.extract_events_ac_thr(
                event_detection_type=detection_method,
                total_customers=total_customers,
                customer_threshold=ac_customers_threshold,
                crossing_mode='both'
            )
            self.county_processors[county] = county_processor
        
        # Create one dataframe with all counties data
        self.all_counties_events_df = pd.concat([processor.county_df_with_events_ac_thr for processor in self.county_processors.values()], ignore_index=True)
        # Only keep records where customers_out >= customer_threshold
        self.all_counties_events_df = self.all_counties_events_df[self.all_counties_events_df[constants.CUSTOMERS_COL] >= ac_customers_threshold].copy()
        # Sort by timestamp
        self.all_counties_events_df = self.all_counties_events_df.sort_values(constants.TIMESTAMP_COL).reset_index(drop=True)
        print(f"\nTotal records with outages >= {ac_customers_threshold}: {len(self.all_counties_events_df)}")
        print(f"Date range: {self.all_counties_events_df[constants.TIMESTAMP_COL].min()} to {self.all_counties_events_df[constants.TIMESTAMP_COL].max()}")

        event_number_column = f'event_number_ac_threshold_{ac_customers_threshold}'
        # Remove duplicate event numbers across counties and reassign new event numbers
        temp_df = self.all_counties_events_df[['county', event_number_column]].copy()
        temp_df = temp_df.drop_duplicates(subset=['county', event_number_column], keep='first')
        temp_df = temp_df.sort_values(by=['county', event_number_column])
        temp_df['new_ids'] = np.arange(1, len(temp_df)+1)
        self.all_counties_events_df = pd.merge(self.all_counties_events_df, temp_df, on=['county', event_number_column], how='left')
        self.all_counties_events_df = self.all_counties_events_df.drop(event_number_column, axis=1)
        self.all_counties_events_df = self.all_counties_events_df.rename(columns={'new_ids': event_number_column})

        # Identify temporal overlapping events across all counties
        self.all_counties_events_df = find_time_overlapping_groups(self.all_counties_events_df, 
                                                                   event_col=event_number_column)
        
        self.ac_customers_threshold = ac_customers_threshold

    def save_spatiotemporal_event_plots(self, temporal_event_id: int, pdf_path: str|None = None):
        '''
        Save spatiotemporal event plots for all events in the state to a PDF file.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to plot
        pdf_path : str
            Path to save the PDF file (default: None, which saves to 'RESULTS_DIR/spatiotemporal_events_{state_name}.pdf')
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # check if the event id exists in the all_counties_events_df
        if temporal_event_id not in self.all_counties_events_df['event_number_temporal'].unique():
            raise ValueError(f"Temporal event ID {temporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        save_plots_to_pdf(events_df = self.all_counties_events_df, 
                          event_id = temporal_event_id, 
                          overlap_method = 'standard_overlap', 
                          graph_of_counties = self.county_adjacency_graph,
                          county_event_col = event_number_column,
                          pdf_path = pdf_path)
        
    def get_spatiotemporal_events(self, temporal_event_id: int) -> pd.DataFrame:
        '''
        Get the spatiotemporal events for a specific temporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to get spatiotemporal events for
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing spatiotemporal events
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # check if the event id exists in the all_counties_events_df
        if temporal_event_id not in self.all_counties_events_df['event_number_temporal'].unique():
            raise ValueError(f"Temporal event ID {temporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        
        spatiotemporal_events = segregate_by_space(events_df = self.all_counties_events_df, 
                                                   temporal_event_number = temporal_event_id,
                                                   temporal_event_col='event_number_temporal', 
                                                   county_event_col=event_number_column,
                                                   graph_of_counties=self.county_adjacency_graph,
                                                   time_overlap_method='standard_overlap')
        return spatiotemporal_events
    
    def view_event_map_temporal(self, temporal_event_id: int):
        '''
        View the event map for a specific temporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # check if the event id exists in the all_counties_events_df
        if temporal_event_id not in self.all_counties_events_df['event_number_temporal'].unique():
            raise ValueError(f"Temporal event ID {temporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        plot_event_on_map(events_df = self.all_counties_events_df, 
                          event_id = temporal_event_id, 
                          event_col='event_number_temporal',
                          county_wide_events_col=event_number_column,
                          state_fips_code=self.state_fips_code)
        
    def view_event_map_spatiotemporal(self, temporal_event_id: int, spatiotemporal_event_id: int):
        '''
        View the event map for a specific spatiotemporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to get spatiotemporal events for
        spatiotemporal_event_id : int
            Spatiotemporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # get the spatiotemporal events for the given temporal event id
        spatiotemporal_events = self.get_spatiotemporal_events(temporal_event_id)
        
        # check if the event id exists in the all_counties_events_df
        if spatiotemporal_event_id not in spatiotemporal_events['event_number_spatiotemporal'].unique():
            raise ValueError(f"Spatiotemporal event ID {spatiotemporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        plot_event_on_map(events_df = spatiotemporal_events, 
                          event_id = spatiotemporal_event_id, 
                          event_col='event_number_spatiotemporal',
                          county_wide_events_col=event_number_column,
                          state_fips_code=self.state_fips_code)
        
    def view_event_timeline_temporal(self, temporal_event_id: int):
        '''
        View the event timeline for a specific temporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # check if the event id exists in the all_counties_events_df
        if temporal_event_id not in self.all_counties_events_df['event_number_temporal'].unique():
            raise ValueError(f"Temporal event ID {temporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        plot_event_timeline(events_df = self.all_counties_events_df, 
                            event_id = temporal_event_id, 
                            event_col_to_identify='event_number_temporal',
                            event_col=event_number_column)

    def view_event_timeline_spatiotemporal(self, temporal_event_id: int, spatiotemporal_event_id: int):
        '''
        View the event timeline for a specific spatiotemporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to get spatiotemporal events for
        spatiotemporal_event_id : int
            Spatiotemporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # get the spatiotemporal events for the given temporal event id
        spatiotemporal_events = self.get_spatiotemporal_events(temporal_event_id)
        
        # check if the event id exists in the all_counties_events_df
        if spatiotemporal_event_id not in spatiotemporal_events['event_number_spatiotemporal'].unique():
            raise ValueError(f"Spatiotemporal event ID {spatiotemporal_event_id} not found in the events data.")
        
        event_number_column = f'event_number_ac_threshold_{self.ac_customers_threshold}'
        plot_event_timeline(events_df = spatiotemporal_events, 
                            event_id = spatiotemporal_event_id, 
                            event_col_to_identify='event_number_spatiotemporal',
                            event_col=event_number_column)

    def view_performance_curve_temporal(self, temporal_event_id: int):
        '''
        View the multicounty performance curve for a specific temporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # check if the event id exists in the all_counties_events_df
        if temporal_event_id not in self.all_counties_events_df['event_number_temporal'].unique():
            raise ValueError(f"Temporal event ID {temporal_event_id} not found in the events data.")
        
        plot_eaglei_multicounty_performance_curve(events_df = self.all_counties_events_df, 
                                                  event_number = temporal_event_id, 
                                                  event_method='temporal')
        
    def view_performance_curve_spatiotemporal(self, temporal_event_id: int, spatiotemporal_event_id: int):
        '''
        View the performance curve for a specific spatiotemporal event ID.

        Parameters:
        -----------
        temporal_event_id : int
            Temporal event ID to get spatiotemporal events for
        spatiotemporal_event_id : int
            Spatiotemporal event ID to plot
        
        Returns:
        --------
        None
        '''
        if self.all_counties_events_df.empty:
            raise ValueError("No events data found. Please run auto_process_all_counties() first.")
        
        # get the spatiotemporal events for the given temporal event id
        spatiotemporal_events = self.get_spatiotemporal_events(temporal_event_id)
        
        # check if the event id exists in the all_counties_events_df
        if spatiotemporal_event_id not in spatiotemporal_events['event_number_spatiotemporal'].unique():
            raise ValueError(f"Spatiotemporal event ID {spatiotemporal_event_id} not found in the events data.")
        
        plot_eaglei_multicounty_performance_curve(events_df = spatiotemporal_events, 
                                                  event_number = spatiotemporal_event_id, 
                                                  event_method='spatiotemporal')

    def get_county_processor(self, county_name: str) -> EagleiCountyProcessor:
        '''
        Get the EagleiCountyProcessor instance for a specific county.
        
        Parameters:
        -----------
        county_name : str
            Name of the county
        
        Returns:
        --------
        EagleiCountyProcessor
            Instance of EagleiCountyProcessor for the specified county
        '''
        if county_name not in self.counties:
            raise ValueError(f"County {county_name} not found in the counties list.")
        return EagleiCountyProcessor(
                eaglei_df=self.eaglei_df,
                county_name=county_name,
                verbose=self.verbose
            )



# ------------------------- Public API for EagleiCountyProcessor -------------------------

# A class which can load data for a specific county from the eaglei_df and perform the gap filling and event extraction
class EagleiCountyProcessor:
    def __init__(self, 
                 eaglei_df: pd.DataFrame, 
                 county_name: str, 
                 customer_column: str = constants.CUSTOMERS_COL, 
                 timestamp_column: str = constants.TIMESTAMP_COL,
                 verbose: int = 1):
        
        self.verbose = verbose

        # check if customer_column and timestamp_column are in the eaglei_df
        if customer_column not in eaglei_df.columns:
            raise ValueError(f"Customer column {customer_column} not found in the EAGLE-i data.")
        if timestamp_column not in eaglei_df.columns:
            raise ValueError(f"Timestamp column {timestamp_column} not found in the EAGLE-i data.")
        
        self.customer_column = customer_column
        self.timestamp_column = timestamp_column

        # check if timestamp_column is in datetime format
        if not is_datetime64_any_dtype(eaglei_df[timestamp_column]):
            raise ValueError(f"Timestamp column {timestamp_column} is not in datetime format.")
        
        # check if county_name is in the eaglei_df
        if county_name not in eaglei_df['county'].unique():
            raise ValueError(f"County name {county_name} not found in the EAGLE-i data.")
        
        self.county_name = county_name
        self.county_df = eaglei_df[eaglei_df['county'] == county_name].copy().reset_index(drop=True)
        
        if self.verbose > 0:
            print(f"Total Data Points in EAGLEi: {eaglei_df.shape[0]}")
            print(f"Total Data Points in EAGLEi ({county_name}): {self.county_df.shape[0]} ({(self.county_df.shape[0]/eaglei_df.shape[0])*100:.2f}%)")
        
        self.gaps_customer_df = None
        self.county_df_filled = None
        self.gaps_rank_quantile = None
        self.county_df_with_events = None
        self.event_stats_ac = None
        self.county_df_with_events_ac_thr = None
        self.event_stats_ac_thr = None


    def data_quality_analysis(self,
                              year: int = 2024):
        # check if leap year
        if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
            expected_num_timestamps=366*24*4 # 366 days - 24 hr - 4 timestamps per hr
        else:
            expected_num_timestamps=365*24*4
        # need to make this work per year
        num_timestamps=len(self.customer_column)
        DQI=num_timestamps/expected_num_timestamps
        print(f"DQI: {DQI}")

    def identify_gaps(self, 
                      min_customers_before_gap: int = 10,
                      min_customers_after_gap: int = 2,
                      max_gap_minutes: int = 24*60 # 1 day
                      ):
        self.gaps_customer_df = identify_and_rank_time_gaps(
            self.county_df.copy(), 
            min_customers_before_gap = min_customers_before_gap,
            min_customers_after_gap = min_customers_after_gap,
            max_gap_minutes = max_gap_minutes,
            timestamp_column = self.timestamp_column,
            customer_column = self.customer_column,
            verbose = self.verbose
        )
        if not self.gaps_customer_df.empty:
            analyze_gap_rankings(self.gaps_customer_df, top_n=10, verbose=self.verbose)


    def gaps_distribution_at_quantile(self, 
                                      rank_threshold_quantile: float = 0.40):
        if self.gaps_customer_df is None:
            raise ValueError("Gaps must be identified before analyzing distribution.")
        if self.gaps_customer_df.empty:
            print("No gaps identified to analyze.")
            return

        decided_rank_threshold = self.gaps_customer_df['rank'].quantile(rank_threshold_quantile)

        if self.verbose > 0:
            print(f"Decided Rank Threshold at Quantile {rank_threshold_quantile}: {decided_rank_threshold:.2f}")
            print(f"Gaps that will be Filled: {self.gaps_customer_df[self.gaps_customer_df['rank']>decided_rank_threshold].shape[0]} out of {self.gaps_customer_df.shape[0]} total gaps ({(self.gaps_customer_df[self.gaps_customer_df['rank']>decided_rank_threshold].shape[0]/self.gaps_customer_df.shape[0])*100:.2f}%)")

        # Distribution of gap durations that will be filled
        filled_distribution = self.gaps_customer_df[self.gaps_customer_df['rank']>decided_rank_threshold]['duration_category'].value_counts().sort_index()
        if self.verbose > 0:
            print("Distribution of gap durations that will be filled:")
            print(filled_distribution)
        
        # Distribution of gap durations that will not be filled
        not_filled_distribution = self.gaps_customer_df[self.gaps_customer_df['rank']<=decided_rank_threshold]['duration_category'].value_counts().sort_index()
        if self.verbose > 0:
            print("Distribution of gap durations that will not be filled:")
            print(not_filled_distribution)

    def visualize_gaps(self, 
                       rank_threshold_quantile = None):
        if self.gaps_customer_df is None:
            raise ValueError("Gaps must be identified before visualization.")
        if self.gaps_customer_df.empty:
            print("No gaps identified to visualize.")
            return
        if rank_threshold_quantile is None:
            if self.gaps_rank_quantile is not None:
                visualize_gap_analysis(self.gaps_customer_df, self.gaps_rank_quantile)
            else:
                print("No rank_threshold_quantile provided and no previous quantile found. Please provide a quantile value.")
        else:
            visualize_gap_analysis(self.gaps_customer_df, rank_threshold_quantile)

    def fill_gaps(self,
                  auto_decide_rank_threshold: bool = True, 
                  rank_threshold_quantile: float = 0.40):
        if self.gaps_customer_df is None:
            raise ValueError("Gaps must be identified before filling.")
        if self.gaps_customer_df.empty:
            print("No gaps identified to fill. Copying original data.")
            self.county_df_filled = self.county_df.copy()
            self.county_df_filled['filled_gap'] = None
            return
        
        if auto_decide_rank_threshold:
            candidate_quantiles = [q/100.0 for q in range(90, 1, -1)]
            for q in candidate_quantiles:
                decided_rank_threshold = self.gaps_customer_df['rank'].quantile(q)
                not_filled_distribution = self.gaps_customer_df[self.gaps_customer_df['rank']<=decided_rank_threshold]['duration_category'].value_counts().sort_index()
                if (not_filled_distribution['<30min'] == 0) and (not_filled_distribution['30-60min'] == 0):
                    rank_threshold_quantile = q
                    if self.verbose > 0:
                        print(f"Selected quantile: {q}")
                    break

        decided_rank_threshold = self.gaps_customer_df['rank'].quantile(rank_threshold_quantile)
        self.gaps_rank_quantile = rank_threshold_quantile
        self.county_df_filled = fill_data_gaps_eaglei(
            self.county_df,
            self.gaps_customer_df,
            timestamp_column=self.timestamp_column,
            rank_threshold=decided_rank_threshold,
            verbose=self.verbose
        )

    def extract_events_ac(self):
        if self.county_df_filled is None:
            raise ValueError("Data gaps must be filled before extracting events.")
        
        self.county_df_with_events = extract_events_eaglei_ac(self.county_df_filled, timestamp_column=self.timestamp_column)
        
        if self.county_df_with_events is None or 'event_number_ac' not in self.county_df_with_events.columns:
            raise ValueError("Event extraction failed or 'event_number_ac' column not found.")
        
        if self.verbose > 0:
            print(f"Total Events Created (AC): {self.county_df_with_events['event_number_ac'].nunique()}")
        
        # Ensure event_stats_ac is always a DataFrame (get_eaglei_event_stats may return a dict for a single event)
        _stats = get_eaglei_event_stats(self.county_df_with_events,
                                       event_numbers = self.county_df_with_events['event_number_ac'].unique(),
                                       event_method = 'ac',
                                       timestamp_column = self.timestamp_column,
                                       customer_column = self.customer_column)
        if isinstance(_stats, dict):
            # wrap single-event dict into a DataFrame
            self.event_stats_ac = pd.DataFrame([_stats])
        else:
            self.event_stats_ac = _stats

    def extract_events_ac_thr(self,
                              event_detection_type: str = "flat",
                              total_customers: int=0,
                              customer_threshold: float = 10, 
                              crossing_mode: str = 'both'):
        
        if self.county_df_filled is None:
            raise ValueError("Data gaps must be filled before extracting events.")
        
        self.county_df_with_events_ac_thr = extract_events_eaglei_ac_threshold(self.county_df_filled, 
                                                                               timestamp_column=self.timestamp_column, 
                                                                               customer_column=self.customer_column,
                                                                               event_detection_type=event_detection_type,
                                                                               total_customers=total_customers,
                                                                               customer_threshold=customer_threshold, 
                                                                               crossing_mode=crossing_mode,
                                                                               active_only=True)
        # event_col_name = f'event_number_ac_threshold_{event_detection_type}_{customer_threshold}'
        event_col_name = [col for col in self.county_df_with_events_ac_thr.columns if col.startswith('event_number_')]
        if len(event_col_name) == 0:
            raise ValueError("Event extraction failed or event number column not found.")
        elif len(event_col_name) > 1:
            raise ValueError("Multiple event number columns found. Unable to determine the correct one.")
        event_col_name = event_col_name[0]
        
        if self.verbose > 0:
            print(f"Total Events Created (AC with Threshold = {event_detection_type} at {customer_threshold}): {self.county_df_with_events_ac_thr[event_col_name].nunique()}")
        
        event_method_name = f'ac_threshold_{event_detection_type}_{event_col_name.split("_")[-1]}'
        self.event_stats_ac_thr = get_eaglei_event_stats(self.county_df_with_events_ac_thr,
                                                         event_numbers = self.county_df_with_events_ac_thr[event_col_name].unique(),
                                                         event_method = event_method_name,
                                                         timestamp_column = self.timestamp_column,
                                                         customer_column = self.customer_column)
    
    def plot_top_n_largest_events(self, 
                                  event_method: str = 'ac', 
                                  top_n: int = 50):
        
        if (top_n <= 0) or (top_n > 100):
            raise ValueError("top_n must be a positive integer between 1 and 100.")
        
        if event_method == 'ac':
            if (self.county_df_with_events is None) or (self.event_stats_ac is None):
                raise ValueError("Events must be extracted before plotting.")
            top_n_event_numbers = self.event_stats_ac.sort_values(by='num_outages', ascending=False).iloc[0:top_n]['event_number'].tolist()
            plot_multiple_eaglei_performance_curves(self.county_df_with_events,
                                                    top_n_event_numbers,
                                                    event_method=event_method)
        elif event_method.startswith('ac_threshold_'):
            if self.county_df_with_events_ac_thr is None or self.event_stats_ac_thr is None:
                raise ValueError("AC threshold events must be extracted before plotting.")
            top_n_event_numbers = self.event_stats_ac_thr.sort_values(by='num_outages', ascending=False).iloc[0:top_n]['event_number'].tolist()
            plot_multiple_eaglei_performance_curves(self.county_df_with_events_ac_thr,
                                                    top_n_event_numbers,
                                                    event_method=event_method)
        else:
            raise ValueError("event_method must be 'ac' or start with 'ac_threshold_'.")
        
    def plot_customers_histograms(self):
        if self.county_df_filled is None:
            raise ValueError("Data gaps must be filled before this action.")
        
        # create a subplot with 1 row and 3 columns
        _, axs = plt.subplots(1, 3, figsize=(18,5))

        # Customers > 0
        series = self.county_df_filled.loc[self.county_df_filled[self.customer_column] > 0, self.customer_column]
        tmp_cust = np.asarray(series, dtype=float)
        if tmp_cust.size == 0:
            tmp_cust_log = np.array([])
        else:
            tmp_cust_log = np.log10(tmp_cust)
        sns.histplot(tmp_cust_log, bins=50, kde=False, ax=axs[0])
        axs[0].set_title('Customers Out > 0')
        axs[0].set_xlabel('Log of Customers Out')
        axs[0].set_ylabel('Frequency')

        # Customers > 1
        series = self.county_df_filled.loc[self.county_df_filled[self.customer_column] > 1, self.customer_column]
        tmp_cust = np.asarray(series, dtype=float)
        if tmp_cust.size == 0:
            tmp_cust_log = np.array([])
        else:
            tmp_cust_log = np.log10(tmp_cust)
        sns.histplot(tmp_cust_log, bins=50, kde=False, ax=axs[1])
        axs[1].set_title('Customers Out > 1')
        axs[1].set_xlabel('Log of Customers Out')
        axs[1].set_ylabel('Frequency')

        # Customers > 2
        series = self.county_df_filled.loc[self.county_df_filled[self.customer_column] > 2, self.customer_column]
        tmp_cust = np.asarray(series, dtype=float)
        if tmp_cust.size == 0:
            tmp_cust_log = np.array([])
        else:
            tmp_cust_log = np.log10(tmp_cust)
        sns.histplot(tmp_cust_log, bins=50, kde=False, ax=axs[2])
        axs[2].set_title('Customers Out > 2')
        axs[2].set_xlabel('Log of Customers Out')
        axs[2].set_ylabel('Frequency')

        plt.suptitle(f'Histogram of Log of Customers Out in {self.county_name} County (Eagle-i individual outages)', fontsize=16)
        plt.tight_layout()
        plt.show()

