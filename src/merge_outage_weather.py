import pandas as pd
import numpy as np
import os
import xarray as xr

from src.configManager import ConfigManager

def merge_outages_with_weather_netcdf(state: str, 
                                      county: str, 
                                      start: int, 
                                      end: int, 
                                      config: ConfigManager) -> None:
    """
    Merge outage data with weather NetCDF file, adding outage variables as time-series data.
    
    This function loads an existing weather NetCDF file (with station-based weather data)
    and adds county-level outage data as additional time-series variables. The resulting
    merged NetCDF allows users to select any time range and access both weather data
    (from any station) and outage data for that period.
    
    Parameters:
    -----------
    state : str
        State name
    county : str
        County name
    start : int
        Start year
    end : int
        End year
    config : ConfigManager
        Configuration manager object
    """
    
    print(f"Merging outage data with weather NetCDF for {county}, {state} ({start}-{end})")
    
    # Create file paths
    weather_file_dir = os.path.join(config.get("data_paths.weather_data_dir"), state)
    weather_file_name = config.get("file_patterns.cleaned_weather_file_pattern").format(state=state, county=county, start=start, end=end)
    weather_file_path = os.path.join(weather_file_dir, weather_file_name)

    outage_file_dir = os.path.join(config.get("data_paths.outage_data_dir"), state)
    outage_file_name = config.get("file_patterns.cleaned_outage_file_pattern").format(start=start, end=end, county=county, state=state)
    outage_file_path = os.path.join(outage_file_dir, outage_file_name)
    
    # Output path for merged NetCDF
    merged_file_dir = os.path.join(config.get("data_paths.merged_data_dir"), state)
    merged_file_name = config.get("file_patterns.merged_file_pattern").format(start=start, end=end, county=county, state=state)
    merged_file_path = os.path.join(merged_file_dir, merged_file_name)
    
    # Check if weather NetCDF exists
    if not os.path.exists(weather_file_path):
        raise FileNotFoundError(f"Weather NetCDF file not found: {weather_file_path}")
    
    if not os.path.exists(outage_file_path):
        raise FileNotFoundError(f"Outage data file not found: {outage_file_path}")
    
    # Load weather NetCDF dataset
    print(f"Loading weather NetCDF from {weather_file_path}")
    ds_weather = xr.open_dataset(weather_file_path)
    
    # Load outage data
    print(f"Loading outage data from {outage_file_path}")
    df_outage = pd.read_parquet(outage_file_path)
    
    # Get the time coordinate from weather dataset
    time_coord = ds_weather.coords['time']
    
    # Create a dataframe with full time range
    full_time_df = pd.DataFrame({'time': pd.to_datetime(time_coord.values)})

    # find the name of the event number column in outage data, that starts with 'event_number_'
    event_number_col = [col for col in df_outage.columns if col.startswith('event_number_')]
    if len(event_number_col) > 0:
        print(f"WARNING: Found multiple event number columns in outage data. Using the first one: {event_number_col[0]}" ) if len(event_number_col) > 1 else None
        event_number_col = event_number_col[0]
    else:
        event_number_col = None
        raise ValueError("No event number column found in outage data.")
    
    if event_number_col.startswith('event_number_ac_threshold_'):
        tokens = event_number_col.split("_")
        event_customer_threshold = tokens[-1]
        event_threshold_method = "_".join(tokens[4:-1])
    else:
        raise ValueError(f"Event number column name format is unexpected: {event_number_col}")
        
        
    
    # Merge outage data with full time range
    df_outage_full = full_time_df.merge(
        df_outage[['run_start_time', 'customers_out', event_number_col]].rename(columns={'run_start_time': 'time'}),
        on='time',
        how='left'
    )
    
    # Fill missing outage values with 0 (no outage)
    df_outage_full['customers_out'] = df_outage_full['customers_out'].fillna(0)
    df_outage_full[event_number_col] = df_outage_full[event_number_col].fillna(0)
    
    # Convert to numpy array
    outage_customers = df_outage_full['customers_out'].values.astype(np.int64)
    outage_event_numbers = df_outage_full[event_number_col].values.astype(np.int64)
    
    # Add outage data as new variables to the dataset
    ds_weather['customers_out'] = (['time'], outage_customers)
    ds_weather['customers_out'].attrs = {
        'long_name': 'Number of customers without power',
        'units': 'count',
        'description': 'Total number of customers experiencing power outage'
    }
    ds_weather['event_number_eaglei'] = (['time'], outage_event_numbers)
    ds_weather['event_number_eaglei'].attrs = {
        'long_name': 'Outage event number',
        'units': 'count',
        'description': f'Identifier for distinct outage events, based on a customer threshold of {event_customer_threshold}'
    }
    
    # # Calculate additional outage metrics if MCC data is available
    # try:
    #     mcc_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'MCC.csv')
    #     county_fips_file_path = os.path.join(config.get("data_paths.eaglei_data_dir"), 'county_fips_master.csv')
        
    #     if os.path.exists(mcc_file_path) and os.path.exists(county_fips_file_path):
    #         # Load MCC Data
    #         pdf = pd.read_csv(mcc_file_path)
    #         county_to_fips = pd.read_csv(county_fips_file_path, encoding='latin')
            
    #         # Find total number of customers in county
    #         ans = county_to_fips[county_to_fips['county_name'] == f'{county} County']
    #         ans = ans[ans['state_name'] == state]
            
    #         if len(ans) > 0:
    #             target_fips = ans['fips'].values[0]
    #             pdf['County_FIPS'] = pd.to_numeric(pdf['County_FIPS'], downcast='integer', errors='coerce')
    #             result = pdf[pdf['County_FIPS'] == target_fips]
                
    #             if len(result) > 0:
    #                 total_county_customers = result['Customers'].values[0]
                    
    #                 # Calculate normalized outage (fraction of total customers)
    #                 normalized_outage = outage_customers / total_county_customers
    #                 ds_weather['customers_out_normalized'] = (['time'], normalized_outage.astype(np.float32))
    #                 ds_weather['customers_out_normalized'].attrs = {
    #                     'long_name': 'Normalized customer outages',
    #                     'units': 'fraction',
    #                     'description': f'Fraction of total customers affected (total: {total_county_customers})'
    #                 }
                    
    #                 # Add total customers as an attribute
    #                 ds_weather.attrs['total_county_customers'] = int(total_county_customers)
                    
    #                 print(f"Added normalized outage data (total customers: {total_county_customers})")
    # except Exception as e:
    #     print(f"Note: Could not calculate normalized outage metrics: {e}")
    
    # Update dataset attributes
    ds_weather.attrs.update({
        'title': f'Merged Weather and Outage Data for {county}, {state} ({start}-{end})',
        'description': 'This dataset contains weather data from multiple stations within the county, along with county-level outage data as time-series variables.',
        'state': state,
        'county': county,
        'county_fips_code': str(df_outage['fips_code'].unique()[0]) if 'fips_code' in df_outage.columns else 'unknown',
        'start_year': start,
        'end_year': end,
        'temporal_resolution': '15 minutes',
        'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    })

    # Remove an existing attribute if present
    if 'min_data_threshold' in ds_weather.attrs:
        del ds_weather.attrs['min_data_threshold']

    # Add the event customer threshold and event method as attributes
    ds_weather.attrs['event_customer_threshold'] = event_customer_threshold
    ds_weather.attrs['event_threshold_method'] = event_threshold_method

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
        if len(total_customers) > 0:
            ds_weather.attrs['total_county_customers'] = int(total_customers[0])
        else:
            ds_weather.attrs['total_county_customers'] = 'unknown'
    else:
        ds_weather.attrs['total_county_customers'] = 'unknown'
        print(f"Note: Total customers data file not found: {customers_data_file_path}")
    
    # Save merged dataset
    print(f"Saving merged NetCDF to {merged_file_path}")
    # create directory if it does not exist
    os.makedirs(merged_file_dir, exist_ok=True)
    ds_weather.to_netcdf(merged_file_path)
    
    # Close the dataset
    ds_weather.close()
    
    print(f"Successfully merged outage and weather data!")
    print(f"  - Time points: {len(time_coord)}")
    print(f"  - Weather stations: {len(ds_weather.coords['station'])}")
    print(f"  - Weather variables: {[v for v in ds_weather.data_vars if v not in ['customers_out', 'customers_out_normalized', 'event_number_eaglei']]}")
    print(f"  - Outage variables: {['customers_out', 'event_number_eaglei']}")
    
    return None