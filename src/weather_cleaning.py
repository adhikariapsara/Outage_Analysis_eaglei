import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from shapely.geometry import Point
# from shapely.ops import nearest_points

# =====================================
# Define the common 15-minute timestamp range
# =====================================
# start = pd.Timestamp('2018-01-01 00:15:00')
# end = pd.Timestamp('2024-12-30 23:45:00')
# full_time_range = pd.date_range(start=start, end=end, freq="15min")
# full_df = pd.DataFrame({"valid": full_time_range})

# =====================================
# Function to preprocess individual weather datasets
# =====================================
def preprocess_weather_data(full_df, df, column, replace_dict=None, create_occurrence=False, keep_max=True, drop_columns=None):

    df["valid"] = pd.to_datetime(df["valid"]).dt.round("15min")

    if replace_dict:
        df[column] = df[column].replace(replace_dict)

    df[column] = pd.to_numeric(df[column], errors='coerce')

    if keep_max:
        df = df.sort_values(by=["valid", column], ascending=[True, False])
        df = df.groupby("valid").first().reset_index()

    df = full_df.merge(df, on="valid", how="left")

    df[column] = df[column].interpolate(method="nearest")
    df.ffill(inplace=True)

    # ALWAYS drop problematic columns if they exist
    for col in ['station', 'lat', 'lon']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    if drop_columns:
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)

    if create_occurrence:
        df["poccurence"] = df[column].apply(lambda x: 0 if x == 0 else 1)

    return df


def create_netcdf_weather_stations(state, county, start_year, end_year, raw_data_file, netcdf_output_file, weather_variables, min_data_threshold=0.05):
    """
    Create a NetCDF file with weather data for all stations organized by timestamp.
    
    Parameters:
    -----------
    state : str
        State name
    county : str
        County name
    start_year : int
        Start year for the time range
    end_year : int
        End year for the time range
    raw_data_file : str
        Path to the raw weather data parquet file
    netcdf_output_file : str
        Path where the NetCDF file will be saved
    weather_variables : list
        List of weather variables to include (e.g., ['tmpf', 'sknt', 'p01i', 'gust'])
    min_data_threshold : float, optional
        Minimum fraction of non-null data required for a station to be included (default: 0.05 = 5%)
    
    Returns:
    --------
    xarray.Dataset
        The created dataset (also saved to file)
    """
    
    print(f"Loading weather data from {raw_data_file}")
    weather_data = pd.read_parquet(raw_data_file)
    
    # Ensure required columns exist
    required_cols = ['station', 'valid', 'lon', 'lat']
    columns_to_keep = required_cols.copy()
    columns_to_keep.extend([var for var in weather_variables if var in weather_data.columns])
    weather_data = weather_data[columns_to_keep].copy()
    
    # Round timestamps to 15-minute intervals
    weather_data['valid'] = pd.to_datetime(weather_data['valid']).dt.round('15min')
    
    # Create full time range
    start = pd.Timestamp(f'{start_year}-01-01 00:00:00')
    end = pd.Timestamp(f'{end_year}-12-31 23:45:00')
    full_time_range = pd.date_range(start=start, end=end, freq='15min')
    
    # Get unique stations
    stations = weather_data['station'].unique()
    print(f"Found {len(stations)} weather stations")
    
    # Filter stations based on data availability
    valid_stations = []
    station_metadata = {}
    
    for station in stations:
        station_data = weather_data[weather_data['station'] == station]
        
        # Calculate data availability for key variables
        total_timestamps = len(full_time_range)
        available_timestamps = len(station_data)
        data_fraction = available_timestamps / total_timestamps
        
        # Check if station has sufficient data
        if data_fraction >= min_data_threshold:
            valid_stations.append(station)
            # Store metadata (use first available lat/lon)
            station_metadata[station] = {
                'lat': station_data['lat'].iloc[0],
                'lon': station_data['lon'].iloc[0]
            }
        else:
            print(f"Skipping station {station} - insufficient data ({data_fraction*100:.1f}%)")
    
    print(f"Retained {len(valid_stations)} stations with sufficient data")
    
    if len(valid_stations) == 0:
        raise ValueError("No stations have sufficient data. Consider lowering min_data_threshold.")
    
    # Initialize data arrays
    n_times = len(full_time_range)
    n_stations = len(valid_stations)
    
    # Create dictionary to store data arrays
    data_vars = {}
    
    # Station metadata arrays
    station_lats = np.array([station_metadata[s]['lat'] for s in valid_stations])
    station_lons = np.array([station_metadata[s]['lon'] for s in valid_stations])
    
    # Process each weather variable
    for var in weather_variables:
        if var not in weather_data.columns:
            print(f"Warning: Variable '{var}' not found in weather data. Skipping.")
            continue
        
        print(f"Processing variable: {var}")
        
        # Initialize array with NaN
        var_array = np.full((n_times, n_stations), np.nan, dtype=np.float32)
        
        # Fill data for each station
        for i, station in enumerate(valid_stations):

            # replace missing wind gust values with corresponding wind speed
            if var == 'gust':
                mask = weather_data[var] == 'M'
                weather_data.loc[mask, var] = weather_data.loc[mask, 'sknt']

            station_data = weather_data[weather_data['station'] == station][['valid', var]].copy()

            # Handle special replacements for precipitation
            if var == 'p01i':
                station_data[var] = station_data[var].replace({'T': 0.001, 'M': 0})

            # Convert to numeric
            station_data[var] = pd.to_numeric(station_data[var], errors='coerce')

            # Remove duplicates, keeping max value
            station_data = station_data.sort_values(by=[var], ascending=False)
            station_data = station_data.groupby('valid').first().reset_index()
            
            # Create a full time series for this station
            station_full = pd.DataFrame({'valid': full_time_range})
            station_full = station_full.merge(station_data, on='valid', how='left')
            
            # Interpolate missing values
            station_full[var] = station_full[var].interpolate(method='nearest')
            station_full[var] = station_full[var].ffill()
            station_full[var] = station_full[var].bfill()
            
            # Fill the array
            var_array[:, i] = station_full[var].values
        
        data_vars[var] = (['time', 'station'], var_array)
    
    # Create additional variables for precipitation occurrence if p01i exists
    if 'p01i' in data_vars:
        precip_occurrence = (data_vars['p01i'][1] > 0).astype(np.float32)
        data_vars['poccurence'] = (['time', 'station'], precip_occurrence)
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': full_time_range,
            'station': valid_stations,
            'lat': ('station', station_lats),
            'lon': ('station', station_lons)
        },
        attrs={
            'title': f'Weather Station Data for {county}, {state}',
            'description': 'Time-series weather data from multiple stations with 15-minute resolution',
            'state': state,
            'county': county,
            'start_year': start_year,
            'end_year': end_year,
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temporal_resolution': '15 minutes',
            'min_data_threshold': min_data_threshold
        }
    )
    
    # Add variable attributes
    if 'tmpf' in ds.data_vars:
        ds['tmpf'].attrs = {'long_name': 'Air Temperature', 'units': 'degrees Fahrenheit'}
    if 'sknt' in ds.data_vars:
        ds['sknt'].attrs = {'long_name': 'Wind Speed', 'units': 'knots'}
    if 'gust' in ds.data_vars:
        ds['gust'].attrs = {'long_name': 'Wind Gust', 'units': 'knots'}
    if 'p01i' in ds.data_vars:
        ds['p01i'].attrs = {'long_name': 'Precipitation (1-hour)', 'units': 'inches'}
    if 'poccurence' in ds.data_vars:
        ds['poccurence'].attrs = {'long_name': 'Precipitation Occurrence', 'units': 'boolean (0/1)'}
    
    ds['lat'].attrs = {'long_name': 'Latitude', 'units': 'degrees_north'}
    ds['lon'].attrs = {'long_name': 'Longitude', 'units': 'degrees_east'}
    
    # Save to NetCDF
    print(f"Saving NetCDF file to {netcdf_output_file}")
    ds.to_netcdf(netcdf_output_file)
    
    print(f"NetCDF file created successfully!")
    print(f"  - Time points: {n_times}")
    print(f"  - Stations: {n_stations}")
    print(f"  - Variables: {list(data_vars.keys())}")
    
    return ds


# =====================================
# Preprocess each weather component
# =====================================
def main(state, county, start_year, end_year, raw_data_file, cleaned_data_file, weather_variables):

    # Event Datasets
    weather_data=pd.read_parquet(raw_data_file)
    # columns_to_keep = ['station', 'valid', 'tmpf', 'sknt', 'p01i', 'gust', 'lon', 'lat']
    columns_to_keep = ['station', 'valid', 'lon', 'lat']
    columns_to_keep.extend([var for var in weather_variables if var in weather_data.columns])
    weather_data = weather_data[columns_to_keep].copy()
    print("Cleaning weather data.")

    start = pd.Timestamp(f'{start_year}-01-01 00:15:00')
    end = pd.Timestamp(f'{end_year}-12-31 23:45:00')
    full_time_range = pd.date_range(start=start, end=end, freq="15min")
    full_time_range_df = pd.DataFrame({"valid": full_time_range})
    # print(full_time_range_df.shape)
    
    temp = [full_time_range_df.copy()]
    for var in [v for v in weather_variables if v != 'p01i']:
        if var not in weather_data.columns:
            raise ValueError(f"Variable '{var}' not found in weather data.")
        else:
            returned_df = preprocess_weather_data(full_time_range_df, weather_data, var)
            # print(returned_df.shape)
            temp.append(returned_df[var].copy())

    if 'p01i' in weather_variables:
        returned_df = preprocess_weather_data(
            full_df=full_time_range_df,
            df=weather_data,
            column='p01i',
            replace_dict={'T': 0.001, 'M': 0},
            create_occurrence=True,
            drop_columns=['station', 'lon', 'lat'],
            keep_max=True
        )
        # print(returned_df.shape)
        temp.append(returned_df[['p01i', 'poccurence']].copy())

    # =====================================
    # Merge all cleaned DataFrames on 'valid' timestamp
    # =====================================
    # Merge all DataFrames
    weather_dataset=pd.concat(temp,axis=1)
    # combined_df = df_temp.copy()
    # for df_component in [df_temp, df_gust, df_sped, df_precip]:
    #     combined_df = combined_df.merge(df_component, on='valid', how='inner')

    # Fill missing gust values with wind speed
    if 'gust' in weather_dataset.columns and 'sknt' in weather_dataset.columns:
        weather_dataset['gust'] = weather_dataset['gust'].fillna(weather_dataset['sknt'])
    
    print("Weather data has been successfully cleaned. Saving to parquet.")
    # Preview
    #print(weather_dataset.head())
    weather_dataset.to_parquet(cleaned_data_file)