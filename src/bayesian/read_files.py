import pandas as pd
import xarray as xr
import datetime as dt

def read_files(state,county,multi_county, start_year, end_year):
    if multi_county==False:
        ds = xr.open_dataset(f'../../merged_data/{state}/merged_data_{state}_{county}_2015_2025.nc')
        event_data = pd.read_parquet(f'../../events_stats/county_events/{state}/{county}_event_stats_2015_2025.parquet')

    else:
        ds = xr.open_dataset(
            f'../../multi_county_data/multi_county_data_2015_2025_{county}_counties.nc',
            chunks={"time": 10000})
        event_data = pd.read_parquet(f'../../events_stats/spatiotemporal_events/{county}_event_stats_2015_2025.parquet')
        # for comparison, we only care about eaglei event data
        event_data=event_data[event_data['event_method']=='eaglei']

    if start_year > 2015 or end_year < 2025:
        event_data['start_time'] = pd.to_datetime(event_data['start_time'])
        event_data['end_time'] = pd.to_datetime(event_data['end_time'])
        start_date = dt.datetime(start_year, 1, 1)
        end_date = dt.datetime(end_year, 12, 31)
        event_data = event_data[event_data['start_time'].dt.year >= start_date.year]
        event_data = event_data[event_data['end_time'].dt.year <= end_date.year]
        ds = ds.sel(time=slice(start_date, end_date))
    elif start_year < 2015 or end_year > 2025:
        raise ValueError("Error: Attempting to analyze years outside of data download range. Please select start"
                         "and end years between 2015 and 2025.")
    return ds, event_data