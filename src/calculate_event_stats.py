# importing necessary libraries
import os
import pandas as pd
import xarray as xr

def calculate_events_stats(dataset, event_threshold, event_type):
    # get a single dataframe merging customers_out, event_number_eaglei and maximum weather variables across all stations
    event_col_name=f"event_number_{event_type}"
    temp1 = dataset[['customers_out', event_col_name]].to_dataframe().reset_index()
    temp2 = dataset[['tmpf', 'sknt', 'gust', 'p01i']].max(dim='station').to_dataframe().reset_index()
    merged_df = pd.merge(temp1, temp2, on='time')

    merged_df_non_zero = merged_df[merged_df['customers_out'] > 0].copy().reset_index()

    event_numbers = (
        merged_df_non_zero[event_col_name]
        .value_counts()
        .loc[lambda x: x >= event_threshold]
        .index
        .tolist()
    )
    # drop any zero or 1 event numbers
    event_numbers = [e for e in event_numbers if e not in (0, -1)]

    # import the function to calculate EAGLEi event stats
    from src.eaglei_modules.eagleiEventProcessing import get_eaglei_event_stats
    events_stats = get_eaglei_event_stats(merged_df_non_zero, event_numbers, event_method=event_type,
                                      timestamp_column='time')

    # append weather stats to the events_stats dataframe
    weather_stats_list = []
    for event_num in event_numbers:
        event_data = merged_df_non_zero[merged_df_non_zero[event_col_name] == event_num]
        weather_stats = {
            'event_number': event_num,
            'max_sknt': event_data['sknt'].max(),
            'min_sknt': event_data['sknt'].min(),
            'avg_sknt': event_data['sknt'].mean(),

            'max_gust': event_data['gust'].max(),
            'min_gust': event_data['gust'].min(),
            'avg_gust': event_data['gust'].mean(),

            'max_tmpf': event_data['tmpf'].max(),
            'min_tmpf': event_data['tmpf'].min(),
            'avg_tmpf': event_data['tmpf'].mean(),

            'total_p01i': event_data['p01i'].sum(),
            'max_p01i': event_data['p01i'].max(),
            'avg_p01i': event_data['p01i'].mean()
        }
        weather_stats_list.append(weather_stats)
    weather_stats_df = pd.DataFrame(weather_stats_list)
    events_stats = pd.merge(events_stats, weather_stats_df, on='event_number', how='left')

    return events_stats

# dataset params
ds_merged = xr.open_dataset(f'../multi_county_data/multi_county_data_2014_2024_102_counties.nc')
event_threshold=30
state='Illinois'

''' # CALCULATING EVENT STATS FOR EACH COUNTY

list_of_counties=list(ds_merged.counties.split(', '))
events_stats_col = 'event_number_eaglei'
event_type = 'eaglei'
path = '../events_stats/county_events'
if not os.path.exists(path):
    os.makedirs(path)

ds_merged["county"] = (
    ds_merged["county"]
    .str.replace(state, "", regex=False)
    .str.strip()
)

for county in list_of_counties:
    ds_county = ds_merged.sel(county=county)
    print(ds_county['customers_out'].head)
    events_stats = calculate_events_stats(ds_county, event_threshold, event_type)
    events_stats.to_parquet(f'{path}/{county}_event_stats.parquet')
    
'''

# CALCULATING SPATIOTEMPORAL EVENT STATS
path = '../events_stats/spatiotemporal_events'
if not os.path.exists(path):
    os.makedirs(path)

events_stats = calculate_events_stats(ds_merged,event_threshold,event_type='mc_spatiotemporal')

events_stats.to_parquet(f'{path}/event_stats_{state}.parquet')


