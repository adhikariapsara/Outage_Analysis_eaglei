# importing necessary libraries
import os
import pandas as pd
import xarray as xr
from src.eaglei_modules.eagleiEventProcessing import get_eaglei_event_stats
from src.configManager import ConfigManager

def match_event_with_weather_stats(weather_df, counties_affected, start_time, end_time, event_num):

    # fast datetime index slice
    selected = weather_df.loc[start_time:end_time]

    # county filter
    if isinstance(counties_affected, list):
        selected = selected[selected['county'].isin(counties_affected)]
    elif isinstance(counties_affected, str):
        selected = selected[selected['county'] == counties_affected]

    stats = selected.agg({
        'sknt': ['max', 'min', 'mean'],
        'gust': ['max', 'min', 'mean'],
        'tmpf': ['max', 'min', 'mean'],
        'p01i': ['sum', 'max', 'mean']
    })

    return {
        'event_number': event_num,

        'max_sknt': stats.loc['max', 'sknt'],
        'min_sknt': stats.loc['min', 'sknt'],
        'avg_sknt': stats.loc['mean', 'sknt'],

        'max_gust': stats.loc['max', 'gust'],
        'min_gust': stats.loc['min', 'gust'],
        'avg_gust': stats.loc['mean', 'gust'],

        'max_tmpf': stats.loc['max', 'tmpf'],
        'min_tmpf': stats.loc['min', 'tmpf'],
        'avg_tmpf': stats.loc['mean', 'tmpf'],

        'total_p01i': stats.loc['sum', 'p01i'],
        'max_p01i': stats.loc['max', 'p01i'],
        'avg_p01i': stats.loc['mean', 'p01i'],

    }

def calculate_event_stats_single_county(state: str, county: str, start: int, end: int, config: ConfigManager) -> None:

    # Output path for merged NetCDF
    merged_file_dir = os.path.join(config.get("data_paths.merged_data_dir"), state)
    merged_file_name = config.get("file_patterns.merged_file_pattern").format(start=start, end=end, county=county, state=state)
    merged_file_path = os.path.join(merged_file_dir, merged_file_name)

    ds = xr.open_dataset(merged_file_path)
    event_col_name=f"event_number_eaglei"
    outage_df = ds[['customers_out', event_col_name, 'time']].to_dataframe().reset_index()
    outage_df['county'] = county
    outage_df['state']=ds.attrs['state']
    weather_df=ds[['tmpf', 'sknt', 'gust', 'p01i','station','time']].to_dataframe().reset_index()
    weather_df['county'] = county
    weather_df = weather_df.set_index('time').sort_index()
    # filter valid events
    outage_df = outage_df[~outage_df[event_col_name].isin([0, -1])]
    event_numbers = outage_df[event_col_name].unique()
    print("Pt 1: Calculating Outage Stats.")

    # calc eagle-i event stats
    events_stats = get_eaglei_event_stats(
        outage_df,
        event_numbers=event_numbers,
        event_method='eaglei',
        timestamp_column='time',
        counties=county
    )
    weather_stats_list=[]
    # iterate through events and match weather at the time
    print("Pt 2: Calculating Weather Stats.")
    for row in events_stats.itertuples(index=False):
        start_time = row.start_time
        end_time = row.end_time
        event_num = row.event_number
        weather_stats_list.append(match_event_with_weather_stats(
        weather_df,
        county,
        start_time,
        end_time,
        event_num))

    # merge datasets, return
    weather_stats_df = pd.DataFrame(weather_stats_list)
    events_stats = pd.merge(events_stats, weather_stats_df, on='event_number', how='left')
    event_file_dir = os.path.join(config.get("data_paths.event_data_dir"), "county_events", state)
    event_file_name = config.get("file_patterns.event_stats_file_pattern").format(start=start, end=end, county=county)
    event_file_path = os.path.join(event_file_dir, event_file_name)
    events_stats.to_parquet(event_file_path)

def list_to_csv(value):
    if isinstance(value, list):
        return ', '.join(map(str, value))  # Convert each element to string
    elif pd.isna(value):
        return ''  # Handle NaN values
    else:
        return str(value) # Fallback for unexpected types

def calculate_event_stats_multi_county(ds, event_type):
    event_col_name=f"event_number_{event_type}"
    outage_df = ds[['customers_out', event_col_name, 'county', 'time']].to_dataframe().reset_index()
    outage_df['state']=ds.attrs['states']
    weather_df=ds[['tmpf', 'sknt', 'gust', 'p01i','station','time']].to_dataframe().reset_index()
    weather_df = weather_df.set_index('time').sort_index()
    # filter valid events
    outage_df = outage_df[~outage_df[event_col_name].isin([0, -1])]
    event_numbers = outage_df[event_col_name].unique()

    print("Pt 1: Calculating Outage Stats.")
    events_stats = get_eaglei_event_stats(
        outage_df,
        event_numbers=event_numbers,
        event_method=event_type,
        timestamp_column='time',
        counties=outage_df['county'].unique()
    )
    print("Pt 2: Calculating Weather Stats.")
    parts = weather_df["station"].str.split("_", expand=True)
    weather_df["county"] = parts[0]+'_'+parts[1]
    weather_stats_list=[]
    for row in events_stats.itertuples(index=False):
        if row.event_number % 100 == 0:
            print(f"  Processing weather stats for event {row.event_number}/{len(event_numbers)}")
        start_time = row.start_time
        end_time = row.end_time
        event_num = row.event_number
        counties_affected=row.counties_affected
        weather_stats_list.append(match_event_with_weather_stats(
        weather_df,
        counties_affected,
        start_time,
        end_time,
        event_num))
    weather_stats_df = pd.DataFrame(weather_stats_list)
    events_stats = pd.merge(events_stats, weather_stats_df, on=f'event_number', how='left')
    events_stats['event_method']=event_type
    events_stats['counties_affected'] = events_stats['counties_affected'].apply(list_to_csv)
    return events_stats

def initialize_multi_county_calc(start: int, end: int, config: ConfigManager) -> None:
    multi_county_data_file_dir = config.get("data_paths.multi_county_data_dir")
    os.makedirs(multi_county_data_file_dir, exist_ok=True)
    multi_county_data_file_name = (config.get("file_patterns.multi_county_data_file_pattern").format
                                   (start=start, end=end,
                                    label=config.get("multi_county_analysis_parameters.label")))
    multi_county_data_file_path = os.path.join(multi_county_data_file_dir, multi_county_data_file_name)

    ds=xr.open_dataset(multi_county_data_file_path)
    print(f"  → Calculating Event Statistics for county-level events...")
    event_stats_eaglei=calculate_event_stats_multi_county(ds, "eaglei")
    print(f"  → Calculating Event Statistics for spatiotemporal events...")
    event_stats_multi_county=calculate_event_stats_multi_county(ds, "mc_spatiotemporal")

    event_stats=pd.concat([event_stats_eaglei,event_stats_multi_county])
    event_file_dir = os.path.join(config.get("data_paths.event_data_dir"), "spatiotemporal_events")
    event_file_name = config.get("file_patterns.event_stats_file_pattern").format(start=start, end=end,
                                                                                       county=config.get(
                                                                                           "multi_county_analysis_parameters.label"))
    os.makedirs(event_file_dir, exist_ok=True)
    event_file_path = os.path.join(event_file_dir, event_file_name)
    event_stats.to_parquet(event_file_path)


