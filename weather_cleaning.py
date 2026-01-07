import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points

# =====================================
# Define the common 15-minute timestamp range
# =====================================
start = pd.Timestamp('2018-01-01 00:15:00')
end = pd.Timestamp('2024-12-30 23:45:00')
full_time_range = pd.date_range(start=start, end=end, freq="15min")
full_df = pd.DataFrame({"valid": full_time_range})

# =====================================
# Function to preprocess individual weather datasets
# =====================================
def preprocess_weather_data(df, column, replace_dict=None, create_occurrence=False, keep_max=True, drop_columns=None):
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

# =====================================
# Preprocess each weather component
# =====================================
def main(state, county, start, end):

    file=f'weather_data/{state}/weather_{state}_{county}_{start}_{end}.parquet'
    # Event Datasets
    weather_data=pd.read_parquet(file)
    columns_to_keep = ['station', 'valid', 'tmpf', 'sknt', 'p01i', 'gust', 'lon', 'lat']
    weather_data = weather_data[columns_to_keep]
    print("Cleaning weather data.")
    #df_dew    = preprocess_weather_data(file,        'dwpf')
    df_gust   = preprocess_weather_data(weather_data,            'gust')
    df_sped   = preprocess_weather_data(weather_data,       'sknt')
    df_temp   = preprocess_weather_data(weather_data,            'tmpf')
    #df_pres   = preprocess_weather_data('weather_data_NOAA/asos_pressure.csv',        'mslp')
    #df_rh     = preprocess_weather_data('weather_data_NOAA/asos_RH.csv',              'relh')
    #df_wdir   = preprocess_weather_data('weather_data_NOAA/asos_wind_direction.csv',  'drct')
    df_precip = preprocess_weather_data(
        df=weather_data,
        column='p01i',
        replace_dict={'T': 0.001, 'M': 0},
        create_occurrence=True,
        drop_columns=['station', 'lon', 'lat'],
        keep_max=True
    )

    # =====================================
    # Merge all cleaned DataFrames on 'valid' timestamp
    # =====================================
    # Merge all DataFrames
    weather_dataset=pd.concat([df_temp['valid'],df_gust['gust'],df_sped['sknt'],df_precip['p01i'],df_temp['tmpf']],axis=1)
    # combined_df = df_temp.copy()
    # for df_component in [df_temp, df_gust, df_sped, df_precip]:
    #     combined_df = combined_df.merge(df_component, on='valid', how='inner')

    # Fill missing gust values with wind speed
    weather_dataset['gust'] = weather_dataset['gust'].fillna(weather_dataset['sknt'])
    print("Weather data has been successfully cleaned. Saving to parquet.")
    # Preview
    #print(weather_dataset.head())
    weather_dataset.to_parquet(f'weather_data/{state}/cleaned_weather_data_{county}.parquet')


#use case example
# state='Washington'
# county='King'
# start=2018
# end=2024
# main(state, county, start, end)