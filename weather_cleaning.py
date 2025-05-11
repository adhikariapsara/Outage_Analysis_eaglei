import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# Define the common 15-minute timestamp range
# =====================================
start = pd.Timestamp('2018-01-01 00:15:00')
end = pd.Timestamp('2023-12-30 23:45:00')
full_time_range = pd.date_range(start=start, end=end, freq="15min")
full_df = pd.DataFrame({"valid": full_time_range})

# =====================================
# Function to preprocess individual weather datasets
# =====================================
def preprocess_weather_data(file, column, replace_dict=None, create_occurrence=False, keep_max=True, drop_columns=None):
    df = pd.read_csv(file)
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

df_dew    = preprocess_weather_data('weather_data_NOAA/asos_Dew_Temp.csv',        'dwpf')
df_gust   = preprocess_weather_data('weather_data_NOAA/asos_gust.csv',            'gust_mph')
df_sped   = preprocess_weather_data('weather_data_NOAA/asos_windspeed.csv',       'sped')
df_temp   = preprocess_weather_data('weather_data_NOAA/asos_Temp.csv',            'tmpf')
df_pres   = preprocess_weather_data('weather_data_NOAA/asos_pressure.csv',        'mslp')
df_rh     = preprocess_weather_data('weather_data_NOAA/asos_RH.csv',              'relh')
df_wdir   = preprocess_weather_data('weather_data_NOAA/asos_wind_direction.csv',  'drct')
df_precip = preprocess_weather_data(
    file='weather_data_NOAA/asos_precipitation.csv',
    column='p01m',
    replace_dict={'T': 0.001, 'M': 0},
    create_occurrence=True,
    drop_columns=['station', 'lon', 'lat'],
    keep_max=True
)

# =====================================
# Merge all cleaned DataFrames on 'valid' timestamp
# =====================================
# Merge all DataFrames
combined_df = df_wdir.copy()
for df_component in [df_dew, df_pres, df_rh, df_temp, df_gust, df_sped, df_precip]:
    combined_df = combined_df.merge(df_component, on='valid', how='inner')

# Fill missing gust values with wind speed
combined_df['gust_mph'] = combined_df['gust_mph'].fillna(combined_df['sped'])

# Preview
print("Merged weather dataset preview (gust NaNs filled):")
print(combined_df.head())

combined_df.to_csv('cleaned_data/combined_data_weather.csv', index=False)