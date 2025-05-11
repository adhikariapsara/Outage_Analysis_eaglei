import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load DataSets; Outage and Weather DataSets
df_outage_data = pd.read_excel('cleaned_data\Merged_ZOH_Cleaned_data_2018_2023_Cook.xlsx')
df_weather_data = pd.read_csv('cleaned_data\combined_data_weather.csv')

# Change the datetime
df_weather_data['DATE'] = pd.to_datetime(df_weather_data['valid'])
df_outage_data['run_start_time'] = pd.to_datetime(df_outage_data['run_start_time'])
df_outage_data = df_outage_data[(df_outage_data['run_start_time'] >= '2018-01-01') & (df_outage_data['run_start_time'] < '2023-01-01')]#Training_Data
#df_outage_data = df_outage_data[df_outage_data['run_start_time'] >= '2023-01-01']# Validation_data

df_weather_data = df_weather_data[(df_weather_data['DATE'] >= '2018-01-01') & (df_weather_data['DATE'] < '2023-01-01')]#Training_Data
#df_weather_data = df_weather_data[df_weather_data['DATE'] >= '2023-01-01'] #Validation_Data
df_outage_data.set_index('run_start_time', inplace = True )
# Set datetime as index for weather data
df_weather_data.set_index('DATE', inplace=True)
print(df_outage_data.head())
# Convert 'ws' and 'T' columns to float
df_weather_data['sped'] = pd.to_numeric(df_weather_data['sped'], errors='coerce').astype(float)
#df_weather_data['T'] = pd.to_numeric(df_weather_data['T'], errors='coerce').astype(float)
df_weather_data['gust_mph']=pd.to_numeric(df_weather_data['gust_mph'], errors='coerce').astype(float)
df_weather_data['drct'] = pd.to_numeric(df_weather_data['drct'], errors='coerce').astype(float)
df_weather_data['dwpf'] = pd.to_numeric(df_weather_data['dwpf'],errors = 'coerce').astype(float)
df_weather_data['tmpf'] = pd.to_numeric(df_weather_data['tmpf'],errors = 'coerce').astype(float)
df_weather_data['mslp'] = pd.to_numeric(df_weather_data['mslp'],errors = 'coerce').astype(float)
df_weather_data['relh'] = pd.to_numeric(df_weather_data['relh'],errors = 'coerce').astype(float)
df_weather_data['p01m'] = pd.to_numeric(df_weather_data['p01m'],errors = 'coerce').astype(float)
df_weather_data['poccurence'] = pd.to_numeric(df_weather_data['poccurence'],errors = 'coerce').astype(float)
print(df_weather_data.head())
# Load MCC Data
pdf = pd.read_csv('MCC.csv')
target_fips = '17031'  # For Cook County
result = pdf[pdf['County_FIPS'] == target_fips]
costumers = result['Customers'].values
print(costumers)

df_outage_data['sum'] = df_outage_data['sum'].apply(lambda x: 0 if x < 5000 else x)  # Apply threshold
df_outage_data['N_sum'] = df_outage_data['sum'] / costumers  # Normalize

# Find zero index dates in the outage data

df_outage_data = df_outage_data.groupby(df_outage_data.index).max()
zero_indices = df_outage_data.index[df_outage_data['sum'] == 0].tolist()
#print(zero_indices)
# Loop through each consecutive pair of zero indices
# Initialize list for no_outage events
no_outage_list = []

# Loop through each zero index and extract corresponding weather data
for zero_index in zero_indices:
    zero_index_weather = pd.to_datetime(zero_index)

    # Extract weather data for the zero outage period
    max_ws = df_weather_data['sped'].loc[zero_index_weather:zero_index_weather].max()
    max_g = df_weather_data['gust_mph'].loc[zero_index_weather:zero_index_weather].max()
    pp_max = df_weather_data['p01m'].loc[zero_index_weather:zero_index_weather].max()
    pocc = df_weather_data['poccurence'].loc[zero_index_weather:zero_index_weather].max()
    max_dirct = df_weather_data['drct'].loc[zero_index_weather:zero_index_weather].max()
    max_dwpf = df_weather_data['dwpf'].loc[zero_index_weather:zero_index_weather].max()
    max_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].max()
    max_mslp = df_weather_data['mslp'].loc[zero_index_weather:zero_index_weather].max()
    max_relh = df_weather_data['relh'].loc[zero_index_weather:zero_index_weather].max()
    


    # Append to no_outage_list
    no_outage_list.append({
        'cust_out_max': 0,  # No customers affected
        'out_duration_max': 0,
        'area_cost_out_h': 0,
        'area_KW_h': 0,
        'wind_speed': max_ws,
        'impact_time': 0,
        'outage_slope': 0,
        'recovery_duration': 0,
        'recovery_slope': 0,
        'cust_normalized': 0,
        'gust': max_g,
        'precipitation': pp_max,
        'ppocc': pocc,
        'wind_direction':max_dirct,
        'dew_point_temp': max_dwpf,
        'Air_temp': max_tmpf,
        'Pressure': max_mslp,
        'RH':max_relh,
        'cummulative_customer_out' : 0
        

    })

# Convert no_outage_list to DataFrame
no_outage_df = pd.DataFrame(no_outage_list)

print(no_outage_df.head())

#no_outage_df = no_outage_df.drop(['run_start_time', 'sum','ws','N_sum'], axis =1)
event_data_list = []
for i in range(len(zero_indices) - 1):
    first_zero_index = zero_indices[i]
    second_zero_index = zero_indices[i + 1]

    # Ensure the indices in both dataframes match before slicing
    first_zero_index_weather = pd.to_datetime(first_zero_index)
    second_zero_index_weather = pd.to_datetime(second_zero_index)

    # Slicing the outage data for the current event
    sliced_df = df_outage_data.loc[first_zero_index:second_zero_index].reset_index(drop=False)
    sliced_df.index = range(1, len(sliced_df) + 1)
    #print(sliced_df.head())
    # Check if there are any non-zero values to process
    if (sliced_df['sum'].values > 0).any():
        sliced_df['run_start_time'] = pd.to_datetime(sliced_df['run_start_time'])
        sliced_df['time_hours'] = (sliced_df['run_start_time'] - sliced_df['run_start_time'].iloc[0]).dt.total_seconds() / 3600
        sliced_df['KW_out'] = (
            sliced_df['N_sum'] * 0.34 * 4.19 +
            sliced_df['N_sum'] * 0.35 * 23.91 +
            sliced_df['N_sum'] * 0.31 * 1301.0041
        )
        sliced_df['change'] = sliced_df['sum'].diff()
        sliced_df['cummulative_customer_out'] = sliced_df.loc[sliced_df['change'] > 0, 'sum'].sum()

       
        
        ''''
        plt.figure(figsize=(12, 6))
        plt.plot(sliced_df['time_hours'], sliced_df['sum'], label='Customer Outages', marker='o')
        plt.plot(sliced_df['time_hours'], sliced_df['N_sum'], label='Normalized Customer Outages', marker='x')
        plt.plot(sliced_df['time_hours'], sliced_df['KW_out'], label='Estimated KW Outage', linestyle='--')

        plt.xlabel('Time since event start (hours)')
        plt.ylabel('Outage Metrics')
        plt.title('Outage Event Time Series')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
       '''

        # Check for the max index safely
        if sliced_df['sum'].max() > 0:  # Ensure there is at least one non-zero value
            max_index = sliced_df['sum'].idxmax()
            impact_time = sliced_df['time_hours'].iloc[max_index] - sliced_df['time_hours'].iloc[0]
            outage_slope = (sliced_df['N_sum'].iloc[max_index] - sliced_df['N_sum'].iloc[0]) / impact_time
            recovery_duration = sliced_df['time_hours'].iloc[-1] - sliced_df['time_hours'].iloc[max_index]
            recovery_slope = sliced_df['N_sum'].iloc[max_index] / recovery_duration
            area_KW_h = np.trapezoid(sliced_df['KW_out'], sliced_df['time_hours'])
            cust_out_max = sliced_df['sum'].max()
            out_duration_max = sliced_df['time_hours'].max()
            area_cost_out = np.trapezoid(sliced_df['N_sum'], sliced_df['time_hours'])
            cust_normalized = sliced_df['N_sum'].max()
            cummulative_customer_out_max = sliced_df['cummulative_customer_out'].max()
            

            # Now use loc for slicing weather data for the current event
            max_ws = df_weather_data['sped'].loc[first_zero_index_weather:second_zero_index_weather].max()
            #max_T = df_weather_data['T'].loc[first_zero_index_weather:second_zero_index_weather].max()
            #min_T = df_weather_data['T'].loc[first_zero_index_weather:second_zero_index_weather].min()
            max_g = df_weather_data['gust_mph'].loc[first_zero_index_weather:second_zero_index_weather].max()
            pp_max = df_weather_data['p01m'].loc[first_zero_index_weather:second_zero_index_weather].max()
            pocc = df_weather_data['poccurence'].loc[first_zero_index_weather:second_zero_index_weather].max()
            max_dirct = df_weather_data['drct'].loc[first_zero_index_weather:second_zero_index_weather].max()
            max_dwpf = df_weather_data['dwpf'].loc[first_zero_index_weather:second_zero_index_weather].max()
            max_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].max()
            max_mslp = df_weather_data['mslp'].loc[first_zero_index_weather:second_zero_index_weather].max()
            max_relh = df_weather_data['relh'].loc[first_zero_index_weather:second_zero_index_weather].max()
            

            event_data = {
                'cust_out_max': cust_out_max,
                'out_duration_max': out_duration_max,
                'area_cost_out_h': area_cost_out,
                'area_KW_h': area_KW_h,
                'wind_speed': max_ws,
                'impact_time': impact_time,
                'outage_slope': outage_slope,
                'recovery_duration': recovery_duration,
                'recovery_slope': recovery_slope,
                'cust_normalized': cust_normalized,
                #'Maximum_Temperature(degree)': max_T,
                #'Minimum_Temperature(degree)': min_T,
                'gust':max_g,
                'precipitation':pp_max,
                'ppocc':pocc,
                'wind_direction':max_dirct,
                'dew_point_temp': max_dwpf,
                'Air_temp': max_tmpf,
                'Pressure': max_mslp,
                'RH':max_relh,
                'cummulative_customer_out' : cummulative_customer_out_max
            

            }

            event_data_list.append(event_data)

# Convert event data list to DataFrame and save
event_df = pd.DataFrame(event_data_list)
event_df.to_csv('Results/Outage_Events_Summary_All_Cook_gust_Modified_SH_all_5000_2018-2022_all_weather.csv', index=False)
print(event_df.head())
print(no_outage_df.head())
analysis_df = pd.concat([event_df, no_outage_df], ignore_index= True)
print(analysis_df.head())
analysis_df.to_csv('Results/Visulaization_of_Data_Cook_All_2018-2022_5000_all_weather.csv',index = False)