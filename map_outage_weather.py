import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(state, county, start, end, value, approach):
    # Load MCC Data
    print("Aligning outage and weather data.")
    pdf = pd.read_csv('Eagle-idatasets/MCC.csv')

    # find number of customers in county
    county_to_fips=pd.read_csv('Eagle-idatasets/county_fips_master.csv', encoding='latin')
    ans=county_to_fips[county_to_fips['county_name']==f'{county} County']
    ans=ans[ans['state_name']==state]
    target_fips=ans['fips'].values[0]
    pdf['County_FIPS']=pd.to_numeric(pdf['County_FIPS'], downcast='integer',errors='coerce')
    result = pdf[pdf['County_FIPS'] == target_fips]
    customers = result['Customers'].values[0]
    #print(customers)

    # Load DataSets; Outage and Weather DataSets
    df_outage_data = pd.read_parquet(f'outage_data//{state}//{county}//Merged_Cleaned_data_{start}_{end}_{county}_{state}.parquet')
    df_weather_data = pd.read_parquet(f'weather_data/{state}/cleaned_weather_data_{county}.parquet')

    if approach=='percentile':
        event_detector=df_outage_data['sum'].quantile(value)
        print(f'Using percentile based event detection: Events defined '
              f'as those greater than {event_detector} customers.')
    elif approach=='percent_customers':
        event_detector=round(value*int(customers))
        print(f'Defining events as a percent of customers in the county out: Events defined '
              f'as those greater than {event_detector} customers.')
    elif approach=='flat':
        event_detector=value
        print(f'Defining events as a flat value: Events defined '
              f'as those greater than {event_detector} customers.')
    else:
        print('Invalid event detection method. Using a flat 100 customers as event definition.')
        event_detector=100

    # Change the datetime
    df_weather_data['DATE'] = pd.to_datetime(df_weather_data['valid'])
    # drop first row that weather does not interpolate
    df_outage_data=df_outage_data[(df_outage_data['run_start_time']!=f'{start}-01-01 00:00:00')]
    df_outage_data['run_start_time'] = pd.to_datetime(df_outage_data['run_start_time'])
    df_outage_data = df_outage_data[(df_outage_data['run_start_time'] >= f'{start}-01-01') & (df_outage_data['run_start_time'] < f'{end}-01-01')]#Training_Data
    #df_outage_data = df_outage_data[df_outage_data['run_start_time'] >= '2023-01-01']# Validation_data

    df_weather_data = df_weather_data[(df_weather_data['DATE'] >= f'{start}-01-01') & (df_weather_data['DATE'] < f'{end}-01-01')]#Training_Data
    #df_weather_data = df_weather_data[df_weather_data['DATE'] >= '2023-01-01'] #Validation_Data
    df_outage_data.set_index('run_start_time', inplace = True, drop=False)
    # Set datetime as index for weather data
    df_weather_data.set_index('DATE', inplace=True, drop=False)
    #print(df_outage_data.head())
    # Convert 'ws' and 'T' columns to float
    df_weather_data['sknt'] = pd.to_numeric(df_weather_data['sknt'], errors='coerce').astype(float)
    #df_weather_data['T'] = pd.to_numeric(df_weather_data['T'], errors='coerce').astype(float)
    df_weather_data['gust']=pd.to_numeric(df_weather_data['gust'], errors='coerce').astype(float)
    #df_weather_data['drct'] = pd.to_numeric(df_weather_data['drct'], errors='coerce').astype(float)
    #df_weather_data['dwpf'] = pd.to_numeric(df_weather_data['dwpf'],errors = 'coerce').astype(float)
    df_weather_data['tmpf'] = pd.to_numeric(df_weather_data['tmpf'],errors = 'coerce').astype(float)
    #df_weather_data['mslp'] = pd.to_numeric(df_weather_data['mslp'],errors = 'coerce').astype(float)
    #df_weather_data['relh'] = pd.to_numeric(df_weather_data['relh'],errors = 'coerce').astype(float)
    df_weather_data['p01i'] = pd.to_numeric(df_weather_data['p01i'],errors = 'coerce').astype(float)
    #df_weather_data['poccurence'] = pd.to_numeric(df_weather_data['poccurence'],errors = 'coerce').astype(float)
    #print(df_weather_data.head())

    df_outage_data['sum'] = df_outage_data['sum'].apply(lambda x: 0 if x < event_detector else x)  # Apply threshold
    df_outage_data['N_sum'] = df_outage_data['sum'] / customers  # Normalize

    # Find zero index dates in the outage data

    df_outage_data = df_outage_data.groupby(df_outage_data.index).max()
    zero_indices = df_outage_data.index[df_outage_data['sum'] == 0].tolist()
    #print(zero_indices)
    # Loop through each consecutive pair of zero indices
    # Initialize list for no_outage events
    no_outage_list = []

    print("Finding timestamps with no outage events based on threshold.")

    # Loop through each zero index and extract corresponding weather data
    for zero_index in zero_indices:
        zero_index_weather = pd.to_datetime(zero_index)

        # Extract weather data for the zero outage period
        max_ws = df_weather_data['sknt'].loc[zero_index_weather:zero_index_weather].max()
        avg_ws = df_weather_data['sknt'].loc[zero_index_weather:zero_index_weather].mean()

        max_g = df_weather_data['gust'].loc[zero_index_weather:zero_index_weather].max()
        avg_g = df_weather_data['gust'].loc[zero_index_weather:zero_index_weather].mean()

        pp_sum = df_weather_data['p01i'].loc[zero_index_weather:zero_index_weather].sum()

        max_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].max()
        min_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].min()
        avg_tmpf = df_weather_data['tmpf'].loc[zero_index_weather:zero_index_weather].mean()

        year=df_weather_data['DATE'].loc[zero_index_weather].year
        month=df_weather_data['DATE'].loc[zero_index_weather].month
        day=df_weather_data['DATE'].loc[zero_index_weather].day


        # Append to no_outage_list
        no_outage_list.append({
            'cust_out_max': 0,  # No customers affected
            'out_duration_max': 0,
            'area_cost_out_h': 0,
            'area_KW_h': 0,
            'max_wind_speed': max_ws,
            'avg_wind_speed': avg_ws,
            'impact_time': 0,
            'outage_slope': 0,
            'recovery_duration': 0,
            'recovery_slope': 0,
            'cust_normalized': 0,
            'max_gust': max_g,
            'avg_gust': avg_g,
            'precipitation': pp_sum,
            'Air_temp_max': max_tmpf,
            'Air_temp_min': min_tmpf,
            'Air_temp_avg': avg_tmpf,
            'start_datetime': df_weather_data['DATE'].loc[zero_index_weather],
            'end_datetime': df_weather_data['DATE'].loc[zero_index_weather],
            'cummulative_customer_out' : 0


        })

    # Convert no_outage_list to DataFrame
    no_outage_df = pd.DataFrame(no_outage_list)

    #print(no_outage_df.head())
    print("Finding outage events based on threshold.")

    #no_outage_df = no_outage_df.drop(['run_start_time', 'sum','ws','N_sum'], axis =1)
    event_data_list = []
    for i in range(len(zero_indices) - 1):
        first_zero_index = zero_indices[i]
        second_zero_index = zero_indices[i + 1]

        # Ensure the indices in both dataframes match before slicing
        first_zero_index_weather = pd.to_datetime(first_zero_index)
        second_zero_index_weather = pd.to_datetime(second_zero_index)

        # Slicing the outage data for the current event
        sliced_df = df_outage_data.loc[first_zero_index:second_zero_index].copy()
        sliced_df.reset_index(drop=True, inplace=True)
        sliced_df.index += 1  # Start index from 1
        # sliced_df.index = range(1, len(sliced_df) + 1)
        #print(sliced_df.head())
        
        # Check if there are any non-zero values to process
        # if (sliced_df['sum'].values > 0).any():
        if (sliced_df['sum'] > 0).any():
            sliced_df.loc[:,'run_start_time'] = pd.to_datetime(sliced_df['run_start_time'])
            sliced_df.loc[:,'time_hours'] = (sliced_df['run_start_time'] - sliced_df['run_start_time'].iloc[0]).dt.total_seconds() / 3600
            sliced_df.loc[:,'KW_out'] = (
                sliced_df['N_sum'] * 0.34 * 4.19 +
                sliced_df['N_sum'] * 0.35 * 23.91 +
                sliced_df['N_sum'] * 0.31 * 1301.0041
            )
            sliced_df.loc[:,'change'] = sliced_df['sum'].diff()
            sliced_df.loc[:,'cummulative_customer_out'] = sliced_df.loc[sliced_df['change'] > 0, 'sum'].sum()



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
                max_index = int(sliced_df['sum'].idxmax())
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
                max_ws = df_weather_data['sknt'].loc[first_zero_index_weather:second_zero_index_weather].max()
                avg_ws = df_weather_data['sknt'].loc[first_zero_index_weather:second_zero_index_weather].mean()

                max_g = df_weather_data['gust'].loc[first_zero_index_weather:second_zero_index_weather].max()
                avg_g = df_weather_data['gust'].loc[first_zero_index_weather:second_zero_index_weather].mean()

                pp_sum = df_weather_data['p01i'].loc[first_zero_index_weather:second_zero_index_weather].sum()

                max_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].max()
                min_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].min()
                avg_tmpf = df_weather_data['tmpf'].loc[first_zero_index_weather:second_zero_index_weather].mean()

                event_data = {
                    'cust_out_max': cust_out_max,
                    'out_duration_max': out_duration_max,
                    'area_cost_out_h': area_cost_out,
                    'area_KW_h': area_KW_h,
                    'max_wind_speed': max_ws,
                    'avg_wind_speed': avg_ws,
                    'impact_time': impact_time,
                    'outage_slope': outage_slope,
                    'recovery_duration': recovery_duration,
                    'recovery_slope': recovery_slope,
                    'cust_normalized': cust_normalized,
                    'max_gust':max_g,
                    'avg_gust':avg_g,
                    'precipitation':pp_sum,
                    'start_datetime': df_weather_data['DATE'].loc[first_zero_index_weather],
                    'end_datetime': df_weather_data['DATE'].loc[second_zero_index_weather],
                    'Air_temp_max': max_tmpf,
                    'Air_temp_min': min_tmpf,
                    'Air_temp_avg': avg_tmpf,
                    'cummulative_customer_out' : cummulative_customer_out_max


                }

                event_data_list.append(event_data)

    # Convert event data list to DataFrame and save
    print("Saving outage and non-outage events to parquet.")
    event_df = pd.DataFrame(event_data_list)
    event_df.to_parquet(f'Results/Outage_Events_Summary_All_{county}_{approach}_{value}_{start}-{end}.parquet', index=False)
    #print(event_df.head())
    #print(no_outage_df.head())
    analysis_df = pd.concat([event_df, no_outage_df], ignore_index= True)
    #print(analysis_df.head())
    analysis_df.to_parquet(f'Results/Data_All_{county}_{approach}_{value}_{start}-{end}.parquet',index = False)


# Example
# state='Rhode Island'
# county='Bristol'
# start=2018
# end=2019
# main(state, county, start, end, 0.001)
