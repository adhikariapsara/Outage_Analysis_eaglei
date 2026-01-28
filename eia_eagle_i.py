
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def plot_eaglei_eia(eagle_i_data,eia_data,start_date,end_date,year,i):
    fig, ax1 = plt.subplots()
    # # First plot: EIA
    values=[0,int(eia_data['Number of Customers Affected']),0]
    edges=[start_date-pd.to_timedelta(1,unit='h'),start_date,end_date,end_date+pd.to_timedelta(1,unit='h')]
    plt.stairs(values,edges,label='EIA Event',color='r')

    # # Second plot: EAGLE-I
    plt.step(eagle_i_data['run_start_time'],eagle_i_data['customers_out'],color='b', label='EAGLE_I Event')
    plt.xlabel('Date')
    plt.ylabel('Num Customers Affected')
    plt.legend()
    plt.title(f"Comparison of Datasets for Event in {eia_data['states']}, {eia_data['counties']}")
    plt.savefig(f'EIA_disturbances_data/EIA_eaglei_plots/{year}_event_{i}.png')
    plt.close()
import re

def extract_states_and_counties(text):
    if pd.isna(text):
        return [], []
    states = []
    counties = []

    # Split rows by semicolon
    parts = [p.strip() for p in text.split(";") if p.strip()]

    for part in parts:
        tokens = [t.strip() for t in part.split(":") if t.strip()]

        for token in tokens:
            if re.search(r"\b(county|parish)\b", token, flags=re.I):
                # Split multiple counties/parishes and remove word
                cleaned = [
                    re.sub(r"\s*\b(county|parish)\b\s*", "", c, flags=re.I).strip()
                    for c in token.split(",")
                    if re.search(r"\b(county|parish)\b", c, flags=re.I)
                ]
                counties.extend(cleaned)
            else:
                states.append(token)

    return states, counties

year = 2021 # compare EIA data with EAGLE-I 2023 data
eia_data=pd.read_csv(f'EIA_disturbances_data/{year}/us.csv')
eaglei_data=pd.read_csv(f'Eagle-idatasets/eaglei_outages_{year}.csv')
eaglei_data['run_start_time']=pd.to_datetime(eaglei_data['run_start_time'])

# Make new dataframe that contains EAGLE-I and EIA overlap comparisons
outage_overlap=pd.DataFrame()

# for eia data, lets only look at those listing customers out and containing start dates
eia_data = eia_data[~eia_data['Number of Customers Affected'].str.contains('Unknown|0', na=False)]
eia_data = eia_data[eia_data['Date Event Began'].notna() & (eia_data['Date Event Began'].str.strip() != '')].reset_index()

# separate states and counties into two separate columns
eia_data[["states", "counties"]] = eia_data["Area Affected"].apply(
    lambda x: pd.Series(extract_states_and_counties(x))
)
for i in range(len(eia_data)):
    # find beginning of events and their location
    start_date=eia_data.loc[i,'Date Event Began'] + " " + eia_data.loc[i,'Time Event Began']
    start_date=datetime.strptime(start_date, '%m/%d/%Y %H:%M:%S')
    print(eia_data.loc[i,'states'])
    print(eia_data.loc[i,'counties'])
    selected_eaglei=eaglei_data[eaglei_data['state'].isin(eia_data.loc[i,'states'])]
    if eia_data.loc[i,'counties']:
        selected_eaglei=selected_eaglei[selected_eaglei['county'].isin(eia_data.loc[i,'counties'])]
    # If we know the event's end date, we can filter eagle-i by that. Otherwise assume event is 2 days long
    if eia_data.loc[i,"Date of Restoration"] != "Unknown":
        end_date=datetime.strptime(eia_data.loc[i,"Date of Restoration"]+ " " +
                                   eia_data.loc[i,'Time of Restoration'],'%m/%d/%Y %H:%M:%S')
    else:
        end_date= start_date+pd.to_timedelta(2, unit='D')
    mask = ((selected_eaglei['run_start_time'] >= start_date-pd.to_timedelta(1,unit='h')) &
            (selected_eaglei['run_start_time'] <= end_date+pd.to_timedelta(1,unit='h')))
    selected_eaglei = selected_eaglei.loc[mask]
    # find total outage mag per time stamp
    total_eaglei=selected_eaglei.groupby("run_start_time", as_index=False)["customers_out"].sum()
    # Plot the outage events seen for both eagle-i and EIA
    plot_eaglei_eia(total_eaglei, eia_data.loc[i], start_date,end_date,year,i)
    # Now, add values to created dataframe for numerical comparison
    # outage_overlap.loc[i,'locations']=
    # outage_overlap.loc[i,'eia_outage_mag']=
    # outage_overlap.loc[i, 'eaglei_outage_mag']=
    # outage_overlap.loc[i, 'start_date']=
    # outage_overlap.loc[i, 'end_date']=
