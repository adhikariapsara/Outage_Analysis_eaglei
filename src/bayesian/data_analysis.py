import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import math
import xarray as xr

plt.style.use('classic')



def plot_max_temp(df):
    plt.hist(df["Air_temp"].dropna(), bins=50, color="royalblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Maximum Temp")
    plt.ylabel("Frequency")
    plt.title("Histogram of Maximum Temperature")
    plt.show()

def plot_seasonal_weather_behavior(df):
    plt.scatter(df['month'], df['gust'])
    plt.scatter(df['month'], df['precipitation'])
    plt.scatter(df['month'],df['Air_temp'])
    plt.legend(['Gust', 'Precipitation', 'Temperature'])
    plt.show()

def lognormal_pdf(x, shape, scale):
    return lognorm.pdf(x, shape, loc=0, scale=scale)

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def plot_weather_distribution(weather_data, weather_param, county, state, double_peak):
    #weather_data[weather_param].drop(0, inplace=True)
    y, x, _ = plt.hist(weather_data[weather_param], bins=np.linspace(0,60,61), color='skyblue', edgecolor='black',
                       label='Histogram')
    # n is y-value of normalized distribution (frequency), bins is x value (wind speed)
    y = np.append(y, 0)
    plt.close()

    bin_width = np.mean(np.diff(x))
    y_pdf = y / (np.sum(y) * bin_width)
    if double_peak:
        # tweak guess values as needed
        params, _ = curve_fit(bimodal, x, y_pdf, p0=(40,10,0.25,70,10,0.25))
        x_fit = np.linspace(min(x), max(x), 120)
        y_fit = bimodal(x_fit, *params)
    else:
        params, _ = curve_fit(lognormal_pdf, x, y_pdf, p0=[1, np.mean(x)])
        x_fit = np.linspace(min(x), max(x), 61)
        y_fit = lognormal_pdf(x_fit, *params)

    # Generate fitted curve
    area = simpson(y_fit, x_fit)
    print("Area under fitted PDF:", area)
    np.savetxt(f'weather_profiles/{state}_{county}_{weather_param}_profile_params.txt', params)
    plt.plot(x, y_pdf, 'bo', label='Raw Binned Data')
    plt.plot(x_fit, y_fit, 'r-', label='Fitted PDF')
    wind_profile = pd.DataFrame(columns=[x_fit, y_fit]).transpose()
    wind_profile.to_csv(f'weather_profiles/{state}_{county}_{weather_param}_profile.csv')
    plt.xlabel(weather_param)
    plt.ylabel('Probability of Occurrence')
    plt.title(f'{weather_param} Occurrences in Dataset for {county} County, {state}')
    plt.legend()

    plt.grid(True)
    plt.savefig(f'weather_profiles/{state}_{county}_{weather_param}_histogram.png')

def plot_multiple_wind_profiles(target_variable):
    # Load areas we are analyzing
    profile_WA = np.array(pd.read_csv(f'weather_profiles/Washington_King_{target_variable}_profile.csv').transpose())
    profile_MA = np.array(pd.read_csv(f'weather_profiles/Massachusetts_Suffolk_{target_variable}_profile.csv').transpose())
    profile_CA = np.array(pd.read_csv(f'weather_profiles/California_Los Angeles_{target_variable}_profile.csv').transpose())
    profile_FL = np.array(pd.read_csv(f'weather_profiles/Florida_Miami-Dade_{target_variable}_profile.csv').transpose())
    profile_AZ = np.array(pd.read_csv(f'weather_profiles/Arizona_Maricopa_{target_variable}_profile.csv').transpose())
    profile_TX = np.array(pd.read_csv(f'weather_profiles/Texas_Harris_{target_variable}_profile.csv').transpose())
    profile_IL = np.array(pd.read_csv(f'weather_profiles/Illinois_Cook_{target_variable}_profile.csv').transpose())

    plt.plot(profile_WA[0],profile_WA[1],label='King County WA')
    plt.plot(profile_MA[0],profile_MA[1],label='Suffolk County MA')
    plt.plot(profile_CA[0],profile_CA[1],label='Los Angeles County CA')
    plt.plot(profile_FL[0],profile_FL[1],label='Miami-Dade County FL')
    plt.plot(profile_AZ[0],profile_AZ[1],label='Maricopa County AZ')
    plt.plot(profile_TX[0],profile_TX[1],label='Harris County TX')
    plt.plot(profile_IL[0],profile_IL[1],label='Cook County IL')
    plt.xlabel('Temperature (degrees F)')
    plt.ylabel('Probability')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_bayesian_fits(target_variable,target_output):

    fit_WA=pd.read_csv(f'exp_fit/Washington_King_{target_output}_{target_variable}_fit.csv')
    fit_MA=pd.read_csv(f'exp_fit/Massachusetts_Suffolk_{target_output}_{target_variable}_fit.csv')
    fit_CA=pd.read_csv(f'exp_fit/California_Los Angeles_{target_output}_{target_variable}_fit.csv')
    fit_FL=pd.read_csv(f'exp_fit/Florida_Miami-Dade_{target_output}_{target_variable}_fit.csv')
    fit_AZ=pd.read_csv(f'exp_fit/Arizona_Maricopa_{target_output}_{target_variable}_fit.csv')
    fit_TX=pd.read_csv(f'exp_fit/Texas_Harris_{target_output}_{target_variable}_fit.csv')
    fit_IL=pd.read_csv(f'exp_fit/Illinois_Cook_{target_output}_{target_variable}_fit.csv')

    plt.plot(fit_WA[target_variable],fit_WA['y_avg'],label='King County WA')
    plt.plot(fit_MA[target_variable], fit_MA['y_avg'], label='Suffolk County MA')
    plt.plot(fit_CA[target_variable], fit_CA['y_avg'], label='LA County CA')
    plt.plot(fit_FL[target_variable], fit_FL['y_avg'], label='Miami-Dade County FL')
    plt.plot(fit_AZ[target_variable], fit_AZ['y_avg'], label='Maricopa County AZ')
    plt.plot(fit_TX[target_variable], fit_TX['y_avg'], label='Harris County TX')
    plt.plot(fit_IL[target_variable], fit_IL['y_avg'], label='Cook County IL')
    plt.xlabel('Max Wind Gust (MPH)')
    plt.ylabel('Customers Out')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    breakpoint()

def plot_multiple_outage_probabilities(target_variable):
    breakpoint()

def myround(x, base=1):
    return base * round(x/base)

def sigmoid(x, w0, k):
    y = (1 / (1.0 + np.exp(-1 * k * (x - w0))))
    # y = 1 / (k * np.sqrt(2 * 3.14) * np.exp(-1*0.5 * ((x - w0) / k) ** 2))
    return y

def fit_outage_probability(weather_outage,target_variable,state,county,color):


    weather_outage[target_variable] = (weather_outage[target_variable]).round()
    weather_outage = (weather_outage.groupby([target_variable], as_index=False)['outage'].mean())

    x=weather_outage[target_variable]
    y=weather_outage['outage']
    plt.scatter(x,y, label='Raw Binned Data',color=color)

    mask = y == 0
    y[mask] = 1e-15
    popt, pcov = curve_fit(sigmoid,x, y,p0=[80,0.1])
    plt.plot(weather_outage[target_variable], sigmoid(weather_outage[target_variable], *popt), 'r-', label="Fitted Curve",color=color)

    plt.xlabel(target_variable)
    plt.ylabel('Probability of Outage')
    plt.xlim(0,120)
    plt.ylim(0,1)
    plt.title(f'Outage Probability Curve for {county} County, {state}')
    plt.grid(True)

    w0 = popt[0]
    k = popt[1]
    x_fit = np.linspace(-20, 130, 150)
    y_fit = sigmoid(x_fit, w0, k)
    # Calculate errors
    y_pred=sigmoid(x,*popt)
    mse = np.mean(np.abs(y- y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs((y - y_pred)))
    print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

    plt.savefig(f'exp_fit/{state}_{county}_{target_variable}_outage_probability.png')
    np.savetxt(f'exp_fit/{state}_{county}_{target_variable}_outage_probability.csv', [x_fit, y_fit], delimiter=',')


def plot_event_profile(location, state,county, color, var_y,target_variable,target_output):
    # download weather profile
    weather_profile=pd.read_csv(f'weather_profiles/{state}_{county}_tmpf_profile.csv',
                                names=[target_variable,'probability'])
    # download Bayesian Fit
    bayesian_results=pd.read_csv(f'exp_fit/{state}_{county}_{target_output}_{target_variable}_max_fit.csv')


    fig, ax = plt.subplots()
    ax.plot(weather_profile[target_variable], weather_profile['probability'],color=color)
    # make wind speed labels = actual wind speed using some kind of interpolation
    wind_speed_labels = list([70,80,90])
    x_labels = list()
    bayesian_results = bayesian_results.sort_values(f"{target_variable}_max")

    for i in wind_speed_labels:
        # interpolate the numbers
        lower = bayesian_results[bayesian_results[f"{target_variable}_max"] <= i].iloc[-1]
        upper = bayesian_results[bayesian_results[f"{target_variable}_max"] >= i].iloc[0]
        x1, y1 = lower[f"{target_variable}_max"], lower['y_avg']
        x2, y2 = upper[f"{target_variable}_max"], upper['y_avg']
        y = y1 + (i - x1) * (y2 - y1) / (x2 - x1)
        if np.isnan(y):
            y=y1
        x_labels.append(y)
    x_labels = [round(x) for x in x_labels]
    sec = ax.secondary_xaxis(location=1)
    sec.set_xlabel('Customer Losses')
    sec.set_xticks(wind_speed_labels, labels=x_labels)

    plt.xlabel('Air Temperature')
    plt.ylabel('Probability')
    plt.axvline(location, color='gray')
    plt.figtext(s=f" VAR = {var_y} Customers", x=0.4, y=0.8)
    #plt.title(f'Event Profile Plot for {county} County, {state}')
    plt.savefig(f'var_plots/var_{state}_{county}_{target_variable}')
    plt.close()

def get_customers_in_county(county,state):
    # find number of customers in county
    pdf = pd.read_csv('../Eagle-idatasets/MCC.csv')
    county_to_fips=pd.read_csv('../Eagle-idatasets/county_fips_master.csv', encoding='latin')
    ans=county_to_fips[county_to_fips['county_name']==f'{county} County']
    ans=ans[ans['state_name']==state]
    target_fips=ans['fips'].values[0]
    pdf['County_FIPS']=pd.to_numeric(pdf['County_FIPS'], downcast='integer',errors='coerce')
    result = pdf[pdf['County_FIPS'] == target_fips]
    customers = result['Customers'].values[0]
    return customers

def calculate_VAR_CVAR(state, county, target_variable, target_output, type):

    # download weather profile
    weather_profile=pd.read_csv(f'weather_profiles/{state}_{county}_tmpf_profile.csv',
                                names=[target_variable,'probability'])
    # download Bayesian Fit
    bayesian_results=pd.read_csv(f'exp_fit/{state}_{county}_{target_output}_{target_variable}_max_fit.csv')

    params=[]
    with open(f'weather_profiles/{state}_{county}_tmpf_profile_params.txt', 'r') as file:
        for line in file:
            params.extend([float(num) for num in line.split()])

    # shape,scale=params
    # loc=0

    df_sorted = weather_profile.sort_values(target_variable)
    # df_sorted["probability"] /= df_sorted["probability"].sum()
    # df_sorted["cum_prob"] = df_sorted["probability"].cumsum()
    # Compute cumulative probability
    df_sorted["cum_prob"] = df_sorted["probability"].cumsum()
    #Find wind speed where cumulative probability crosses 95%
    location = df_sorted.loc[
        df_sorted["cum_prob"] >= 0.95, target_variable
    ].iloc[0]


    #location = lognorm.ppf(0.95, shape, loc=loc, scale=scale)

    print("95th percentile Weather:", location)
    # interpolate for var
    df = bayesian_results.sort_values(f"{target_variable}_max")

    lower = df[df[f"{target_variable}_max"] <= location].iloc[-1]
    upper = df[df[f"{target_variable}_max"] >= location].iloc[0]
    x1, y1 = lower[f"{target_variable}_max"], lower[type]
    x2, y2 = upper[f"{target_variable}_max"], upper[type]

    VAR = y1 + (location - x1) * (y2 - y1) / (x2 - x1)
    VAR = math.ceil(VAR * 100) / 100
    print(VAR)
    VAR_norm=VAR/get_customers_in_county(county,state)
    print(VAR_norm)
    return VAR, location

# inputs - change as needed
state='Washington'
county='Clallam'
start='2014'
end='2024'
# approach='percentile'
color='black'
value=0.75
weather_variable="gust"
# read in data
# VAR, location = calculate_VAR_CVAR(state, county, weather_variable,'cust_out_max','y_lower')
# plot_event_profile(location, state,county, color, VAR,weather_variable,'cust_out_max')
weather_data=xr.open_dataset(f'../../weather_data/{state}/cleaned_weather_{state}_{county}_{start}_{end}.nc').to_dataframe()
# weather_outage=pd.read_parquet(f'../Results/Data_All_{county}_{approach}_{value}_{start}-{end}.parquet')
# # outage_events=(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.parquet')
#
# #weather_outage = weather_outage[weather_outage['max_gust']<70]
#plot_multiple_bayesian_fits(target_variable,'cust_out_max')
# fit_outage_probability(weather_outage,target_variable,state,county,color)
#plot_multiple_wind_profiles(target_variable)
plot_weather_distribution(weather_data, weather_variable, county, state, False)

