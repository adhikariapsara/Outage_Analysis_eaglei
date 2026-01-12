import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import math
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
    weather_data[weather_param].drop(0, inplace=True)
    weather_data[weather_param].fillna(1e-3, inplace=True)
    y, x, _ = plt.hist(weather_data[weather_param], bins=np.linspace(0,100,100), color='skyblue', edgecolor='black',
                       label='Histogram')
    # n is y-value of normalized distribution (frequency), bins is x value (wind speed)
    y = np.append(y, 0)
    plt.close()

    bin_width = np.mean(np.diff(x))
    y_pdf = y / (np.sum(y) * bin_width)
    if double_peak:
        # tweak guess values as needed
        params, _ = curve_fit(bimodal, x, y_pdf, p0=(60,10,0.25,90,10,0.25))
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = bimodal(x_fit, *params)
    else:
        params, _ = curve_fit(lognormal_pdf, x, y_pdf, p0=[1, np.mean(x)])
        x_fit = np.linspace(min(x), max(x), 100)
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

def plot_multiple_bayesian_fits(target_variable):

    fit_WA=pd.read_csv(f'exp_fit/Washington_King_Customer Out(%) _{target_variable}_fit.csv')
    fit_MA=pd.read_csv(f'exp_fit/Massachusetts_Suffolk_Customer Out(%) _{target_variable}_fit.csv')
    fit_CA=pd.read_csv(f'exp_fit/California_Los Angeles_Customer Out(%) _{target_variable}_fit.csv')
    fit_FL=pd.read_csv(f'exp_fit/Florida_Miami-Dade_Customer Out(%) _{target_variable}_fit.csv')
    fit_AZ=pd.read_csv(f'exp_fit/Arizona_Maricopa_Customer Out(%) _{target_variable}_fit.csv')
    fit_TX=pd.read_csv(f'exp_fit/Texas_Harris_Customer Out(%) _{target_variable}_fit.csv')
    fit_IL=pd.read_csv(f'exp_fit/Illinois_Cook_Customer Out(%) _{target_variable}_fit.csv')

    plt.plot(fit_WA[target_variable],fit_WA['y_avg'],label='King County WA')
    plt.plot(fit_MA[target_variable], fit_MA['y_avg'], label='Suffolk County MA')
    plt.plot(fit_CA[target_variable], fit_CA['y_avg'], label='LA County CA')
    plt.plot(fit_FL[target_variable], fit_FL['y_avg'], label='Miami-Dade County FL')
    plt.plot(fit_AZ[target_variable], fit_AZ['y_avg'], label='Maricopa County AZ')
    plt.plot(fit_TX[target_variable], fit_TX['y_avg'], label='Harris County TX')
    plt.plot(fit_IL[target_variable], fit_IL['y_avg'], label='Cook County IL')
    plt.xlabel('Max Wind Gust (MPH)')
    plt.ylabel('Pct Customers Out')
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


def plot_event_profile(data, wind_prob_map, location, zone_id, var_y, cvar):
    fig, ax = plt.subplots()
    ax.plot(wind_prob_map[0], wind_prob_map[1])
    # make wind speed labels = actual wind speed using some kind of interpolation
    wind_speed_labels = list([20,40,60,80])
    x_labels = list()
    for i in wind_speed_labels:
        # interpolate the numbers
        y = data.loc[i, 'Loss (KWh)']
        x_labels.append(y)
    #x_labels[0] = 0
    x_labels = ['%.2f' % elem for elem in x_labels]
    # plt.xticks(wind_speed_labels, labels=x_labels)
    sec = ax.secondary_xaxis(location=1)
    sec.set_xlabel('Load Loss (KWh)')
    sec.set_xticks(wind_speed_labels, labels=x_labels)
    # for i, label in enumerate(wind_speed_labels):
    #     plt.text(wind_prob_map[0,i], wind_prob_map[1,i] - 5, label, ha='center', va='top', fontsize=8, color='gray')
    plt.xlabel('Wind Speed (MPH)')
    plt.ylabel('Probability')
    plt.axvline(location, color='red')
    plt.figtext(s=f" VAR = {var_y} KWh", x=0.4, y=0.8)
    plt.figtext(s=f" CVAR = {CVAR} KWh", x=0.4, y=0.75)
    plt.fill_between(x=wind_prob_map[0], y1=wind_prob_map[1], where=(wind_prob_map[0] >= location), color="b",
                     alpha=0.2)
    # plt.title(f'Load Loss Distribution under Extreme Events Profile')
    plt.savefig(f'plot/VAR_FEEDER_{zone_id}')
    plt.close()

def calculate_VAR_CVAR(state, county, target_variable):

    # download weather profile
    weather_profile=pd.read_csv(f'weather_profiles/{state}_{county}_{target_variable}_profile.csv',
                                names=[target_variable,'probability'])
    # download Bayesian Fit
    bayesian_results=pd.read_csv(f'exp_fit/{state}_{county}_Customer Out(%) _max_{target_variable}_fit.csv')

    params=[]
    with open(f'weather_profiles/{state}_{county}_{target_variable}_profile_params.txt', 'r') as file:
        for line in file:
            params.extend([float(num) for num in line.split()])

    shape,scale=params
    loc=0
    location = lognorm.ppf(0.95, shape, loc=loc, scale=scale)

    print("95th percentile Wind Speed:", location)
    # interpolate for var
    x1 = math.floor(location)
    x2 = math.ceil(location)
    y1 = (bayesian_results.loc[x1, 'Percent Customer Out'])
    y2 = (bayesian_results.loc[x2, 'Percent Customer Out'])
    VAR = y1 + (y2 - y1) / (x2 - x1) * (location - x1)
    VAR = math.ceil(VAR * 100) / 100
    wind_speed_labels = list(wind_prob_map[0])
    x_labels = list()
    for i in wind_speed_labels:
        # interpolate the numbers
        if i>80:
            val=data.loc[80,'Loss (KWh)']
        else:
            x1 = math.floor(i)
            x2 = math.ceil(i)
            y1 = data.loc[x1, 'Loss (KWh)']
            y2 = data.loc[x2, 'Loss (KWh)']
            val = y1 + (y2 - y1) / (x2 - x1) * (i - x1)
            if math.isnan(val):
                val=0
        x_labels.append(val)
    x=np.array(x_labels)
    y = np.array(wind_prob_map[1])
    mask = x >= VAR
    CVAR=(x[mask]*y[mask]).sum()*(1/(1-0.95))
    CVAR = math.ceil(CVAR * 100) / 100
    return VAR, location, CVAR

# inputs - change as needed
state='Massachusetts'
county='Suffolk'
start='2018'
end='2024'
approach='percentile'
color='green'
value=0.75
target_variable="gust"
# read in data
calculate_VAR_CVAR(state, county, target_variable)
weather_data=pd.read_parquet(f'../weather_data/{state}/cleaned_weather_data_{county}.parquet')
# weather_outage=pd.read_parquet(f'../Results/Data_All_{county}_{approach}_{value}_{start}-{end}.parquet')
# # outage_events=(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.parquet')
#
# #weather_outage = weather_outage[weather_outage['max_gust']<70]
# #plot_multiple_bayesian_fits(target_variable)
# fit_outage_probability(weather_outage,target_variable,state,county,color)
# #plot_multiple_wind_profiles(target_variable)
#plot_weather_distribution(weather_data, target_variable, county, state, False)

