import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
#plt.style.use('classic')



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
    y, x, _ = plt.hist(weather_data[weather_param], bins=np.linspace(20,120,60), color='skyblue', edgecolor='black',
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
    #np.savetxt(f'data/parsed/{state}_{county}_{weather_param}_profile_params.txt', params)
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

def myround(x, base=1):
    return base * round(x/base)

def sigmoid(x, w0, k):
    y = (1 / (1.0 + np.exp(-1 * k * (x - w0))))
    # y = 1 / (k * np.sqrt(2 * 3.14) * np.exp(-1*0.5 * ((x - w0) / k) ** 2))
    return y

def fit_outage_probability(weather_outage,target_variable):
    weather_outage['outage'] = 0
    for i in weather_outage.index:
        if weather_outage.loc[i, 'cust_out_max'] > 0:
            weather_outage.loc[i, 'outage'] = 1

    weather_outage[target_variable] = (weather_outage[target_variable]).round()
    weather_outage = (weather_outage.groupby([target_variable], as_index=False)['outage'].mean())

    x=weather_outage[target_variable]
    y=weather_outage['outage']
    plt.scatter(x,y, label='Raw Binned Data')

    mask = y == 0
    y[mask] = 1e-15
    popt, pcov = curve_fit(sigmoid,x, y,p0=[40,0.01])
    plt.plot(weather_outage[target_variable], sigmoid(weather_outage[target_variable], *popt), 'r-', label="Fitted Curve")

    plt.xlabel(target_variable)
    plt.ylabel('Probability of Outage')
    plt.grid(True)
    plt.show()

# inputs - change as needed
state='Washington'
county='King'
start='2018'
end='2024'
threshold=0.001
# read in data

#weather_data=pd.read_parquet(f'../weather_data/{state}/cleaned_weather_data_{county}.parquet')
weather_outage=pd.read_parquet(f'../Results/Data_All_{county}_{threshold}_{start}-{end}.parquet')
# outage_events=(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.parquet')
target_variable="max_gust"
fit_outage_probability(weather_outage,target_variable)
#plot_multiple_wind_profiles(target_variable)
#plot_weather_distribution(weather_data, target_variable, county, state, False)

