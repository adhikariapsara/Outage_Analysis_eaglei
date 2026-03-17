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
    counties = [
        ("Washington", "King", "King County WA", "blue"),
        ("Massachusetts", "Suffolk", "Suffolk County MA", "green"),
        ("California", "Los Angeles", "Los Angeles County CA", "red"),
        ("Florida", "Miami-Dade", "Miami-Dade County FL", "teal"),
        ("Arizona", "Maricopa", "Maricopa County AZ", "magenta"),
        ("Texas", "Harris", "Harris County TX", "olive"),
        ("Illinois", "Cook", "Cook County IL", "black"),
    ]

    for state, county, label, color in counties:
        path = f"weather_profiles/{state}_{county}_{target_variable}_profile.csv"

        profile = np.array(pd.read_csv(path).transpose())

        plt.plot(profile[0], profile[1], label=label, color=color)

    plt.xlabel("Wind Gust (MPH)")
    plt.ylabel("Probability")

    plt.legend(loc="center left", bbox_to_anchor=(0.5, 0.75))
    plt.grid(True)

    plt.tight_layout()

    plt.gcf().patch.set_facecolor("white")
    plt.gca().set_facecolor("white")

    plt.show()

def plot_multiple_bayesian_fits(target_variable,target_output):
    counties = [
        ("Washington", "King", "King County WA", "blue"),
        ("Massachusetts", "Suffolk", "Suffolk County MA", "green"),
        ("California", "Los Angeles", "LA County CA", "red"),
        ("Florida", "Miami-Dade", "Miami-Dade County FL", "teal"),
        ("Arizona", "Maricopa", "Maricopa County AZ", "magenta"),
        ("Texas", "Harris", "Harris County TX", "goldenrod"),
        ("Illinois", "Cook", "Cook County IL", "black"),
    ]
    fig, ax = plt.subplots(figsize=(12, 7))

    for state, county, label, color in counties:
        path = f"results/bayesian_single_{state}_{county}_max_{target_variable}_{target_output}.csv"
        fit = pd.read_csv(path)

        x = fit[f'max_{target_variable}']

        plt.plot(x, fit["y_avg"]/get_customers_in_county(county,state), label=label, color=color, zorder=3)
        # plt.fill_between(
        #     x,
        #     fit["y_upper"],
        #     fit["y_lower"],
        #     color=color,
        #     alpha=0.2,
        #     zorder=1
        # )

    plt.xlabel("Max Wind Gust (MPH)")
    plt.ylabel("Customer-Hours (Normalized)")
    plt.yscale("log")
    plt.xlim(0, 65)
   # plt.ylim(0,1)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=9
    )
    plt.subplots_adjust(right=0.78)
    plt.grid(True)

    plt.gcf().patch.set_facecolor("white")
    plt.gca().set_facecolor("white")

    plt.tight_layout()
    plt.show()

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


def plot_event_profile(location, state,county, color, var_y,CVAR, target_variable,target_output):
    # download weather profile
    weather_profile=pd.read_csv(f'weather_profiles/{state}_{county}_{target_variable}_profile.csv',
                                names=[target_variable,'probability'])
    # download Bayesian Fit
    bayesian_results=pd.read_csv(f"results/bayesian_single_{state}_{county}_max_{target_variable}_{target_output}.csv")


    fig, ax = plt.subplots()
    ax.plot(weather_profile[target_variable], weather_profile['probability'],color=color)
    # make wind speed labels = actual wind speed using some kind of interpolation
    wind_speed_labels = list([0,10,20,30,40,50,60])
    x_labels = list()
    bayesian_results = bayesian_results.sort_values(f"max_{target_variable}")

    for i in wind_speed_labels:
        # interpolate the numbers
        lower = bayesian_results[bayesian_results[f"max_{target_variable}"] <= i].iloc[-1]
        upper = bayesian_results[bayesian_results[f"max_{target_variable}"] > i].iloc[0]
        x1, y1 = lower[f"max_{target_variable}"], lower['y_avg']
        x2, y2 = upper[f"max_{target_variable}"], upper['y_avg']
        y = y1 + (i - x1) * (y2 - y1) / (x2 - x1)
        if np.isnan(y):
            y=y1
        x_labels.append(y)
    x_labels = [round(x) for x in x_labels]
    sec = ax.secondary_xaxis(location=1)
    sec.set_xlabel('Customer-Hour Losses (Normalized)')
    sec.set_xticks(wind_speed_labels, labels=x_labels)

    plt.xlabel('Wind Gust')
    plt.ylabel('Probability')
    plt.axvline(location, color='gray')
    plt.figtext(s=f" VAR = {var_y} Customer-Hours", x=0.5, y=0.8)
    plt.figtext(s=f" CVAR = {CVAR} Customer-Hours", x=0.5, y=0.75)
    #plt.title(f'Event Profile Plot for {county} County, {state}')
    plt.fill_between(x=weather_profile[target_variable], y1=weather_profile['probability'],
                     where=(weather_profile[target_variable] >= location), color=color,
                     alpha=0.2)
    plt.savefig(f'var_plots/var_{state}_{county}_{target_variable}')
    plt.close()

def get_customers_in_county(county,state):
    # find number of customers in county
    pdf = pd.read_csv('../../Eagle-idatasets/MCC.csv')
    county_to_fips=pd.read_csv('../../Eagle-idatasets/county_fips_master.csv', encoding='latin')
    ans=county_to_fips[county_to_fips['county_name']==f'{county} County']
    ans=ans[ans['state_name']==state]
    target_fips=ans['fips'].values[0]
    pdf['County_FIPS']=pd.to_numeric(pdf['County_FIPS'], downcast='integer',errors='coerce')
    result = pdf[pdf['County_FIPS'] == target_fips]
    customers = result['Customers'].values[0]
    return customers

def calculate_VAR_CVAR(state, county, W, L, column_name):

    # Download Probabilistic Weather Profile
    p_W=pd.read_csv(f'weather_profiles/{state}_{county}_{W}_profile.csv',
                                names=[W,'probability'])

    # Download Outage Distribution Given Event Occurrence
    G_r_w=pd.read_csv(f'results/bayesian_single_{state}_{county}_max_{W}_{L}.csv')

    # Download Event Probability
    prob_event_W=pd.read_csv(f'results/bayesian_probability_{state}_{county}_{W}_outage.csv')

    num_customers = get_customers_in_county(county, state)
    alpha=0.95

    F_r_w=pd.DataFrame([G_r_w[f'max_{W}'],(1-prob_event_W[column_name])+G_r_w[column_name]*prob_event_W[column_name]]).transpose()

    # Compute cumulative probability
    p_W["cum_prob"] = p_W["probability"].cumsum()
    # Find wind speed where cumulative probability crosses 95%
    w_alpha = p_W.loc[
        p_W["cum_prob"] >= alpha, W
    ].iloc[0]
    print("Weather Condition at 95th Percentile:", w_alpha)
    # G_r_w = G_r_w.sort_values(f"max_{W}")

    # interpolate and calculate VAR
    lower = F_r_w[F_r_w[f"max_{W}"] <= w_alpha].iloc[-1]
    upper = F_r_w[F_r_w[f"max_{W}"] >= w_alpha].iloc[0]
    x1, y1 = lower[f"max_{W}"], lower[column_name]
    x2, y2 = upper[f"max_{W}"], upper[column_name]
    if x1==x2:
        VAR_alpha= y1
    else:
        VAR_alpha = (y1 + (w_alpha - x1) * (y2 - y1) / (x2 - x1))
    VAR_alpha_norm=VAR_alpha/num_customers
    print('VAR Normalized by County Population:', VAR_alpha_norm)

    # Calculate CVAR
    x=p_W['probability']
    y=F_r_w[column_name]
    w_above_alpha = y >= VAR_alpha
    CVAR_alpha=(x[w_above_alpha]*y[w_above_alpha]).sum()*(1/(1-alpha))
    CVAR_alpha_norm=CVAR_alpha/num_customers
    print('CVAR Normalized by County Population:', CVAR_alpha_norm)

    return VAR_alpha_norm, w_alpha, CVAR_alpha_norm

# inputs - change as needed
state='Illinois'
county='Cook'
start='2014'
end='2024'
color='black'
value=0.75
weather_variable="gust"
#plot_multiple_bayesian_fits(weather_variable,'customer_hours')
#plot_multiple_wind_profiles(weather_variable)
# read in data
VAR, location, CVAR, effective_risk = calculate_VAR_CVAR(state, county, weather_variable,'customer_hours','y_avg')
plot_event_profile(location, state,county, color, VAR, CVAR, weather_variable,'customer_hours')
# weather_data=xr.open_dataset(f'../../weather_data/{state}/cleaned_weather_{state}_{county}_{start}_{end}.nc')
# ds_merged = weather_data.sel(station=weather_data.station != "SDB")
# df=weather_data[[weather_variable, 'sknt']].to_dataframe()
# mask = df[weather_variable] == 0
# df.loc[mask, weather_variable] = df.loc[mask,'sknt']
# weather_outage=pd.read_parquet(f'../Results/Data_All_{county}_{approach}_{value}_{start}-{end}.parquet')
# # outage_events=(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.parquet')
#
# #weather_outage = weather_outage[weather_outage['max_gust']<70]
#plot_multiple_bayesian_fits(target_variable,'cust_out_max')
# fit_outage_probability(weather_outage,target_variable,state,county,color)
#plot_multiple_wind_profiles(target_variable)
# plot_weather_distribution(df, weather_variable, county, state, False)

