import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import lognorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import math
import geopandas as gpd
import xarray as xr
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib as mpl
import pymc as pm
import arviz as az
import re

colors = plt.cm.Set2.colors
# plt.style.use('classic')
# publication-style defaults
mpl.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.linewidth": 1.2,
    "grid.linewidth": 0.6,

    "lines.linewidth": 2.3,

    "figure.dpi": 300,
    "savefig.dpi": 300,

    "font.family": "serif",
    "mathtext.fontset": "cm",
})

def myround(x, base=1):
    return base * round(x/base)

def sigmoid(x, w0, k):
    y = (1 / (1.0 + np.exp(-1 * k * (x - w0))))
    # y = 1 / (k * np.sqrt(2 * 3.14) * np.exp(-1*0.5 * ((x - w0) / k) ** 2))
    return y

def lognormal_pdf(x, shape, scale):
    return lognorm.pdf(x, shape, loc=0, scale=scale)

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

def plot_temp_wind(state, county):
    wmax = np.linspace(0, 65, 66)
    temps = {
        "low": ("blue", "Low Temps"),
        "mid": ("green", "Mid Temps"),
        "high": ("red", "High Temps"),
    }
    for temp, (color, label) in temps.items():
        df = pd.read_csv(
            f"results/monte_carlo_{state}_{county}_{temp}_tmpf.csv",
            header=None
        )
        df.columns = ["customer_hours"]
        df["max_gust"] = wmax
        # filter out non-observed values
        df = df[df["customer_hours"] >= 0]
        plt.scatter(df["max_gust"], df["customer_hours"], color=color, label=label)
    plt.legend()
    plt.yscale('log')
    plt.title(f"Wind Risk Sorted by Temp for {county} County, {state}")
    plt.show()
    breakpoint()

def plot_raw_outage_data(df, county, state, color, weather_variable, outage_variable, log_indicator):

    df[f'{outage_variable}_norm']=df[outage_variable]/get_customers_in_county(county,state)
    plt.scatter(df[weather_variable],df[f'{outage_variable}_norm'],color=color, label=county, alpha=0.5, s=80)
    if log_indicator:
        plt.yscale('log')
    plt.xlabel(weather_variable)
    plt.ylabel(outage_variable)
    plt.title(f'{county} County, {state} Precipitation Function Events')
    plt.grid(True)
    #plt.show()

def create_weather_distribution(weather_data, weather_param, county, state, color, double_peak):
    y, x, _ = plt.hist(weather_data[weather_param], bins=np.linspace(0,65,66), color='skyblue', edgecolor='black',
                       label='Histogram')
    # n is y-value of normalized distribution (frequency), bins is x value (wind speed)
    y = np.append(y, 0)
    plt.close()

    bin_width = np.mean(np.diff(x))
    y_pdf = y / (np.sum(y) * bin_width)
    if double_peak==True:
        # tweak guess values as needed
        params, _ = curve_fit(bimodal, x, y_pdf, p0=(40,10,0.25,70,10,0.25))
        x_fit = np.linspace(min(x), max(x), 120)
        y_fit = bimodal(x_fit, *params)
    else:
        params, _ = curve_fit(lognormal_pdf, x, y_pdf, p0=[1, np.mean(x)])
        x_fit = np.arange(0,66,1)
        y_fit = lognormal_pdf(x_fit, *params)

    # Generate fitted curve
    area = simpson(y_fit, x_fit)
    print("Area under fitted PDF:", area)
    np.savetxt(f'weather_profiles/{state}_{county}_{weather_param}_profile_params.txt', params)
    plt.scatter(x, y_pdf, color=color, alpha=0.65, label='Raw Binned Data')
    plt.plot(x_fit, y_fit, color=color, label='Fitted PDF')
    wind_profile = pd.DataFrame(columns=[x_fit, y_fit]).transpose()
    wind_profile.to_csv(f'weather_profiles/{state}_{county}_{weather_param}_profile.csv')
    plt.xlabel(weather_param)
    plt.ylabel('Probability of Occurrence')
    plt.title(f'{weather_param} Occurrences in Dataset for {county} County, {state}')
    plt.legend()

    plt.grid(True)
    plt.show()

def setup_weather_distribution(ds, state, county, weather_variable, color):
    print("Setting up Weather Profile.")
    if weather_variable=='tmpf':
        double_peak=True
    else:
        double_peak=False
    if weather_variable=='gust':
        ds['gust'] = ds['gust'].where(ds['gust'] != 0, ds['sknt'])
    df = (
        ds[[weather_variable]]
        .to_dataframe()
        .reset_index()[['time', weather_variable]]
    )
    df['gust']=df['gust'].round()
    df=df[df['gust']>0]
    create_weather_distribution(df, weather_variable, county, state, color, double_peak)


def plot_multiple_wind_profiles(target_variable, groups):
    # Load areas we are analyzing
    fig, ax = plt.subplots(figsize=(12, 7))
    for state, county, throwaway, color, throwaway, throwaway, throwaway in groups:
        path = f"weather_profiles/{state}_{county}_{target_variable}_profile.csv"

        profile = np.array(pd.read_csv(path).transpose())

        plt.plot(profile[0], profile[1], label=county, color=color)

    plt.xlabel("$w$")
    plt.ylabel("$p(w)$")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2
    )
    plt.title("$p(w)$ for Selected Counties")
    plt.tight_layout()

    plt.show()

def plot_multiple_outage_probabilities(groups):
    target_variable='gust'
    target_output='probability'
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in groups:
        path = (
            f"results/bayesian_probability_"
            f"{group[0]}_{group[1]}_"
            f"{target_variable}_{target_output}.csv"
        )
        fit = pd.read_csv(path)
        x = fit[target_variable]
        ax.plot(
            x,
            fit["y_avg"],
            label=group[1],
            color=group[3],
            linewidth=2.4
        )
    ax.set_xlabel("Maximum Wind $w$ (mph)")
    ax.set_ylabel("Outage probability $p_E(w)$")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 0.4)
    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2
    )
    fig.tight_layout()
    plt.show()

def plot_multiple_outage_magnitudes(groups):
    fig, ax = plt.subplots(figsize=(12, 7))

    target_variable='max_gust'
    target_output='customer_hours_norm'
    for state, county, throwaway, color, throwaway, throwaway, throwaway in groups:
        path = f"results/bayesian_single_{state}_{county}_{target_variable}_{target_output}.csv"
        fit = pd.read_csv(path)
        x = fit[f'{target_variable}']
        plt.plot(x, fit["y_avg"], label=county, color=color, zorder=3)
    plt.xlabel("Maximum Wind $w$")
    plt.ylabel("Outage Magnitude in Customer-Hours (Normalized)")
    plt.yscale("log")
    plt.xlim(0, 65)
    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2
    )
    fig.tight_layout()
    plt.show()


def plot_event_profile(location, state,county, color, var_y,CVAR, F_r_w, target_variable,target_output):
    # download weather profile
    weather_profile=pd.read_csv(f'weather_profiles/{state}_{county}_{target_variable}_profile.csv',
                                names=[target_variable,'probability'])
    # download Bayesian Fit
    # bayesian_results=pd.read_csv(f"results/bayesian_single_{state}_{county}_max_{target_variable}_{target_output}.csv")


    fig, ax = plt.subplots()
    ax.plot(weather_profile[target_variable], weather_profile['probability'],color=color)
    # make wind speed labels = actual wind speed using some kind of interpolation
    wind_speed_labels = list([0,10,20,30,40,50,60])
    x_labels = list()

    for i in wind_speed_labels:
        # interpolate the numbers
        lower = F_r_w[F_r_w[f"max_{target_variable}"] <= i].iloc[-1]
        upper = F_r_w[F_r_w[f"max_{target_variable}"] > i].iloc[0]
        x1, y1 = lower[f"max_{target_variable}"], lower['y_avg']
        x2, y2 = upper[f"max_{target_variable}"], upper['y_avg']
        y = y1 + (i - x1) * (y2 - y1) / (x2 - x1)
        if np.isnan(y):
            y=y1
        x_labels.append(y)
    x_labels = [round(x) for x in x_labels]
    sec = ax.secondary_xaxis(location=1)
    sec.set_xlabel('ALEC')
    sec.set_xticks(wind_speed_labels, labels=x_labels)

    plt.xlabel('Wind Gust')
    plt.ylabel('Probability')
    plt.axvline(location, color='gray')
    # plt.figtext(s=f" VAR = {math.ceil(var_y*1000)/1000} Customer-Hours Norm", x=0.5, y=0.8)
    plt.figtext(s=f" ALEC = {math.ceil(CVAR*1000)/1000}", x=0.5, y=0.75)
    #plt.title(f'Event Profile Plot for {county} County, {state}')
    plt.fill_between(x=weather_profile[target_variable], y1=weather_profile['probability'],
                     where=(weather_profile[target_variable] >= location), color=color,
                     alpha=0.2)
    plt.yscale('log')
    plt.show()
    # plt.savefig(f'var_plots/var_{state}_{county}_{target_variable}')
    # plt.close()


def normalize_text(text):
    text = str(text).lower().strip()

    # remove trailing "county"
    text = re.sub(r'\s+county$', '', text)

    return text


def get_customers_in_county(county, state=None):

    # Load data
    pdf = pd.read_csv('../../Eagle-idatasets/MCC.csv')

    county_to_fips = pd.read_csv(
        '../../Eagle-idatasets/county_fips_master.csv',
        encoding='latin'
    )

    # Normalize incoming query
    query = normalize_text(county)


    state_normalized = (
        normalize_text(state) if state is not None else None
    )

    county_normalized = query

    # Try detecting embedded state prefix
    if "_" in query:

        parts = query.split("_")

        # candidate state = first token
        possible_state = parts[0]

        # county = everything after first token
        possible_county = "_".join(parts[1:])

        # Use embedded state ONLY if no state explicitly passed
        if state_normalized is None:
            state_normalized = possible_state

        county_normalized = possible_county

    # ---------------------------------------------------
    # Normalize dataframe columns
    # ---------------------------------------------------

    county_to_fips['county_normalized'] = (
        county_to_fips['county_name']
        .astype(str)
        .apply(normalize_text)
    )

    county_to_fips['state_normalized'] = (
        county_to_fips['state_name']
        .astype(str)
        .apply(normalize_text)
    )

    # ---------------------------------------------------
    # Match county
    # ---------------------------------------------------

    ans = county_to_fips[
        county_to_fips['county_normalized'] == county_normalized
    ]

    # Match state if available
    if state_normalized is not None:
        ans = ans[
            ans['state_normalized'] == state_normalized
        ]

    # ---------------------------------------------------
    # Handle failures
    # ---------------------------------------------------

    if ans.empty:
        print(
            f"Error: county '{county}' "
            f"not matched with state '{state}'."
        )
        return 0

    # If multiple counties remain and no state specified
    if len(ans) > 1:
        print(
            f"Error: county '{county}' is ambiguous. "
            f"Specify a state."
        )
        return 0

    # ---------------------------------------------------
    # Get FIPS
    # ---------------------------------------------------

    target_fips = ans['fips'].values[0]

    pdf['County_FIPS'] = pd.to_numeric(
        pdf['County_FIPS'],
        downcast='integer',
        errors='coerce'
    )

    # ---------------------------------------------------
    # Match MCC rows
    # ---------------------------------------------------

    result = pdf[
        pdf['County_FIPS'] == target_fips
    ]

    if result.empty:
        return 0

    customers = result['Customers'].values[0]

    return customers

def get_customers_in_county_group(state, merged_data):
    unique_counties = merged_data.attrs['counties']
    unique_counties=unique_counties.split(', ')
    total_customers_all = sum(
        get_customers_in_county(county, state)
        for county in unique_counties
    )
    return total_customers_all

def calculate_VAR_CVAR(state, county, W, column_name, F_r_w):

    L='customer_hours'
    # Download Probabilistic Weather Profile
    p_W=pd.read_csv(f'weather_profiles/{state}_{county}_max_gust_profile.csv',
                                names=[W,'probability'])

    # # Download Outage Distribution Given Event Occurrence
    # G_r_w=pd.read_csv(f'results/bayesian_single_{state}_{county}_max_{W}_{L}.csv')
    #
    # # Download Event Probability
    # prob_event_W=pd.read_csv(f'results/bayesian_probability_{state}_{county}_{W}_outage.csv')


    num_customers = get_customers_in_county(county, state)
    alpha=0.95

    # F_r_w=pd.DataFrame([G_r_w[f'max_{W}'],(1-prob_event_W[column_name])+G_r_w[column_name]*prob_event_W[column_name]]).transpose()

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
    # log of y, take log - log var
    w_above_alpha = y >= VAR_alpha
    CVAR_alpha=(x[w_above_alpha]*y[w_above_alpha]).sum()*(1/(1-alpha))
    CVAR_alpha_norm=CVAR_alpha/num_customers
    print('CVAR Normalized by County Population:', CVAR_alpha_norm)

    y_log=np.log(F_r_w[column_name]/VAR_alpha)
    ALEC=(x[w_above_alpha]*y_log[w_above_alpha]).sum()*(1/(1-alpha))

    return VAR_alpha_norm, w_alpha, ALEC, F_r_w

def map_counties():
    # -----------------------------
    # Load US states shapefile
    # -----------------------------
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"

    states = gpd.read_file(url)

    # Remove Alaska, Hawaii, Puerto Rico for continental map
    exclude = ["Alaska", "Hawaii", "Puerto Rico"]

    states = states[~states["NAME"].isin(exclude)]

    # -----------------------------
    # Locations
    # -----------------------------
    locations = [
        {
            "label": "Brooklyn, NY\n(Kings County)",
            "lon": -73.9442,
            "lat": 40.6782,
            "color": "brown",
        },
        {
            "label": "Los Angeles, CA\n(LA County)",
            "lon": -118.2437,
            "lat": 34.0522,
            "color": "red",
        },
        {
            "label": "Boston, MA\n(Suffolk County)",
            "lon": -71.0589,
            "lat": 42.3601,
            "color": "green",
        },
        {
            "label": "Houston, TX\n(Harris County)",
            "lon": -95.3698,
            "lat": 29.7604,
            "color": "goldenrod",
        },
        {
            "label": "Phoenix, AZ\n(Maricopa County)",
            "lon": -112.0740,
            "lat": 33.4484,
            "color": "magenta",
        },
        {
            "label": "Seattle, WA\n(King County)",
            "lon": -122.3321,
            "lat": 47.6062,
            "color": "blue",
        },
        {
            "label": "Chicago, IL\n(Cook County)",
            "lon": -87.6298,
            "lat": 41.8781,
            "color": "black",
        },
        {
            "label": "Miami, FL\n(Miami-Dade County)",
            "lon": -80.1918,
            "lat": 25.7617,
            "color": "teal",
        },
    ]

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(16, 10))

    # Draw states
    states.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="black",
        linewidth=0.6
    )

    # Plot points + labels
    for loc in locations:
        ax.scatter(
            loc["lon"],
            loc["lat"],
            color=loc["color"],
            s=80,
            edgecolor="black",
            linewidth=0.5,
            zorder=5
        )

        ax.text(
            loc["lon"] + 1,
            loc["lat"] + 0.5,
            loc["label"],
            fontsize=9,
            color=loc["color"],
            weight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
                pad=1
            ),
            zorder=6
        )

    # Better continental US framing
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)

    # Remove axes
    ax.axis("off")

    # Title
    plt.title(
        "Selected U.S. Urban Areas",
        fontsize=20,
        weight="bold"
    )
    plt.gcf().patch.set_facecolor("white")
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.show()


def map_illinois_counties():
    # --------------------------------------------------
    # Load Illinois county shapefile from Census TIGER
    # --------------------------------------------------
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"

    counties = gpd.read_file(url)

    # Filter only Illinois counties
    il = counties[counties["STATEFP"] == "17"].copy()

    # --------------------------------------------------
    # County group definitions
    # --------------------------------------------------

    urban = (
        "Cook",
    )

    semi_urban = (
        "Champaign", "DeKalb", "DuPage",
        "Kane", "Kankakee", "Kendall",
        "Lake", "LaSalle", "McHenry",
        "McLean", "Madison", "Peoria",
        "Rock Island", "St. Clair", "Sangamon",
        "Tazewell", "Will", "Winnebago"
    )

    rural = (
        "Adams", "Alexander", "Bond", "Boone",
        "Brown", "Bureau", "Calhoun", "Carroll",
        "Cass", "Christian", "Clark", "Clay",
        "Clinton", "Coles", "Crawford", "Cumberland",
        "De Witt", "Douglas", "Edgar", "Edwards",
        "Effingham", "Fayette", "Ford", "Franklin",
        "Fulton", "Gallatin", "Greene", "Grundy",
        "Hamilton", "Hancock", "Hardin", "Henderson",
        "Henry", "Iroquois", "Jackson", "Jasper",
        "Jefferson", "Jersey", "Jo Daviess", "Johnson",
        "Knox", "Lawrence", "Lee", "Livingston",
        "Logan", "McDonough", "Macon", "Macoupin",
        "Marion", "Marshall", "Mason", "Massac",
        "Menard", "Mercer", "Monroe", "Montgomery",
        "Morgan", "Moultrie", "Ogle", "Perry",
        "Piatt", "Pike", "Pope", "Pulaski",
        "Putnam", "Randolph", "Richland", "Saline",
        "Schuyler", "Scott", "Shelby", "Stark",
        "Stephenson", "Union", "Vermilion", "Wabash",
        "Warren", "Washington", "Wayne", "White",
        "Whiteside", "Williamson", "Woodford"
    )

    # --------------------------------------------------
    # Assign categories + colors
    # --------------------------------------------------

    def classify_county(name):
        if name in urban:
            return "Urban"
        elif name in semi_urban:
            return "Semi-Urban"
        elif name in rural:
            return "Rural"
        else:
            return "Unknown"

    il["category"] = il["NAME"].apply(classify_county)

    color_map = {
        "Urban": "black",
        "Semi-Urban": "#4B0082",   # indigo
        "Rural": "#800080",        # purple
        "Unknown": "lightgray"
    }

    il["color"] = il["category"].map(color_map)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 16))

    color_map = {
        "Urban": "#3A3A3A",
        "Semi-Urban": "#8A63D2",
        "Rural": "#C88ACD",
        "Unknown": "#DDDDDD"
    }

    il.plot(
        ax=ax,
        color=il["color"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8
    )
    # --------------------------------------------------
    # Chicago label (Cook County)
    # --------------------------------------------------

    cook = il[il["NAME"] == "Cook"]

    # County centroid
    x = cook.geometry.centroid.x.iloc[0]
    y = cook.geometry.centroid.y.iloc[0]

    # Marker
    ax.scatter(
        x, y,
        color="black",
        s=80,
        edgecolor="white",
        linewidth=1,
        zorder=5
    )

    # Label
    ax.text(
        x + 0.15,
        y + 0.10,
        "Chicago\n(Cook County)",
        fontsize=10,
        color="black",
        weight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
            pad=2
        ),
        zorder=6
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="Urban (Chicago)"),
        Patch(facecolor="#4B0082", edgecolor="black", label="Semi-Urban (Counties with over 100,000 residents)"),
        Patch(facecolor="#800080", edgecolor="black", label="Rural (Counties with under 100,000 residents)"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=18,
        title_fontsize=18,
        frameon=False,          # removes box
    )

    # --------------------------------------------------
    # Styling
    # --------------------------------------------------

    # ax.set_title(
    #     "Illinois Counties by Urbanization Type",
    #     fontsize=20,
    #     weight="bold",
    #     pad=20
    # )

    ax.axis("off")
    plt.gcf().patch.set_facecolor("white")
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.show()

def plot_large_events_profile(groups, weather_variable):
    fig, ax = plt.subplots(figsize=(12, 7))
    for state, county, throwaway, color,  throwaway, throwaway, throwaway in groups:
        p_w = pd.read_csv(f"weather_profiles/{state}_{county}_{weather_variable}_profile.csv",
                          names=[weather_variable, 'probability'])
        p_w = p_w.dropna().reset_index()
        po_w=pd.read_csv(f'results/bayesian_probability_{state}_{county}_{weather_variable}_probability.csv')
        Cout_hrs_norm = pd.read_csv(f'results/bayesian_single_{state}_{county}_max_{weather_variable}_customer_hours_norm.csv')
        p_o_and_w=p_w['probability']*po_w['y_avg']
        Risk=Cout_hrs_norm['y_avg']*p_o_and_w
        total_risk=np.log(Risk.sum())
        print(f'Total {weather_variable} risk for {county} County, {state}: {total_risk}')
        p_o_and_w.to_csv(f'weather_profiles/{state}_{county}_extreme_events_profile_{weather_variable}.csv')
        plt.plot(
            po_w[weather_variable],
            Risk,
            marker='o',
            color=color,
            label=county
        )

    plt.xlabel(f'Max {weather_variable}')
    plt.ylabel('Weighted Risk')
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.25
    )
    plt.show()


def monte_carlo_sim(ds, event_data, multivariable):
    num_trials=1
    # replace 0 gust values with the observed wind speed, round to nearest integer
    var = ds['gust'].where(ds['gust'] != 0, ds['sknt'])
    var = var.round(1)
    if multivariable==True:
        # find temp thresholds
        low_temp_threshold=ds["tmpf"].quantile(0.33, dim="time")
        high_temp_threshold=ds["tmpf"].quantile(0.67, dim="time")
        # for low temp mc simulation:
        ds=ds.where(ds["tmpf"] < high_temp_threshold,ds["tmpf"] > low_temp_threshold)
    # Create time series df that has gust and event number
    df = xr.Dataset({
        'gust': var,
        'event_number': ds['event_number_eaglei']
    }).to_dataframe()
    # remove very high winds from the sample
    df = df[df['gust'] < 66]
    df = (
        df.groupby(level='time')
          .agg({
              'gust': 'max',
              'event_number': 'max'
          })
    )
    # replace all gusts with the max gust from the actual event
    df_merged = df.merge(
        event_data[['event_number', 'max_gust', 'customer_hours']],
        on='event_number',
        how='left'
    )
    df_merged['gust'] = np.where(
        df_merged['event_number'] > 0,
        df_merged['max_gust'],
        df_merged['gust']
    )
    df_merged['customer_hours'] = np.where(
        df_merged['event_number'] > 0,
        df_merged['customer_hours'],
        0
    )
    df_merged = df_merged.drop(columns=['max_gust'])

    # create loss array
    loss=np.full(shape=(66, num_trials), fill_value=0, dtype=float)
    avg_loss=np.full(shape=(66,1),fill_value=0,dtype=float)
    running_avg=np.full(shape=(1,num_trials),fill_value=0,dtype=float)
    # iterate through wind speed
    for w in range(0, 66, 1):
        print(f'wind speed = {w} MPH')
        # find sample pool for the selected wind speed
        df_samples=df_merged[df_merged['gust']==w]
        if df_samples.empty:
            avg_loss[w]=-1
            print('No samples for this integer wind speed')
        else:
        # iterate through MC trials
            sum=0
            avg=np.full(shape=(num_trials,1),fill_value=0,dtype=float)
            for n in range(num_trials):
                # draw random sample from DS
                sample=df_samples.sample()
                # if event number in DS is 0 or -1 assign loss to 0
                if sample['event_number'].values <= 0:
                    loss[w,n]=0
                # otherwise: assign loss to the total customer-hrs associated w event
                else:
                    loss[w,n]=float(sample['customer_hours'].values)
                sum=sum+loss[w,n]
                if sum==0:
                    avg[n]=0
                else:
                    if n==0:
                        avg[n]=loss[w,n]
                    else:
                        avg[n]=np.sum(loss[w,0:n])/n
            # average the loss over all trials
            # plt.scatter(np.linspace(0,num_trials+1,num_trials),avg)
            avg_loss[w]=np.sum(loss[w:,])/num_trials
            # plt.show()
            print(f'Expected loss for wind speed {w}: {avg_loss[w]}')
    column_names=['max_gust', 'customer_hours']
    max_gust=np.linspace(0,65,66)
    customer_hours=avg_loss
    avg_loss_df=pd.DataFrame({
    "max_gust": max_gust,
    "customer_hours": customer_hours.ravel()})
    avg_loss_df=avg_loss_df[avg_loss_df['customer_hours']>=0]
    #plt.scatter(np.linspace(0,65,66), avg_loss)
    return avg_loss_df
#plt.yscale('log')
#plt.show()

def event_histograms(event_data, label, color, state, county, multi_county):
    event_data['max_gust']=event_data['max_gust'].where(event_data['max_gust'] != 0, event_data['max_sknt'])
    event_data['max_gust']=round(event_data['max_gust'])
    event_data['max_tmpf']=round(event_data['max_tmpf'])
    event_data['gust_bin'] = pd.cut(
        event_data['max_gust'],
        bins=[0, 10, 20, 30, 40, 100],
        labels=['0-10', '10-20', '20-30', '30-40', '40+']
    )
    event_data['tmpf_bin'] =pd.cut(
        event_data['max_tmpf'],
        bins=[-20,30,100,140],
        labels=['Below 30', '30-100', 'Above 100']
    )
    unique_counties = event_data['counties_affected'].unique()
    total_customers_all = sum(
        get_customers_in_county(county, state)
        for county in unique_counties
    )
    event_data['customer_hours_norm'] = (
            event_data['customer_hours'] / total_customers_all
    )

    event_data['log_customer_hours_norm'] = np.log(
        event_data['customer_hours_norm']
    )

    # Plot
    ax=sns.histplot(
        data=event_data,
        x='log_customer_hours_norm',
        hue='tmpf_bin',
        bins=50,
        palette=sns.color_palette(f'light:{color}', n_colors=5),
        multiple='fill'  # or 'dodge' or 'fill'
    )
    # ax.set_ylim(0,4000)
    # ax.set_xlim(-14,4)
    plt.xlabel("Log(Customer-Hours/Num_Customers)")
    plt.ylabel("Portion of Events")
    plt.title(f"Customer Hours by Max Wind for {county} County")
    plt.show()

    # sns.ecdfplot(data=event_data, x='customer_hours_norm', complementary=True, log_scale=True, color=color)
    # plt.xlabel('Event Size')
    # plt.ylabel('CCDF')
    # plt.yscale('log')
    # plt.title(f'Empirical CCDF (Log-Log Scale) for {label}')
    # plt.show()

    large_event_threshold=0.05
    # filter only 2015 and up
    event_data['start_time']=pd.to_datetime(event_data['start_time'])
    event_data=event_data[event_data['start_time'].dt.year>=2015]
    # filter by "large events"
    event_data_large=event_data[event_data['customer_hours_norm']>=large_event_threshold]
    SALEDI=np.sum(np.log(event_data_large['customer_hours_norm']/large_event_threshold))*1/10
    # now calculate SALEDI for each gust bin
    SALEDI_by_bin = (
        event_data_large
        .groupby('max_tmpf')['customer_hours_norm']
        .apply(lambda x: np.sum(np.log(x / large_event_threshold)) / 10)
    )
    print(f'SALEDI for {county} County: {SALEDI}')
    print(f'SALEDI by bin: {SALEDI_by_bin}')
    SALEDI_by_bin_cumsum=SALEDI_by_bin.cumsum()
    # plt.plot(SALEDI_by_bin_cumsum.index, SALEDI_by_bin_cumsum, color=color)
    return SALEDI_by_bin

def plot_wind_temp_relationship(ds,color):
    var1 = ds['gust'].where(ds['gust'] != 0, ds['sknt'])
    var1 = var1.round(1)
    var2=ds['tmpf'].round(1)
    plt.scatter(var2,var1,color=color)
    plt.title('Wind-Temperature Relationship')
    plt.ylabel('Max Wind Speed')
    plt.xlabel('Temperature (Degrees F)')
    plt.ylim(0,80)
    plt.xlim(-10,120)

    plt.show()

def create_probability_mapping(ds,multi_county,state,county, weather_variable):

    if multi_county==True:
        print("Aligning stations with counties for each timestamp.")
        if weather_variable=='gust':
            ds['gust'] = xr.where(ds['gust'] != 0, ds['gust'], ds['sknt'])
        stations = ds['station'].values

        # station format: "State_County_Station"
        station_to_county = np.array([
            str(s).split('_')[1] for s in stations
        ])

        ds = ds.assign_coords(county=("station", station_to_county))
        # first align gust to county dimension via groupby
        gust_by_county = ds[weather_variable].groupby('county').max('station')

        # customers_out and event_number_eaglei are already (time, county)
        outages = ds[['customers_out', 'event_number_eaglei']]

        ds2 = xr.Dataset({
            weather_variable: gust_by_county,
            'customers_out': outages['customers_out'],
            'event_number_eaglei': outages['event_number_eaglei']
        })

        # sort
        ds2 = ds2.sortby('time')

        delta_out = ds2["customers_out"].diff("time")
        delta_out = delta_out.reindex(time=ds2.time)  # align first step
        delta_pos = (delta_out > 0).astype("int8")
        event_pos = (ds2["event_number_eaglei"] > 0).astype("int8")

        event_caused = delta_pos * event_pos

        ds2["delta_out"] = delta_out
        ds2["event_caused"] = event_caused
        df = ds2.to_dataframe().reset_index()

    else:
        # Create time series df that has gust, event number, and customers_out
        if weather_variable=='gust':
            var = ds['gust'].where(ds['gust'] != 0, ds['sknt'])
        else:
            var=ds[weather_variable]
        df = xr.Dataset({
            weather_variable: var,
            'event_number_eaglei': ds['event_number_eaglei'],
            'customers_out': ds['customers_out']
        }).to_dataframe()
        df = (
            df.groupby(level='time')
            .agg(
                gust=(weather_variable, 'max'),
                event_number_eaglei=('event_number_eaglei', 'max'),
                customers_out=('customers_out', 'sum')
            )
        )

        df['event_caused'] = (
                (df['event_number_eaglei'] > 0) &
                (df['customers_out'] > df['customers_out'].shift(1))
        ).astype(int)
    # remove any columns where event num eagle-i is 0 since we have uncertainty
    df=df[df['event_number_eaglei']!=0]
    print("Finding share of gusts that resulted in outages.")
    df[f'{weather_variable}_bin']=pd.qcut(df[weather_variable], q=100, labels=False, duplicates='drop')
    grouped = (
        df.groupby(f'{weather_variable}_bin')
        .agg(
            gust=(weather_variable, 'mean'),
            probability=('event_caused', 'mean'),
            sample_size=('event_caused', 'size'),
            outage_count=('event_caused', 'sum')
        )
        .sort_index()
    )
    grouped.to_parquet(f'results/outage_probability_{state}_{county}_{weather_variable}.parquet')

# 2) Helper to fit and predict
def fit_loglinear(x,y_obs,x_new, label):
    y_log = np.log(y_obs + 1e-10)
    with pm.Model() as m:
        if label=='probability':
            a = pm.Uniform('a', lower=-15, upper=0)
            b     = pm.Uniform('b', lower=0, upper=0.5)
        else:
            a=pm.Uniform('a', lower=-10, upper=5)
            b=pm.Uniform('b', lower=-1, upper=1)
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu_log = a + b*x
        pm.Normal('y', mu=mu_log, sigma=sigma, observed=y_log)

        trace = pm.sample(1000, tune=1000, chains=4, cores=1,
                          target_accept=0.95,
                          random_seed=42, return_inferencedata=True)

    post = az.extract(trace).to_dataframe()
    a_samps = post['a'].values[:, None]
    b_samps = post['b'].values[:, None]
    param_stats = post[['a', 'b']].agg(['mean', 'std']).T
    param_stats.columns = ['posterior_mean', 'posterior_std']
    print(param_stats)

    mu_log_new = a_samps + b_samps * x_new
    y_new = np.exp(mu_log_new)
    return y_new.mean(axis=0), y_new.std(axis=0), x_new


def fit_sigmoid(x,y,x_new):
    with pm.Model() as m:
        with pm.Model() as m:
            a = pm.Uniform('a', lower=0, upper=1) # saturation level
            k = pm.Uniform('k', lower=0, upper=10) # slope
            w0 = pm.Uniform('w0', lower=0, upper=100) # midpoint
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = a * pm.math.sigmoid(k * (x - w0))
            pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            trace = pm.sample( 1000, tune=1000,
            chains=4, cores=1,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True )

    post = az.extract(trace).to_dataframe()

    a_samps  = post['a'].values[:, None]
    k_samps  = post['k'].values[:, None]
    w0_samps = post['w0'].values[:, None]

    param_stats = post[['a','k','w0']].agg(['mean','std']).T
    param_stats.columns = ['posterior_mean','posterior_std']
    print(param_stats)

    mu_new = a_samps / (1.0 + np.exp(-k_samps * (x_new - w0_samps)))

    return mu_new.mean(axis=0), mu_new.std(axis=0), x_new

def plot(df, state, county, target_variable,
         target_output, color, mean, std, label, x,  x_new):

    plt.scatter(x, df[target_output], color=color, alpha=0.5, s=80,label='Observed Data')
    # posterior mean
    plt.plot(x_new, mean, color=color, lw=2, label='Predicted Mean')
    # 95% band
    plt.fill_between(
        x_new,
        mean - 2*std,
        mean + 2*std,
        color=color, alpha=0.2, label='95% CI (±2σ)'
    )
    plt.ylabel(target_output, fontsize=18, fontweight='bold')
    if label=='single':
        plt.yscale('log')
    if label=='probability':
        plt.ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim(0,65)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=16, loc='upper left')
    data_results = pd.DataFrame(columns=[target_variable, 'y_avg', 'y_lower', 'y_upper'])
    data_results[target_variable]=x_new
    data_results['y_avg']=mean
    data_results['y_lower']=mean - 2*std
    data_results['y_upper']=mean + 2*std
    if label=='probability':
        # fix columns to 1
        cols = ["y_avg", "y_lower", "y_upper"]
        data_results[cols] = data_results[cols].clip(upper=1)
    data_results.to_csv(f'results/bayesian_{label}_{state}_{county}_{target_variable}_{target_output}.csv')

    plt.xlabel(f'{target_variable}', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(fontsize=12)
    # plt.tight_layout()
    plt.savefig(f'results/bayesian_{label}_{state}_{county}_{target_variable}_{target_output}.png')
    plt.close()


# inputs
# state = 'Illinois'
# county = 'semi_urban'
# color = 'indigo'

#
# # # load datasets for loglinear fits
# df=pd.read_parquet(f'../../events_stats/county_events/{state}/{county}_event_stats.parquet')
# df = pd.read_parquet(f'../../events_stats/spatiotemporal_events/event_stats_{state}_{county}.parquet')
def load_loglinear(state, county, df, color, weather_variable):
    target_variable = f'max_{weather_variable}'
    target_output = 'customer_hours_norm'
    if weather_variable=='gust':
        mask = df[target_variable] == 0
        df.loc[mask, target_variable] = df.loc[mask,'max_sknt']
    label='single'
    df[target_variable]=df[target_variable].round()
    df=df.groupby(target_variable, as_index=False)[target_output].mean()
    df=df[df[target_variable]<=65]
    x = df[target_variable].values
    y=df[target_output].values
    x_new = np.linspace(0, 65, 66)
    mean, std, x_new = fit_loglinear(x,y,x_new, label)
    plot(df, state, county, target_variable, target_output, color, mean, std, label, x=x, x_new=x_new)
# load datasets for sigmoid fits

def load_sigmoid(state,county,color, weather_variable):
    df=pd.read_parquet(f'results/outage_probability_{state}_{county}_{weather_variable}.parquet')
    target_output='probability'
    label=target_output
    x = df[weather_variable].values
    y=df[target_output].values
    x_new = np.linspace(0, 65, 66)
    mean, std, x_new = fit_loglinear(x,y,x_new, label)

    plot(df, state, county, weather_variable, target_output, color, mean, std, label, x=x, x_new=x_new)


# inputs - change as needed
# state='California'
# county='Los Angeles'
# start='2014'
# end='2024'
# color='red'
# value=0.75
# weather_variable="gust"
# outage_variable='customer_hours'
# plot_temp_wind(state, county)
# plot_multiple_bayesian_fits(weather_variable,'customer_hours')
#plot_multiple_wind_profiles(weather_variable)
# VAR, location, ALEC, F_r_w = calculate_VAR_CVAR(state, county, weather_variable,'customer_hours','y_avg')
# plot_event_profile(location, state,county, color, VAR, ALEC, F_r_w, weather_variable,'customer_hours')

# counties = [
#     ("Washington", "King", "blue"),
#     ("Massachusetts", "Suffolk", "green"),
#     ("California", "Los Angeles", "red"),
#     ("Florida", "Miami-Dade", "teal"),
#     ("Arizona", "Maricopa", "magenta"),
#     ("Texas", "Harris", "goldenrod"),
#     ("Illinois", "Cook", "black"),
#     ("New York", "Kings", "brown")
# ]
# counties=[
#     ("Illinois", "Cook", "black"),
#     ("Illinois", "semi_urban", "indigo"),
#     ("Illinois","rural", "purple")
# ]

# plot_multiple_wind_profiles('gust', counties)

# plot_large_events_profile(counties)
    # ds=xr.open_dataset(f'../../weather_data/{state}/cleaned_weather_{state}_{county}_2015_2025.nc')
    # var = ds['gust'].where(ds['gust'] != 0, ds['sknt'])
    # ds_small = xr.Dataset({
    #     'gust': var
    # })
    #
    # df = (
    #     ds[['gust']]
    #     .to_dataframe()
    #     .reset_index()[['time', 'gust']]
    # )
    # df['gust']=df['gust'].round()
    # df=df[df['gust']>0]
    # double_peak='False'
    # plot_weather_distribution(df, 'gust', county, state, color, double_peak)
