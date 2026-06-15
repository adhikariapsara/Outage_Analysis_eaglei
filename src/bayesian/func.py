# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
from scipy.stats import lognorm
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import math
import geopandas as gpd
import xarray as xr
import seaborn as sns
import matplotlib as mpl
import pymc as pm
import arviz as az
import re

colors = plt.cm.Set2.colors
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

def create_weather_distribution(weather_data, weather_param, county, state, color, double_peak):
    y, x, _ = plt.hist(weather_data[weather_param], bins=np.linspace(0,120,121), color='skyblue', edgecolor='black',
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
    plt.savefig(f'weather_profiles/{state}_{county}_{weather_param}_profile.png')
    plt.close()

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
    df[weather_variable]=df[weather_variable].round()
    # df=df[df['gust']>0]
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

def plot_multiple_outage_probabilities(groups, target_variable):
    target_output='probability'
    fig, ax = plt.subplots(figsize=(8, 5))

    for group in groups:
        path = (
            f"results/bayesian_probability_"
            f"{group[0]}_{group[1]}_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
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
    ax.set_xlim(0, 120)
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
    # plt.savefig('groups_probability_comparison.png')

def plot_multiple_outage_probabilities_fig(groups):
    target_variable='gust'
    target_output='probability'

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True
    )

    for ax, (state, label, urban_color, rural_color) in zip(
            axes.ravel(),
            groups
    ):
        urban_path = (
            f"results/bayesian_probability_"
            f"{state}_{state}_rucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )
        fit1 = pd.read_csv(urban_path)
        x1 = fit1[target_variable]
        ax.plot(
            x1,
            fit1["y_avg"],
            label=f'RUCC Level 1',
            color=urban_color,
            linewidth=2.4
        )
        ax.fill_between(
            x1,
            fit1['y_lower'],
            fit1['y_upper'],
            color=urban_color, alpha=0.2
        )
        rural_path = (
            f"results/bayesian_probability_"
            f"{state}_{state}_nonrucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )
        fit2 = pd.read_csv(rural_path)
        x2 = fit2[target_variable]
        ax.plot(
            x2,
            fit2["y_avg"],
            label=f'Other',
            color=rural_color,
            linewidth=2.4
        )
        ax.fill_between(
            x2,
            fit2['y_lower'],
            fit2['y_upper'],
            color=rural_color, alpha=0.2
        )
        ax.set_xlabel("Maximum Wind $w$ (mph)")
        ax.set_ylabel("Outage probability $p_E(w)$")
        ax.set_title(state, fontsize=12)
        ax.set_xlim(0, 65)
        ax.set_ylim(0, 0.5)
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

    plt.show()
    # plt.savefig('groups_probability_comparison.png')

def plot_multiple_outage_magnitudes(groups, target_variable):
    fig, ax = plt.subplots(figsize=(12, 7))

    target_output='customer_hours_norm'
    for state, county, throwaway, color, throwaway, throwaway, throwaway in groups:
        path = f"results/bayesian_single_{state}_{county}_{target_variable}_{target_output}_spatiotemporal.csv"
        fit = pd.read_csv(path)
        x = fit[f'{target_variable}']
        plt.plot(x, fit["y_avg"], label=county, color=color, zorder=3)
    plt.xlabel("Maximum Wind $w$")
    plt.ylabel("Outage Magnitude in Customer-Hours (Normalized)")
    plt.yscale("log")
    plt.xlim(0, 120)

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
    # plt.savefig('groups_bayesian_single_comparison.png')
    #

def plot_multiple_outage_magnitudes_fig(groups):
    target_variable='max_gust'
    target_output='customer_hours_norm'

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True
    )

    for ax, (state, label, urban_color, rural_color) in zip(
            axes.ravel(),
            groups
    ):
        urban_path = (
            f"results/bayesian_single_"
            f"{state}_{state}_rucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )
        fit1 = pd.read_csv(urban_path)
        x1 = fit1[target_variable]
        ax.plot(
            x1,
            fit1["y_avg"],
            label=f'RUCC Level 1',
            color=urban_color,
            linewidth=2.4
        )
        ax.fill_between(
            x1,
            fit1['y_lower'],
            fit1['y_upper'],
            color=urban_color, alpha=0.2
        )
        rural_path = (
            f"results/bayesian_single_"
            f"{state}_{state}_nonrucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )
        fit2 = pd.read_csv(rural_path)
        x2 = fit2[target_variable]
        ax.plot(
            x2,
            fit2["y_avg"],
            label=f'Other',
            color=rural_color,
            linewidth=2.4,
        )
        ax.fill_between(
            x2,
            fit2['y_lower'],
            fit2['y_upper'],
            color=rural_color, alpha=0.2
        )
        ax.set_xlabel("Maximum Wind $w$ (mph)")
        ax.set_ylabel("Expected Loss $E[C_{out-hrs}]$")
        ax.set_title(state, fontsize=12)
        ax.set_yscale('log')
        ax.set_xlim(0, 65)
        ax.set_ylim(10e-5, 10e2)
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

    plt.savefig('groups_single_comparison.png')

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

def calculate_VAR_CVAR(state, group, event_data, color):

    # find the couthrs corresponding with 20mph
    bayesian_single_fit=pd.read_csv(f'results/bayesian_single_{state}_{group}_max_gust_customer_hours_norm_spatiotemporal.csv')
    cout_hrs_20=bayesian_single_fit.loc[20,'y_avg']
    # cout_hrs_20=0.05
    values = bayesian_single_fit['y_avg'].to_numpy()
    #
    # # Interpolated index
    # interp_idx = np.interp(
    #     cout_hrs_20,
    #     values,
    #     np.arange(len(values))
    # )
    # print(f'gust at 0.05: {interp_idx}')
    alpha = (event_data['customer_hours_norm'] <= cout_hrs_20).mean() * 100

    var = event_data['customer_hours_norm'][event_data['customer_hours_norm'] >= cout_hrs_20].reset_index()
    var=var.replace([np.inf, -np.inf], np.nan)
    var.dropna(axis=0, how="any",inplace=True)
    cvar=var['customer_hours_norm'].mean()
    alec=np.sum(np.log(var['customer_hours_norm']/cout_hrs_20))*1/(len(var))
    print(f'CVAR: {cvar:.2f}')
    print(f'ALEC: {alec:.2f} at the {alpha:.2f}th Percentile of data (past 20 MPH)')

    # approach #2: large event threshold flat (95th pctile)
    # alpha=95
    # large_event_threshold = event_data["customer_hours_norm"].quantile(alpha / 100)
    # var=event_data['customer_hours_norm'][event_data['customer_hours_norm'] >= large_event_threshold]
    # gust_at_threshold=bayesian_single_fit.loc['max_gust']

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


def map_us_counties():
    state_colors = {
        "53": {"dark": "#2166AC", "light": "#D1E5F0"},  # Washington
        "48": {"dark": "#B8860B", "light": "#F6E8A6"},  # Texas
        "06": {"dark": "#B2182B", "light": "#F4CCCC"},  # California
        "25": {"dark": "#1B7837", "light": "#D9F0D3"},  # Massachusetts
        "17": {"dark": "#54278F", "light": "#D9D9D9"},  # Illinois
        "04": {"dark": "#C51B7D", "light": "#F4CAE4"},  # Arizona
        "36": {"dark": "#8C510A", "light": "#DFC27D"},  # New York
        "12": {"dark": "#01665E", "light": "#C7EAE5"},  # Florida
    }

    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
    counties = gpd.read_file(url)

    # Create county FIPS code
    counties["FIPS"] = (
            counties["STATEFP"] + counties["COUNTYFP"]
    ).astype(int)
    # ------------------------------------------------------------------
    # Load RUCC data
    # ------------------------------------------------------------------

    rucc = pd.read_csv(
        f"../../misc/Ruralurbancontinuumcodes2023.csv",
        header=None,
        names=["FIPS", "State", "County_Name", "Attribute", "Value"],
        encoding="latin1"

    )
    rucc["FIPS"] = rucc["FIPS"].astype(str).str.zfill(5)
    counties["FIPS"] = (
            counties["STATEFP"] + counties["COUNTYFP"]
    )
    rucc_codes = (
        rucc.loc[rucc["Attribute"] == "RUCC_2023",
        ["FIPS", "Value"]]
        .copy()
    )

    rucc_codes["RUCC_2023"] = rucc_codes["Value"].astype(int)

    counties = counties.merge(
        rucc_codes[["FIPS", "RUCC_2023"]],
        on="FIPS",
        how="left"
    )

    # ------------------------------------------------------------------
    # Binary classification:
    # RUCC == 1  -> dark blue
    # RUCC != 1  -> light blue
    # ------------------------------------------------------------------

    counties["RUCC1"] = counties["RUCC_2023"] == 1

    dark_blue = "#08519c"
    light_blue = "#c6dbef"

    counties["plot_color"] = counties["RUCC1"].map(
        {True: dark_blue, False: light_blue}
    )

    # ------------------------------------------------------------------
    # States to display
    # ------------------------------------------------------------------

    states = {
        "Washington": "53",
        "Illinois": "17",
        "New York": "36",
        "Massachusetts": "25",
        "California": "06",
        "Arizona": "04",
        "Texas": "48",
        "Florida": "12",
    }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 10),
        constrained_layout=True
    )

    for ax, (state_name, state_fips) in zip(
            axes.ravel(),
            states.items()
    ):
        gdf = counties[counties["STATEFP"] == state_fips]

        dark = state_colors[state_fips]["dark"]
        light = state_colors[state_fips]["light"]

        # Rural counties
        gdf[gdf["RUCC_2023"] != 1].plot(
            ax=ax,
            color=light,
            edgecolor="white",
            linewidth=0.25
        )

        # Urban counties
        gdf[gdf["RUCC_2023"] == 1].plot(
            ax=ax,
            color=dark,
            edgecolor="white",
            linewidth=0.25
        )

        ax.set_title(state_name, fontsize=12)
        ax.set_axis_off()
    plt.savefig('states.jpeg')

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    legend_handles = [
        Patch(facecolor=dark_blue,
              edgecolor="black",
              label="RUCC = 1"),
        Patch(facecolor=light_blue,
              edgecolor="black",
              label="RUCC ≠ 1")
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False
    )

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

        print(f'{weather_variable} risk above 20 MPH for {county} County, {state}: {p_o_and_w.iloc[20:].sum()/1e-3} x 10^-3')
        p_o_and_w.to_csv(f'weather_profiles/{state}_{county}_extreme_events_profile_{weather_variable}.csv')
        plt.plot(
            po_w[weather_variable],
            p_o_and_w,
            marker='o',
            color=color,
            label=county
        )

    plt.xlabel(f'Max {weather_variable}')
    plt.ylabel('Joint Probability')
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
    plt.savefig('groups_risk_comparison.png')


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
    return avg_loss_df


def event_histograms(groups):

    # plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for state, county, multi_county, color, customers, throwaway, throwaway in groups:

        event_data = pd.read_parquet(f'../../events_stats/spatiotemporal_events/{county}_event_stats_2015_2025.parquet')
        # for comparison, we only care about eaglei event data
        event_data = event_data[event_data['event_method'] == 'eaglei']
        county_customer_map = {
            county: get_customers_in_county(county, state)
            for county in event_data["counties_affected"].unique()
        }
        event_data["customer_hours_norm"] = (
                event_data["customer_hours"]
                / event_data["counties_affected"].map(county_customer_map)
        )
        event_data['max_gust']=event_data['max_gust'].where(event_data['max_gust'] != 0, event_data['max_sknt'])

        # Plot
        ax=sns.ecdfplot(data=event_data, x='customer_hours_norm', complementary=True, log_scale=True, color=color, label=county)

    plt.xlabel('Event Size')
    plt.ylabel('CCDF')
    plt.yscale('log')
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
    plt.show()

def find_rucc_groups(state_name, group):
    us_state_map = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming",
        "DC": "District of Columbia"
    }
    failed_csv = f"../../outage_data/{state_name}/failed_counties.csv"
    df = pd.read_csv(
        f"../../misc/Ruralurbancontinuumcodes2023.csv",
        header=None,
        names=["FIPS", "State", "County_Name", "Attribute", "Value"],
        encoding="latin1"

    )

    rucc_df = df[
        df["Attribute"].str.contains("RUCC", case=False, na=False)
    ].copy()

    rucc_df = rucc_df.rename(columns={
        "State": "state",
        "County_Name": "county",
        "Value": "rucc"
    })

    # Convert state abbreviations
    rucc_df["state"] = rucc_df["state"].map(us_state_map)

    # Clean county names
    rucc_df["county"] = (
        rucc_df["county"]
        .str.replace(" County", "", regex=False)
        .str.strip()
    )

    # Numeric RUCC
    rucc_df["rucc"] = pd.to_numeric(
        rucc_df["rucc"],
        errors="coerce"
    )

    # ==================================================
    # FILTER TO TARGET STATE
    # ==================================================

    state_df = rucc_df[
        rucc_df["state"].str.lower() == state_name.lower()
        ].copy()

    if state_df.empty:
        raise ValueError(f"No counties found for state: {state_name}")

    # ==================================================
    # LOAD FAILED COUNTIES
    # ==================================================
    try:
        failed_df = pd.read_csv(failed_csv)
        failed_set = set(
            failed_df["county"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        # remove failed counties
        state_df = state_df[
            ~state_df["county"]
            .str.lower()
            .isin(failed_set)
        ]
    except pd.errors.EmptyDataError:
        print("No failed counties found for state:", state_name)

    group1 = tuple(
        sorted(state_df.loc[state_df["rucc"] == 1, "county"].unique())
    )

    group2 = tuple(
        sorted(state_df.loc[state_df["rucc"] != 1, "county"].unique())
    )

    county_groups = (group1, group2)
    if group =='rucc1':
        return group1
    elif group =='nonrucc1':
        return group2
    else:
        return county_groups



def calculate_saledi(state, event_data, type):
    large_event_threshold = 0.05
    # filter by "large events"
    event_data_large=event_data[event_data['customer_hours_norm']>=large_event_threshold]
    county_group=find_rucc_groups(state, type)
    SALEDI=[]
    SALEDI_past_20=[]
    for county in county_group:
        event_data_large_county=event_data_large[event_data_large['counties_affected']==f'{state}_{county}']
        SALEDI_county=np.sum(np.log(event_data_large_county['customer_hours_norm']
                               / large_event_threshold)) * 1 / 10
        if np.isnan(SALEDI_county) or np.isinf(SALEDI_county):
            continue
        else:
            SALEDI.append(SALEDI_county)
        event_data_large_county_past_20=event_data_large_county[event_data_large_county['max_gust']>=20]
        SALEDI_past_20_county=np.sum(np.log(event_data_large_county_past_20['customer_hours_norm']
                                     / large_event_threshold)) * 1 / 10
        if np.isnan(SALEDI_past_20_county) or np.isinf(SALEDI_past_20_county):
            continue
        else:
            SALEDI_past_20.append(SALEDI_past_20_county)

    total_SALEDI=sum(SALEDI) / len(SALEDI)
    total_SALEDI_past_20=sum(SALEDI_past_20) / len(SALEDI_past_20)
    print(f"SALEDI: {total_SALEDI}")
    print(f'SALEDI for large events >= 20 MPH: {total_SALEDI_past_20}')
    print(f'Percent large events attributed to wind: {total_SALEDI_past_20/total_SALEDI*100:.2f}%')

    # SALEDI=np.sum(np.log(event_data_large['customer_hours_norm']/large_event_threshold))*1/5
    # # now calculate SALEDI for each gust bin
    # SALEDI_by_bin = (
    #     event_data_large
    #     .groupby('max_gust')['customer_hours_norm']
    #     .apply(lambda x: np.sum(np.log(x / large_event_threshold)) / 5)
    # )
    # print(f'SALEDI for {county}: {SALEDI}')
    # print(f'SALEDI by bin for {county}: {SALEDI_by_bin}')

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
    grouped=grouped[grouped['sample_size']>100]
    grouped.to_parquet(f'results/outage_probability_{state}_{county}_{weather_variable}.parquet')


# 2) Helper to fit and predict
def fit_loglinear(x, y_obs, x_new, label):
    y_log = np.log(y_obs + 1e-10)

    with pm.Model() as m:
        if label == 'probability':
            a = pm.Normal('a', -15, 0)
            b = pm.Normal('b', 0, 0.5)
            mu_log = a + b * x

        else:
            a = pm.Normal('a', -10, 10)
            b = pm.Normal('b', -1, 1)
            c = pm.HalfNormal("c", sigma=0.01)
            mu_log = a + b * x + c * x ** 2

        sigma_a = pm.Normal("sigma_a", 0, 1)
        sigma_b = pm.Normal("sigma_b", 0, 1)

        sigma = pm.Deterministic("sigma", pm.math.exp(sigma_a + sigma_b * x))

        pm.Normal(
            'y',
            mu=mu_log,
            sigma=sigma,
            observed=y_log
        )

        trace = pm.sample(
            1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True
        )

    post = az.extract(trace).to_dataframe()

    a_samps = post['a'].values[:, None]
    b_samps = post['b'].values[:, None]

    if label=='single':
        c_samps = post['c'].values[:, None]
        param_stats = post[['a', 'b', 'c', 'sigma']].agg(['mean', 'std']).T
        param_stats.columns = ['posterior_mean', 'posterior_std']
        print(param_stats)
        # Linear predictor in log space
        mu_log_new = a_samps + b_samps * x_new + c_samps * x_new ** 2

    else:
        param_stats = post[['a', 'b', 'sigma']].agg(['mean', 'std']).T
        param_stats.columns = ['posterior_mean', 'posterior_std']
        print(param_stats)
        # Linear predictor in log space
        mu_log_new = a_samps + b_samps * x_new

    # Correct LogNormal mean:
    # E[Y|x,a,b,sigma] = exp(mu + sigma^2/2)
    y_draws = np.exp(mu_log_new)
    y_mean = y_draws.mean(axis=0)
    lower = np.percentile(y_draws, 2.5, axis=0)
    upper = np.percentile(y_draws, 97.5, axis=0)

    return y_mean, lower, upper, x_new

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
         target_output, color, mean, lower, upper, label, x,  x_new):

    plt.scatter(x, df[target_output], color=color, alpha=0.5, s=80,label='Observed Data')
    # posterior mean
    plt.plot(x_new, mean, color=color, lw=2, label='Predicted Mean')
    # 95% band
    plt.fill_between(
        x_new,
        lower,
        upper,
        color=color, alpha=0.2, label='95% CI (±2σ)'
    )
    plt.ylabel(target_output, fontsize=18, fontweight='bold')
    if label=='single':
        plt.yscale('log')
        plt.ylim(10e-5,10e2)
    if label=='probability':
        plt.ylim(0,1)
    plt.grid(True, linestyle='--', alpha=0.4)
    if target_variable=='max_gust' or target_variable=='gust':
        plt.xlim(0,65)
    else:
        plt.xlim(0,120)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=16, loc='upper left')
    data_results = pd.DataFrame(columns=[target_variable, 'y_avg', 'y_lower', 'y_upper'])
    data_results[target_variable]=x_new
    data_results['y_avg']=mean
    data_results['y_lower']=lower
    data_results['y_upper']=upper
    if label=='probability':
        # fix columns to 1
        cols = ["y_avg", "y_lower", "y_upper"]
        data_results[cols] = data_results[cols].clip(upper=1)
    data_results.to_csv(f'results/bayesian_{label}_{state}_{county}_{target_variable}_{target_output}_spatiotemporal.csv')

    plt.xlabel(f'{target_variable}', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(fontsize=12)
    # plt.tight_layout()
    plt.savefig(f'results/bayesian_{label}_{state}_{county}_{target_variable}_{target_output}_spatiotemporal.png')
    plt.close()


def load_loglinear(state, county, df, color, weather_variable):
    target_variable = f'max_{weather_variable}'
    target_output = 'customer_hours_norm'
    if weather_variable=='gust':
        mask = df[target_variable] == 0
        df.loc[mask, target_variable] = df.loc[mask,'max_sknt']
    label='single'
    df[target_variable]=df[target_variable].round()
    # Replacing infinite with nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Dropping all the rows with nan values
    df.dropna(inplace=True)
    df=df.groupby(target_variable, as_index=False)[target_output].mean()
    if target_variable=='max_gust':
        df=df[df[target_variable]<=65]
        x_new = np.linspace(0, 65, 66)
    else:
        x_new = np.linspace(0, 120, 121)
    x = df[target_variable].values
    y=df[target_output].values
    mean, lower, upper, x_new = fit_loglinear(x,y,x_new, label)
    plot(df, state, county, target_variable, target_output, color, mean, lower, upper, label, x=x, x_new=x_new)
# load datasets for sigmoid fits

def load_sigmoid(state,county,color, weather_variable):
    df=pd.read_parquet(f'results/outage_probability_{state}_{county}_{weather_variable}.parquet')
    target_output='probability'
    label=target_output
    x = df[weather_variable].values
    y=df[target_output].values
    if weather_variable=='gust':
        x_new = np.linspace(0, 65, 66)
    else:
        x_new = np.linspace(0, 120, 121)
    mean, lower, upper, x_new = fit_loglinear(x,y,x_new, label)

    plot(df, state, county, weather_variable, target_output, color, mean, lower, upper, label, x=x, x_new=x_new)

def hat_graph(groups, setting):
    target_variable = 'max_gust'
    target_output = 'customer_hours_norm'

    fig, ax = plt.subplots(figsize=(15, 6))

    # ---------- Layout ----------
    state_spacing = 3.0  # distance between state groups
    bar_offset = 0.45  # separation within state
    bar_width = 0.70

    urban_positions = []
    rural_positions = []
    state_centers = []

    for i in range(len(groups)):
        center = i * state_spacing

        urban_positions.append(center - bar_offset)
        rural_positions.append(center + bar_offset)

        state_centers.append(center)

    # ---------- Alternate background shading ----------
    for i, center in enumerate(state_centers):
        if i % 2 == 0:
            ax.axvspan(
                center - 1.5,
                center + 1.5,
                color='k',
                alpha=0.025,
                zorder=0
            )

    # ---------- Plot ----------
    for i, (state, label, urban_color, rural_color) in enumerate(groups):
        urban_file = pd.read_csv(
            f"results/bayesian_{setting}_"
            f"{state}_{state}_rucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )

        rural_file = pd.read_csv(
            f"results/bayesian_{setting}_"
            f"{state}_{state}_nonrucc1_"
            f"{target_variable}_{target_output}_spatiotemporal.csv"
        )

        urban = urban_file.loc[
            urban_file[target_variable] == 20
            ].iloc[0]

        rural = rural_file.loc[
            rural_file[target_variable] == 20
            ].iloc[0]

        # Urban CI rectangle
        ax.bar(
            urban_positions[i],
            urban["y_upper"] - urban["y_lower"],
            width=bar_width,
            bottom=urban["y_lower"],
            color=urban_color,
            alpha=0.70,
            edgecolor='0.35',
            linewidth=0.8,
            zorder=3
        )

        # Rural CI rectangle
        ax.bar(
            rural_positions[i],
            rural["y_upper"] - rural["y_lower"],
            width=bar_width,
            bottom=rural["y_lower"],
            color=rural_color,
            alpha=0.70,
            edgecolor='0.35',
            linewidth=0.8,
            zorder=3
        )

        # Point estimate markers
        ax.hlines(
            urban["y_avg"],
            urban_positions[i] - bar_width * 0.28,
            urban_positions[i] + bar_width * 0.28,
            color='black',
            linewidth=2.0,
            zorder=4
        )

        ax.hlines(
            rural["y_avg"],
            rural_positions[i] - bar_width * 0.28,
            rural_positions[i] + bar_width * 0.28,
            color='black',
            linewidth=2.0,
            zorder=4
        )

    # ---------- Urban/Rural labels ----------
    bar_positions = []
    bar_labels = []

    for u, r in zip(urban_positions, rural_positions):
        bar_positions.extend([u, r])
        bar_labels.extend(["Urban", "Rural"])

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=10)

    # ---------- State labels beneath ----------
    secax = ax.secondary_xaxis('bottom')

    secax.set_xticks(state_centers)
    secax.set_xticklabels(
        [g[0] for g in groups],
        fontsize=11,
        fontweight='semibold'
    )

    secax.spines['bottom'].set_visible(False)
    secax.tick_params(
        axis='x',
        pad=28,
        length=0
    )

    # ---------- Labels ----------
    ax.set_ylabel(
        'Estimated Event Magnitude',
        fontsize=13
    )

    ax.set_xlabel('')

    # ---------- Grid ----------
    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.25,
        linewidth=0.8
    )

    ax.set_axisbelow(True)

    # ---------- Spines ----------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_color('0.3')
    ax.spines['bottom'].set_color('0.3')

    # ---------- Legend ----------
    legend_elements = [
        Patch(
            facecolor='0.5',
            alpha=0.8,
            edgecolor='0.35',
            label='Urban'
        ),
        Patch(
            facecolor='0.5',
            alpha=0.35,
            edgecolor='0.35',
            label='Rural'
        )
    ]

    ax.legend(
        handles=legend_elements,
        frameon=False,
        loc='upper left',
        ncol=2
    )

    plt.tight_layout()
    plt.show()
    # target_variable='max_gust'
    # target_output='customer_hours_norm'
    # fig, ax = plt.subplots(figsize=(14, 6))
    # x = np.arange(len(groups))
    # width=0.35
    #
    # for i, (state, label, urban_color, rural_color) in enumerate(groups):
    #     urban_file=pd.read_csv(f"results/bayesian_{setting}_"
    #         f"{state}_{state}_rucc1_"
    #         f"{target_variable}_{target_output}.csv"
    #     )
    #     rural_file=pd.read_csv(f"results/bayesian_{setting}_"
    #         f"{state}_{state}_nonrucc1_"
    #         f"{target_variable}_{target_output}.csv"
    #     )
    #     urban_at_20=urban_file[urban_file[target_variable]==20]
    #     rural_at_20=rural_file[rural_file[target_variable]==20]
    #     # Urban hat
    #     ax.bar(
    #         x[i] - width / 2,
    #         urban_at_20['y_upper'] - urban_at_20['y_lower'],
    #         width,
    #         bottom=urban_at_20['y_lower'],
    #         color=urban_color,
    #         edgecolor='dimgray',
    #         linewidth = 0.8,
    #         alpha=0.75
    #     )
    #
    #     # Rural hat
    #     ax.bar(
    #         x[i] + width / 2,
    #         rural_at_20['y_upper'] - rural_at_20['y_lower'],
    #         width,
    #         bottom=rural_at_20['y_lower'],
    #         color=rural_color,
    #         edgecolor='dimgray',
    #         linewidth=0.8,
    #         alpha=0.75
    #     )
    #
    #     # Estimate marker
    #     ax.hlines(
    #         urban_at_20['y_avg'],
    #         x[i] - width * 0.8,
    #         x[i] - width * 0.2,
    #         color='dimgray',
    #         linewidth=0.8
    #     )
    #
    #     ax.hlines(
    #         rural_at_20['y_avg'],
    #         x[i] + width * 0.2,
    #         x[i] + width * 0.8,
    #         color='dimgray',
    #         linewidth=0.8
    #     )
    #
    #
    # ax.set_xticks(x)
    # ax.set_xticklabels([g[0] for g in groups],
    #                    rotation=45,
    #                    ha='right')
    # ax.set_ylabel('Estimated Customer-Hour Losses (Normalized)')
    # ax.set_xlabel('State')
    # plt.tight_layout()
    # plt.show()

def plot_event_map():
    state='California'
    county='Los Angeles'
    ds=xr.open_dataset(f'../../merged_data/{state}/merged_data_{state}_{county}_2015_2025.nc')
    ds_selected = ds.where(ds.event_number_eaglei == 907, drop=True)
    url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
    counties = gpd.read_file(url)
    # King County
    king_county = counties[
        (counties["NAME"] == county) &
        (counties["STATE_NAME"] == state)
        ]

    # Neighboring counties
    neighbors = counties.cx[
        king_county.total_bounds[0] - 2: king_county.total_bounds[2] + 2,
        king_county.total_bounds[1] - 2: king_county.total_bounds[3] + 2
    ]

    # Build station dataframe
    stations_df = pd.DataFrame({
        "lon": ds_selected.station.lon.values,
        "lat": ds_selected.station.lat.values,
        "name": ds_selected.station.values,  # adjust if different
    }).drop_duplicates()

    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(
            stations_df.lon,
            stations_df.lat
        ),
        crs="EPSG:4326"
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Context counties
    neighbors.plot(
        ax=ax,
        color="#f2f2f2",
        edgecolor="lightgray",
        linewidth=0.5,
    )

    # King County fill
    king_county.plot(
        ax=ax,
        color="#F4CCCC",
        edgecolor="#B2182B",
        linewidth=2,
    )

    # Stations
    stations_gdf.plot(
        ax=ax,
        color="black",
        markersize=40,
        zorder=5,
    )

    # Labels
    for _, row in stations_gdf.iterrows():
        ax.annotate(
            row["name"],
            (row.geometry.x, row.geometry.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            zorder=6,
        )

    # Zoom
    xmin, ymin, xmax, ymax = king_county.total_bounds
    pad = 0.5

    ax.set_xlim(xmin - 1, xmax + 1)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    # Remove coordinate axes
    ax.set_axis_off()
    ax.set_title(
        "Weather Stations in Los Angeles County, CA",
        fontsize=16,
        pad=15
    )
    plt.tight_layout()
    plt.show()


groups=[
    ["Washington", "Washington_rucc1","#2166AC", "#D1E5F0"],
    ["Illinois", "Illinois_rucc1", "#54278F", "#DADAEB"],
    ["New York", "New York_rucc1", "#8C510A", "#DFC27D"],
    ["Massachusetts", "Massachusetts_rucc1", "#1B7837", "#D9F0D3"],
    ["California", "California_rucc1", "#B2182B", "#F4CCCC"],
    ["Arizona", "Arizona_rucc1", "#C51B7D", "#F4CAE4"],
    ["Texas", "Texas_rucc1", "#B8860B", "#F6E8A6"],
    ["Florida", "Florida_rucc1", "#01665E", "#C7EAE5"]
]
# plot_multiple_outage_probabilities_fig(groups)
# plot_multiple_outage_magnitudes_fig(groups)
# hat_graph(groups, setting='single')
# plot_event_map()