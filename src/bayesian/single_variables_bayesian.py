import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pymc as pm
import arviz as az
import xarray as xr
import pytensor.tensor as pt

def myround(x, base):
    return base * round(x/base)

# 2) Helper to fit and predict
def fit_loglinear(y_obs):
    y_log = np.log(y_obs + 1e-10)
    with pm.Model() as m:
        a_log = pm.Uniform('a_log', lower=-10, upper=100)
        b     = pm.Uniform('b', lower=0, upper=100)
        c     = pm.Uniform('c', lower=-10, upper=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        mu_log = a_log + b * x + c
        pm.Normal('y', mu=mu_log, sigma=sigma, observed=y_log)

        trace = pm.sample(1000, tune=1000, chains=4, cores=1,
                          target_accept=0.95,
                          random_seed=42, return_inferencedata=True)

    post = az.extract(trace).to_dataframe()
    a_samps = post['a_log'].values[:, None]
    b_samps = post['b'].values[:, None]
    c_samps = post['c'].values[:, None]
    param_stats = post[['a_log', 'b', 'c']].agg(['mean', 'std']).T
    param_stats.columns = ['posterior_mean', 'posterior_std']
    print(param_stats)

    mu_log_new = a_samps + b_samps * x_new + c_samps
    y_new = np.exp(mu_log_new)
    return y_new.mean(axis=0), y_new.std(axis=0)


def fit_sigmoid(y_obs):
    with pm.Model() as m:
        with pm.Model() as m:
            a = pm.Uniform('a', lower=0, upper=1) # saturation level
            k = pm.Uniform('k', lower=0, upper=10) # slope
            w0 = pm.Uniform('w0', lower=0, upper=100) # midpoint
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = a * pm.math.sigmoid(k * (x - w0))
            pm.Normal('y', mu=mu, sigma=sigma, observed=y_obs)
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

    return mu_new.mean(axis=0), mu_new.std(axis=0)

def plot(df, state, county, target_variable, target_output, color, mean, std):

    plt.scatter(x, df[target_output], color=color, alpha=0.5, s=80,label='Observed Data')
    # posterior mean
    plt.plot(   x_new, mean, color=color, lw=2, label='Predicted Mean')
    # 95% band
    plt.fill_between(
        x_new,
        mean - 2*std,
        mean + 2*std,
        color=color, alpha=0.2, label='95% CI (±2σ)'
    )
    plt.ylabel(target_output, fontsize=18, fontweight='bold')
 #   plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=16, loc='upper left')
    data_results = pd.DataFrame(columns=[target_variable, 'y_avg', 'y_lower', 'y_upper'])
    data_results[target_variable]=x_new
    data_results['y_avg']=mean
    data_results['y_lower']=mean - 2*std
    data_results['y_upper']=mean + 2*std
    data_results.to_csv(f'results/bayesian_probability_{state}_{county}_{target_variable}_{target_output}.csv')

    plt.xlabel(f'{target_variable}', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/bayesian_probability_{state}_{county}_{target_variable}_{target_output}.png')


# inputs
state = 'California'
county = 'Los Angeles'
color = 'red'

target_variable = 'gust'
target_output='outage'

# load datasets for loglinear fits
# df = pd.read_parquet(f'../../events_stats/county_events/{state}/{county}_event_stats.parquet')
# mask = df[target_variable] == 0
# df.loc[mask, target_variable] = df.loc[mask,'max_sknt']
# df = df[df['max_gust'] <= 65]
# df[target_variable] = myround(df[target_variable],base=1)
# # average all outage instances over their target weather variable
# df_grouped = df.groupby(target_variable).agg({
#     target_output: 'mean'
# }).reset_index()


# load datasets for sigmoid fits
ds_merged=xr.open_dataset(f'../../merged_data/{state}/merged_data_{state}_{county}_2014_2024.nc')
ds_merged = ds_merged.sel(station=ds_merged.station != "SDB")
df=ds_merged[[target_variable,'event_number_eaglei', 'sknt']].to_dataframe()
mask = df[target_variable] == 0
df.loc[mask, target_variable] = df.loc[mask,'sknt']
df[target_variable] = myround(df[target_variable],1)
df = df[df['gust'] < 65]
df[target_output]=(df['event_number_eaglei'] > 0).astype(int)
df_grouped = (df.groupby([target_variable], as_index=False).agg(
    freq=(target_variable, 'count'),
    outage=('outage','mean')
))

x = df_grouped[target_variable].values
x_new = np.linspace(0, 65, 66)


# 3) Fit each series
mean, std = fit_sigmoid(df_grouped[target_output])
#mean,  std  = fit_loglinear(df_grouped[target_output])
plot(df_grouped, state, county, target_variable, target_output, color, mean, std)