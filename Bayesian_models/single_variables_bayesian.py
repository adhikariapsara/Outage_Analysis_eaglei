import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pymc as pm
import arviz as az

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

        trace = pm.sample(1000, tune=1000, chains=2, cores=1,
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


def plot(df, state, county, target_variable, target_output, color, mean, std):
    # 4) Plot vertical subplots
    #fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

    panel = [
        (target_output, df[target_output],   mean,   std,   color),
    ]

  #  for ax, (title, y_data, y_mean, y_std, color) in zip(axes, panels):
    # raw data
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
#    ax.set_yscale('log')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=16, loc='upper left')
    data_results = pd.DataFrame(columns=[target_variable, 'y_avg', 'y_lower', 'y_upper'])
    data_results[target_variable]=x_new
    data_results['y_avg']=mean
    data_results['y_lower']=mean - 2*std
    data_results['y_upper']=mean + 2*std
    data_results.to_csv(f'exp_fit/{state}_{county}_{target_output}_{target_variable}_fit.csv')

    plt.xlabel(f'{target_variable}', fontsize=18, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'../Results/Bayesian_{target_variable}_{target_output}_{county}.png')


# inputs
state = 'Arizona'
county = 'Maricopa'
color = 'magenta'
start = '2018'
end = '2024'
approach = 'percentile'
value = 0.75
# target variables: gust_max, precipitation, Air_temp_max, Air_temp_min
target_variable = 'Air_temp_max'
# target outputs: cust_out_max, cust_normalized, area
target_output = 'cust_out_max'

# load datasets
df = pd.read_parquet(f'../Results/Outage_Events_Summary_All_{county}_{approach}_{value}_{start}-{end}.parquet')
print(df.head())
print(df.columns)

df = df[df['Air_temp_max'] > 60]
df[target_variable] = myround(df[target_variable],base=1)
# average all outage instances over their target weather variable
df_grouped = df.groupby(target_variable).agg({
    target_output: 'mean'
}).reset_index()

x = df_grouped[target_variable].values
x_new = np.linspace(x.min(), x.max(), 100)

# 3) Fit each series
mean,  std  = fit_loglinear(df_grouped[target_output])
plot(df_grouped, state, county, target_variable, target_output, color, mean, std)