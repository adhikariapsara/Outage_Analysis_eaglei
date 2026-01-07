import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pymc as pm
import arviz as az

# inputs
state='Illinois'
county='Cook'
start='2018'
end='2024'
threshold=0.0001
# target variables: wind_speed, gust, precipitation, Air_temp, Air_temp_min
target_variable='max_gust'

#load datasets
df = pd.read_parquet(f'../Results/Outage_Events_Summary_All_{county}_{threshold}_{start}-{end}.parquet')
print(df.head())
print(df.columns)


df = df[df['max_gust']<60]
df[target_variable] = (df[target_variable] ).round()

# average all outage instances over their target weather variable
df_gust = df.groupby(target_variable).agg({
    
    'area_cost_out_h': 'mean',
    'cust_normalized':    'mean'
   
}).reset_index()

x     = df_gust[target_variable].values
x_new = np.linspace(x.min(), x.max(), 100)

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


# 3) Fit each series
mean_auc,   std_auc   = fit_loglinear(df_gust['area_cost_out_h'])
mean_cust,  std_cust  = fit_loglinear(df_gust['cust_normalized'])


# 4) Plot vertical subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 12), sharex=True)

panels = [
    ('Area Under Curve',      df_gust['area_cost_out_h'],   mean_auc,   std_auc,   'C0'),
    ('Customer Out(%) ',   df_gust['cust_normalized'],      mean_cust,  std_cust,  'C1'),
    
]

for ax, (title, y_data, y_mean, y_std, color) in zip(axes, panels):
    # raw data
    ax.scatter(x, y_data, color=color, alpha=0.5, s=80,label='Observed Data')
    # posterior mean
    ax.plot(   x_new, y_mean, color=color, lw=2, label='Predicted Mean')
    # 95% band
    ax.fill_between(
        x_new,
        y_mean - 2*y_std,
        y_mean + 2*y_std,
        color=color, alpha=0.2, label='95% CI (±2σ)'
    )
    ax.set_ylabel(title, fontsize=18, fontweight='bold')
#    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=16, loc='upper left')
    data_results = pd.DataFrame(columns=[target_variable, 'y_avg', 'y_lower', 'y_upper'])
    data_results[target_variable]=x_new
    data_results['y_avg']=y_mean
    data_results['y_lower']=y_mean - 2*y_std
    data_results['y_upper']=y_mean + 2*y_std
    data_results.to_csv(f'exp_fit/{state}_{county}_{title}_{target_variable}_fit.csv')

axes[-1].set_xlabel(f'{target_variable}', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(fontsize=12)
plt.tight_layout()
#plt.show()
plt.savefig(f'../Results/Bayesian_{target_variable}_{county}.png')

