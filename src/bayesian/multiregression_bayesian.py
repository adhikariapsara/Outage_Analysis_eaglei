import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def fit_and_plot_interaction_contour(df_sub, ax, title):
    # Extract inputs and log‐target
    gust      = df_sub['max_gust'].values
    temp      = df_sub['max_tmpf'].values
    ppocc_val = df_sub['poccurence'].iloc[0]
    y_log     = np.log(df_sub['num_outages'].values + 1e-6).reshape(-1,1)

    # Standardize
    scaler_x = StandardScaler().fit(np.column_stack((gust, temp)))
    scaler_y = StandardScaler().fit(y_log)
    Xs       = scaler_x.transform(np.column_stack((gust, temp)))
    x1_s, x2_s = Xs[:,0], Xs[:,1]
    y_s        = scaler_y.transform(y_log).flatten()

    # Fit Bayesian interaction model
    with pm.Model() as model:
        a      = pm.Normal("a", mu=0, sigma=5)
        b1     = pm.Normal("b_gust", mu=0, sigma=5)
        b2     = pm.Normal("b_temp", mu=0, sigma=5)
       
        sigma  = pm.HalfNormal("sigma", sigma=1)
        mu_log = a + b1*x1_s + b2*x2_s 
        pm.Normal("y_obs", mu=mu_log, sigma=sigma, observed=y_s)
        trace = pm.sample(1000, tune=1000, chains=4, cores=1,
                          random_seed=42, return_inferencedata=True)

    # Extract posterior samples
    post   = az.extract(trace).to_dataframe()
    a_s     = post['a'].values[:,None]
    b1_s    = post['b_gust'].values[:,None]
    b2_s    = post['b_temp'].values[:,None]
    

    # Create a grid in original space
    gusts = np.linspace(gust.min(), gust.max(), 50)
    temps = np.linspace(temp.min(), temp.max(), 50)
    G, T  = np.meshgrid(gusts, temps)

    # Standardize the grid
    grid_s = scaler_x.transform(np.column_stack((G.ravel(), T.ravel())))
    Gs = grid_s[:,0].reshape(G.shape)
    Ts = grid_s[:,1].reshape(T.shape)

    # Compute posterior‐mean prediction in log‐space
    mu_log_samps = (
        a_s
        + b1_s    * Gs.ravel()[None,:]
        + b2_s    * Ts.ravel()[None,:]
        
        
    )
    mu_log_mean = mu_log_samps.mean(axis=0)

    # Back‐transform to original scale
    mu_log_orig = scaler_y.inverse_transform(mu_log_mean[:,None]).flatten()
    Y_pred      = np.exp(mu_log_orig).reshape(G.shape)

    # Plot contours
    cf = ax.contourf(
        G, T, Y_pred,
        levels=20, cmap='viridis', alpha=0.8
    )
    ax.contour(G, T, Y_pred, levels=10, colors='k', linewidths=0.5)
    ax.scatter(
        gust, temp,
        c='black', edgecolor='k',
        s=80, alpha=0.6
    )
    label = "Precipitation" if ppocc_val == 1 else "No precipitation"
    ax.set_title(label,
                 fontsize=25, fontname='Times New Roman')

  
    ax.set_xlabel("Maximum Gust (mph)", fontsize=25, fontname='Times New Roman')
    ax.set_ylabel("Maximum Air Temp (°F)", fontsize=25, fontname='Times New Roman')
    return cf

# ——— Main ———
state='Washington'
county='Clallam'
# threshold=0.001
# start=2018
# end=2024
df = pd.read_parquet(f'../../merged_data/{state}/events_stats_{state}_{county}.parquet')

# two dataframes, representing events with & without precip
df0 = df[df['poccurence']==0]
df1 = df[df['poccurence']==1]

# Two vertically stacked subplots
# Create two independent subplots (no shared x-axis)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))  # removed sharex=True

# Top subplot: No precipitation
cf0 = fit_and_plot_interaction_contour(df0, axes[0], title="Dry")
cbar0 = fig.colorbar(cf0, ax=axes[0], shrink=0.8, pad=0.02)
cbar0.set_label("Number of Outages", fontsize=20)
cbar0.ax.tick_params(labelsize=20)
cbar0.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
# Bottom subplot: Precipitation
cf1 = fit_and_plot_interaction_contour(df1, axes[1], title="Wet")
cbar1 = fig.colorbar(cf1, ax=axes[1], shrink=0.8, pad=0.02)
cbar1.set_label("Number of Outages", fontsize=20)
cbar1.ax.tick_params(labelsize=20)
cbar1.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
# Increase xtick font size and style for bottom subplot
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_fontsize(25)
        label.set_fontname('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontsize(25)
        label.set_fontname('Times New Roman')

plt.tight_layout()
fig.subplots_adjust(hspace = 0.4,bottom=0.1)
plt.show()