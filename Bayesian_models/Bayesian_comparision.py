import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 

# Load and filter data
df = pd.read_excel('Outage_Events_Summary_All_Cook_gust_Modified_SH_all_5000_2018-2022_all_weather.xlsx')
df = df[df['out_duration_max'] < 24]

# Prepare inputs
y = df['cust_normalized'].values.reshape(-1, 1)  # reshape for scaler
x2 = df['Air_temp'].values.reshape(-1, 1)
x1 = df['gust'].values.reshape(-1, 1)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(np.hstack((x1, x2)))
y_scaled = scaler_y.fit_transform(y).flatten()  # flatten y after scaling

x1_s, x2_s = x_scaled[:, 0], x_scaled[:, 1]

# ----------------- Model 1: y = a1 * exp(b1*x1) + a2 *exp( b2*x2) + c2 -----------------
with pm.Model() as model1:
    a1 = pm.Normal("a1", mu=1, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10)

    a2 = pm.Normal("a2", mu=1, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10)

    c = pm.Normal("c", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Model equation
    mu1 = a1 * pm.math.exp(b1 * x1_s) + a2 * pm.math.exp(b2 * x2_s) + c

    # Likelihood
    y_obs1 = pm.Normal("y_obs", mu=mu1, sigma=sigma, observed=y_scaled)

    # Sampling
    trace1 = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=1,
        target_accept=0.95,
        nuts={"max_treedepth": 15},
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )


# ----------------- Model 2: log(y) = log(a) + b1*x1 + b2*x2 + c -----------------
# Apply log and standardize y again
y_log = np.log(df['cust_normalized'].values + 1e-6).reshape(-1, 1)
y_log_s = scaler_y.fit_transform(y_log).flatten()

with pm.Model() as model2:
    log_a = pm.Normal("log_a", mu=0, sigma=5)
    b1 = pm.Normal("b1", mu=0, sigma=5)
    b2 = pm.Normal("b2", mu=0, sigma=5)
    c2 = pm.Normal("c2", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu_log = log_a + b1 * x1_s + b2 * x2_s + c2
    y_obs2 = pm.Normal("y_obs", mu=mu_log, sigma=sigma, observed=y_log_s)

    trace2 = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=1,
        target_accept=0.95,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )
#------ Model 3 = gust only model 
with pm.Model() as model3b:
    log_a = pm.Normal("log_a", mu=0, sigma=5)
    b1 = pm.Normal("b1", mu=0, sigma=5)
    c = pm.Normal("c", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=1)

    mu_log3b = log_a + b1 * x1_s + c
    y_obs3b = pm.Normal("y_obs", mu=mu_log3b, sigma=sigma, observed=y_log_s)

    trace3b = pm.sample(
        1000,
        tune=1000,
        chains=4,
        cores=1,
        target_accept=0.95,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )
loo1 = az.loo(trace1)
loo2 = az.loo(trace2)
loo3 = az.loo(trace3b)

# Bayes Factor calculations
bf_12 = np.exp(0.5 * (loo1.elpd_loo - loo2.elpd_loo))
bf_13 = np.exp(0.5 * (loo1.elpd_loo - loo3.elpd_loo))
bf_23 = np.exp(0.5 * (loo2.elpd_loo - loo3.elpd_loo))

# Print results
print(f"Bayes Factor (Model 1 vs Model 2): {bf_12:.3f}")
print(f"Bayes Factor (Model 1 vs Model 3): {bf_13:.3f}")
print(f"Bayes Factor (Model 2 vs Model 3): {bf_23:.3f}")



# Extract posterior samples from trace1
posterior_samples1 = az.extract(trace1)

a1_samples = posterior_samples1["a1"].values[:, np.newaxis, np.newaxis]
b1_samples = posterior_samples1["b1"].values[:, np.newaxis, np.newaxis]
a2_samples = posterior_samples1["a2"].values[:, np.newaxis, np.newaxis]
b2_samples = posterior_samples1["b2"].values[:, np.newaxis, np.newaxis]
c_samples = posterior_samples1["c"].values[:, np.newaxis, np.newaxis]

# Create meshgrid in standardized space
x1_range_std = np.linspace(min(x1_s), max(x1_s), 30)
x2_range_std = np.linspace(min(x2_s), max(x2_s), 30)
X1_s, X2_s = np.meshgrid(x1_range_std, x2_range_std)

# Compute predicted standardized y from posterior samples
Y_samples_scaled = a1_samples * np.exp(b1_samples * X1_s) + a2_samples * np.exp(b2_samples * X2_s) + c_samples

# Summary statistics
Y_scaled_mean = np.mean(Y_samples_scaled, axis=0)
Y_scaled_upper = np.percentile(Y_samples_scaled, 97.5, axis=0)
Y_scaled_lower = np.percentile(Y_samples_scaled, 2.5, axis=0)

# Inverse-transform y from standardized to original scale
Y_mean = Y_scaled_mean * scaler_y.scale_ + scaler_y.mean_
Y_upper = Y_scaled_upper * scaler_y.scale_ + scaler_y.mean_
Y_lower = Y_scaled_lower * scaler_y.scale_ + scaler_y.mean_

# Inverse-transform x1 and x2 meshgrid
X1_flat = X1_s.flatten()
X2_flat = X2_s.flatten()
X_orig = scaler_x.inverse_transform(np.column_stack((X1_flat, X2_flat)))
X1_orig = X_orig[:, 0].reshape(X1_s.shape)
X2_orig = X_orig[:, 1].reshape(X2_s.shape)

# Observed data (inverse transform)
x_orig_full = scaler_x.inverse_transform(np.column_stack((x1_s, x2_s)))
x1_data_clean = x_orig_full[:, 0]
x2_data_clean = x_orig_full[:, 1]
y_data_clean = y_scaled * scaler_y.scale_ + scaler_y.mean_

# Create 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Observed data
ax.scatter(x1_data_clean, x2_data_clean, y_data_clean, color='black', alpha=0.5, label='Observed Data')

# Prediction surface (median)
ax.plot_surface(X1_orig, X2_orig, Y_mean, color='blue', alpha=0.5, edgecolor='none', label='Posterior Mean')

# Optionally plot uncertainty surfaces
# ax.plot_surface(X1_orig, X2_orig, Y_upper, color='skyblue', alpha=0.3, edgecolor='none')
# ax.plot_surface(X1_orig, X2_orig, Y_lower, color='skyblue', alpha=0.3, edgecolor='none')

ax.set_xlabel("Gust")
ax.set_ylabel("Air Temperature")
ax.set_zlabel("Customer Outage (cust_normalized)")
ax.set_title("Model 1: Posterior Mean Surface (Bayesian Exponential Additive Model)")
ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()
posterior_samples2 = az.extract(trace2)

log_a_samples = posterior_samples2["log_a"].values[:, np.newaxis, np.newaxis]
b1_samples = posterior_samples2["b1"].values[:, np.newaxis, np.newaxis]
b2_samples = posterior_samples2["b2"].values[:, np.newaxis, np.newaxis]
c_samples = posterior_samples2["c2"].values[:, np.newaxis, np.newaxis]


# --- Create meshgrid using standardized x1_s, x2_s ranges ---
x1_range_std = np.linspace(min(x1_s), max(x1_s), 30)
x2_range_std = np.linspace(min(x2_s), max(x2_s), 30)
X1_s, X2_s = np.meshgrid(x1_range_std, x2_range_std)

# --- Predict log(y) in standardized space ---
Y_log_samples = log_a_samples + b1_samples * X1_s + b2_samples * X2_s + c_samples

# --- Summary statistics in log space ---
Y_log_mean = np.mean(Y_log_samples, axis=0)
Y_log_upper = np.percentile(Y_log_samples, 97.5, axis=0)
Y_log_lower = np.percentile(Y_log_samples, 2.5, axis=0)

# --- Inverse transform y: first unscale log(y), then exp() ---
Y_log_mean_unscaled = Y_log_mean * scaler_y.scale_ + scaler_y.mean_
Y_log_upper_unscaled = Y_log_upper * scaler_y.scale_ + scaler_y.mean_
Y_log_lower_unscaled = Y_log_lower * scaler_y.scale_ + scaler_y.mean_

Y_mean = np.exp(Y_log_mean_unscaled)
Y_upper = np.exp(Y_log_upper_unscaled)
Y_lower = np.exp(Y_log_lower_unscaled)

# --- Inverse transform x1 and x2 from meshgrid (reshape needed) ---
X1_flat = X1_s.flatten()
X2_flat = X2_s.flatten()
X_orig = scaler_x.inverse_transform(np.column_stack((X1_flat, X2_flat)))
X1_orig = X_orig[:, 0].reshape(X1_s.shape)
X2_orig = X_orig[:, 1].reshape(X2_s.shape)

# --- Observed data (unscale x and y) ---
x_orig_full = scaler_x.inverse_transform(np.column_stack((x1_s, x2_s)))
x1_data_clean = x_orig_full[:, 0]
x2_data_clean = x_orig_full[:, 1]
y_data_clean = np.exp(y_log_s * scaler_y.scale_ + scaler_y.mean_)

# --- 3D Plot ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter observed data
ax.scatter(x1_data_clean, x2_data_clean, y_data_clean, color='black', alpha=0.6, label='Observed Data')

# Plot surfaces
ax.plot_surface(X1_orig, X2_orig, Y_mean, color='blue', alpha=0.5, edgecolor='none', label='Mean Prediction')
#ax.plot_surface(X1_orig, X2_orig, Y_upper, color='lightblue', alpha=0.3, edgecolor='none')
#ax.plot_surface(X1_orig, X2_orig, Y_lower, color='lightblue', alpha=0.3, edgecolor='none')

# Labels
ax.set_xlabel("Gust")
ax.set_ylabel("Air temperature")
ax.set_zlabel("Customer Outage (cust_normalized)")
ax.set_title("Model 2: 3D Bayesian Surface with Uncertainty (Original Scale)")
ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()


# Extract posterior samples
posterior_samples3b = az.extract(trace3b)

log_a_samples = posterior_samples3b["log_a"].values[:, np.newaxis]
b1_samples = posterior_samples3b["b1"].values[:, np.newaxis]
c_samples = posterior_samples3b["c"].values[:, np.newaxis]

# Create test range for x1 (gust, standardized)
x1_range_std = np.linspace(min(x1_s), max(x1_s), 100)

# Predict log(y) from posterior samples
mu_log_samples = log_a_samples + b1_samples * x1_range_std + c_samples

# Convert from log-scale to original y-scale (standardized)
y_scaled_samples = np.exp(mu_log_samples)

# Convert standardized y to original y
y_pred_samples = y_scaled_samples * scaler_y.scale_ + scaler_y.mean_

# Compute summary statistics
y_median = np.median(y_pred_samples, axis=0)
y_lower = np.percentile(y_pred_samples, 2.5, axis=0)
y_upper = np.percentile(y_pred_samples, 97.5, axis=0)

# Inverse transform x1 from standardized to original scale
x1_range_orig = scaler_x.inverse_transform(np.column_stack((x1_range_std, np.zeros_like(x1_range_std))))[:, 0]

# Original observed data
x1_data_orig = scaler_x.inverse_transform(np.column_stack((x1_s, x2_s)))[:, 0]
y_data_orig = np.exp(y_log_s * scaler_y.scale_ + scaler_y.mean_)

# --- Plot ---
plt.figure(figsize=(10, 6))

# Credible interval
plt.fill_between(x1_range_orig, y_lower, y_upper, color='skyblue', alpha=0.4, label='95% Credible Interval')

# Posterior median
plt.plot(x1_range_orig, y_median, color='blue', label='Posterior Median')

# Observed data
plt.scatter(x1_data_orig, y_data_orig, color='black', alpha=0.6, label='Observed Data', s=20)

plt.xlabel("Gust")
plt.ylabel("Customer Outage (cust_normalized)")
plt.title("Model 3: 2D Posterior Prediction with 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
