import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import requests
import warnings
import gc 
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & DATA ACQUISITION
# ==========================================
print("Fetching 4-year dataset from NASA POWER API...")

def fetch_nasa_power_data(lat, lon, start_date, end_date):
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame.from_dict(data['properties']['parameter']['ALLSKY_SFC_SW_DWN'], orient='index', columns=['GHI'])
        df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
        return df[df['GHI'] >= 0] 
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

df_ghi = fetch_nasa_power_data(12.9716, 77.5946, "20220101", "20251231")

print("Fitting Beta Distributions...")
df_ghi['Hour'] = df_ghi.index.hour
ghi_max = df_ghi['GHI'].max()
df_ghi['R'] = df_ghi['GHI'] / ghi_max 
beta_params = {}

for hour in range(24):
    hour_data = df_ghi[(df_ghi['Hour'] == hour) & (df_ghi['R'] > 0.01)]['R']
    hour_data = np.clip(hour_data, a_min=None, a_max=0.9999)
    if len(hour_data) > 10:
        a, b, loc, scale = stats.beta.fit(hour_data, floc=0, fscale=1)
        beta_params[hour] = {'alpha': a, 'beta': b}
    else:
        beta_params[hour] = {'alpha': 0, 'beta': 0}

# ==========================================
# 2. SCENARIO CONFIGURATION
# ==========================================
PV_AREA = 10
BATT_CAP = 2.4
DAILY_LOAD = 6
EFFICIENCY = 0.18
PERFORMANCE_RATIO = 0.75
HOURS_IN_YEAR = len(df_ghi)

# Generate 4-year load matrix
weights = np.array([0.5]*6 + [2.0, 2.5, 2.0] + [0.5]*8 + [1.5, 2.5, 3.0, 2.5, 1.5] + [0.5]*2)
base_daily_profile = (weights / weights.sum()) * DAILY_LOAD
base_load_array = np.tile(base_daily_profile, int(HOURS_IN_YEAR / 24) + 1)[:HOURS_IN_YEAR]

# ==========================================
# 3. THE CONVERGENCE LOOP
# ==========================================
iterations_to_test = [50, 100, 250, 500, 1000, 2500, 5000]
lpsp_results = []

print(f"\nStarting Convergence Test across {HOURS_IN_YEAR} hours...")
print("-" * 50)

for N in iterations_to_test:
    print(f"Running simulation for N = {N} iterations (Please wait...)")
    np.random.seed(42) 
    
    # Generate Matrices
    sim_GHI_urban = np.zeros((N, HOURS_IN_YEAR))
    sim_PV_power = np.zeros((N, HOURS_IN_YEAR))
    smf_matrix = np.ones((N, HOURS_IN_YEAR))
    
    for i, hour in enumerate(df_ghi['Hour']):
        if 6 <= hour < 10:
            smf_matrix[:, i] = np.random.uniform(0.65, 0.80, N)
        elif 10 <= hour < 15:
            smf_matrix[:, i] = np.random.uniform(0.85, 0.95, N)
        elif 15 <= hour <= 18:
            smf_matrix[:, i] = np.random.uniform(0.70, 0.85, N)
            
        params = beta_params[hour]
        if params['alpha'] > 0:
            R_sampled = stats.beta.rvs(params['alpha'], params['beta'], size=N)
            sim_GHI_urban[:, i] = (R_sampled * ghi_max) * smf_matrix[:, i]
            sim_PV_power[:, i] = (sim_GHI_urban[:, i] * PV_AREA * EFFICIENCY * PERFORMANCE_RATIO) / 1000

    load_matrix = np.tile(base_load_array, (N, 1))
    load_matrix += np.random.normal(0, load_matrix.mean() * 0.1, (N, HOURS_IN_YEAR))
    load_matrix = np.maximum(load_matrix, 0)
    
    battery_soc = np.full(N, BATT_CAP * 0.5) 
    deficit_matrix = np.zeros((N, HOURS_IN_YEAR))
    
    for h in range(HOURS_IN_YEAR):
        balance = load_matrix[:, h] - sim_PV_power[:, h]
        actual_batt_power = np.zeros(N)
        
        discharge_mask = balance > 0
        actual_batt_power[discharge_mask] = np.minimum(balance[discharge_mask], battery_soc[discharge_mask])
        
        charge_mask = balance < 0
        space_left = BATT_CAP - battery_soc[charge_mask]
        actual_batt_power[charge_mask] = np.maximum(balance[charge_mask], -space_left)
        
        battery_soc -= actual_batt_power
        deficit_matrix[:, h] = np.maximum(0, balance - actual_batt_power)
        
    # Calculate LPSP for this iteration step
    total_deficit = np.sum(deficit_matrix, axis=1)
    total_load = np.sum(load_matrix, axis=1)
    mean_lpsp = np.mean(total_deficit / total_load) * 100 
    
    lpsp_results.append(mean_lpsp)
    print(f"-> Result for N={N}: LPSP = {mean_lpsp:.4f}%")
    
    # Free RAM before the next bigger loop starts
    del sim_GHI_urban, sim_PV_power, smf_matrix, load_matrix, deficit_matrix, battery_soc
    gc.collect()

print("-" * 50)
print("Convergence Test Complete!")

# ==========================================
# 4. PLOT CONVERGENCE GRAPH
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(iterations_to_test, lpsp_results, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
plt.axhline(y=lpsp_results[-1], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Convergence Asymptote (~{lpsp_results[-1]:.4f}%)')

plt.title('Monte-Carlo Convergence Test (4-Year Dataset)')
plt.xlabel('Number of Iterations (N)')
plt.ylabel('Mean LPSP (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.savefig('MC_Convergence_Graph.png', dpi=300)
plt.show()