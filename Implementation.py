import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
import requests
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PHASE 1: DATA ACQUISITION & PREPROCESSING
# ==========================================

def fetch_nasa_power_data(lat, lon, start_date, end_date):
    print("Fetching data from NASA POWER API")
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
        df.index.name = 'Datetime'
        df = df[df['GHI'] >= 0] 
        print("Data successfully fetched and cleaned!")
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

lat, lon = 12.9716, 77.5946
df_ghi = fetch_nasa_power_data(lat, lon, "20220101", "20251231")

# ==========================================
# PHASE 2 & 3: NORMALIZATION & BETA FITTING
# ==========================================

print("## Normalizing Irradiance and Fitting Beta Distributions ##")
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
# PHASE 4: LOAD PROFILES & SCENARIO CONFIG
# ==========================================

def generate_base_load_profile(profile_type, daily_load_kwh):
    if profile_type == 'Residential':
        weights = [0.5]*6 + [2.0, 2.5, 2.0] + [0.5]*8 + [1.5, 2.5, 3.0, 2.5, 1.5] + [0.5]*2
    elif profile_type == 'Commercial':
        weights = [0.2]*8 + [1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.5, 1.5] + [0.2]*6
    else:
        weights = [1.0] * 24
    weights = np.array(weights)
    return (weights / weights.sum()) * daily_load_kwh

SCENARIOS = {
    'A': {'name': 'Budget Setup', 'pv_area': 10, 'batt_cap': 2.4, 'daily_load': 6, 'type': 'Residential'},
    'B': {'name': 'Standard Modern Home', 'pv_area': 20, 'batt_cap': 5.0, 'daily_load': 8, 'type': 'Residential'},
    'C': {'name': 'Future-Proof Premium', 'pv_area': 40, 'batt_cap': 10.0, 'daily_load': 12, 'type': 'Residential'},
    'D': {'name': 'Commercial IT Office', 'pv_area': 300, 'batt_cap': 100.0, 'daily_load': 250, 'type': 'Commercial'}
}

# ---------------------------------------------------------
# SELECT YOUR SCENARIO HERE ('A', 'B', 'C', or 'D')
# ---------------------------------------------------------
SELECTED_SCENARIO = 'D'  
config = SCENARIOS[SELECTED_SCENARIO]

print(f"\n--- RUNNING SCENARIO {SELECTED_SCENARIO}: {config['name']} ---")
print(f"PV Area: {config['pv_area']} m² | Battery: {config['batt_cap']} kWh | Load: {config['daily_load']} kWh/day")

# ==========================================
# PHASE 5: MONTE-CARLO SIMULATION SETUP
# ==========================================

np.random.seed(42) # Locks random dice rolls for academic reproducibility

N_ITERATIONS = 2500
HOURS_IN_YEAR = len(df_ghi)
EFFICIENCY = 0.18
PERFORMANCE_RATIO = 0.75

print(f"Initializing Monte-Carlo Simulation ({N_ITERATIONS} iterations)")

sim_GHI_urban = np.zeros((N_ITERATIONS, HOURS_IN_YEAR))
sim_PV_power = np.zeros((N_ITERATIONS, HOURS_IN_YEAR))

smf_matrix = np.ones((N_ITERATIONS, HOURS_IN_YEAR))
for i, hour in enumerate(df_ghi['Hour']):
    if 6 <= hour < 10:
        smf_matrix[:, i] = np.random.uniform(0.65, 0.80, N_ITERATIONS)
    elif 10 <= hour < 15:
        smf_matrix[:, i] = np.random.uniform(0.85, 0.95, N_ITERATIONS)
    elif 15 <= hour <= 18:
        smf_matrix[:, i] = np.random.uniform(0.70, 0.85, N_ITERATIONS)

for i, hour in enumerate(df_ghi['Hour']):
    params = beta_params[hour]
    if params['alpha'] > 0:
        R_sampled = stats.beta.rvs(params['alpha'], params['beta'], size=N_ITERATIONS)
        GHI_sampled = R_sampled * ghi_max
        sim_GHI_urban[:, i] = GHI_sampled * smf_matrix[:, i]
        sim_PV_power[:, i] = (sim_GHI_urban[:, i] * config['pv_area'] * EFFICIENCY * PERFORMANCE_RATIO) / 1000
    else:
        sim_GHI_urban[:, i] = 0
        sim_PV_power[:, i] = 0

# ==========================================
# PHASE 6: TIME-SERIES ENERGY & BATTERY BALANCE
# ==========================================
print("Evaluating Realistic Load and Battery State of Charge (SoC)...")

base_daily_profile = generate_base_load_profile(config['type'], config['daily_load'])
load_matrix = np.tile(base_daily_profile, int(HOURS_IN_YEAR / 24) + 1)[:HOURS_IN_YEAR]
load_matrix = np.tile(load_matrix, (N_ITERATIONS, 1))
load_matrix += np.random.normal(0, load_matrix.mean() * 0.1, (N_ITERATIONS, HOURS_IN_YEAR))
load_matrix = np.maximum(load_matrix, 0) 

battery_capacity = config['batt_cap']
battery_soc = np.full(N_ITERATIONS, battery_capacity * 0.5) 

deficit_matrix = np.zeros((N_ITERATIONS, HOURS_IN_YEAR))
surplus_matrix = np.zeros((N_ITERATIONS, HOURS_IN_YEAR))

for h in range(HOURS_IN_YEAR):
    balance = load_matrix[:, h] - sim_PV_power[:, h]
    actual_batt_power = np.zeros(N_ITERATIONS)
    
    discharge_mask = balance > 0
    actual_batt_power[discharge_mask] = np.minimum(balance[discharge_mask], battery_soc[discharge_mask])
    
    charge_mask = balance < 0
    space_left = battery_capacity - battery_soc[charge_mask]
    actual_batt_power[charge_mask] = np.maximum(balance[charge_mask], -space_left)
    
    battery_soc -= actual_batt_power
    
    deficit_matrix[:, h] = np.maximum(0, balance - actual_batt_power)
    surplus_matrix[:, h] = np.maximum(0, -(balance - actual_batt_power))

# ==========================================
# PHASE 7: RELIABILITY METRICS & ENERGY TOTALS
# ==========================================

print("## Calculating Reliability Metrics & Energy Totals ##")

total_deficit_per_sim = np.sum(deficit_matrix, axis=1)
total_load_per_sim = np.sum(load_matrix, axis=1)
lpsp_array = total_deficit_per_sim / total_load_per_sim
mean_lpsp = np.mean(lpsp_array)

hours_with_deficit = np.sum(deficit_matrix > 0.01, axis=1) 
prob_deficit_array = hours_with_deficit / HOURS_IN_YEAR
mean_prob_deficit = np.mean(prob_deficit_array)

total_surplus_per_sim = np.sum(surplus_matrix, axis=1)
mean_yearly_surplus = np.mean(total_surplus_per_sim)

mean_yearly_load = np.mean(total_load_per_sim)
mean_yearly_grid_import = np.mean(total_deficit_per_sim)
mean_yearly_microgrid_supply = mean_yearly_load - mean_yearly_grid_import

print("-" * 50)
print(f"RESULTS FOR: {config['name']}")
print("-" * 50)
print(f"Mean LPSP:                    {mean_lpsp*100:.3f}%")
print(f"Mean Prob of Deficit:         {mean_prob_deficit*100:.3f}%\n")
print(f"Total Annual Consumption:     {mean_yearly_load:,.0f} kWh")
print(f"Supplied by Microgrid:        {mean_yearly_microgrid_supply:,.0f} kWh")
print(f"Bought from Main Grid:        {mean_yearly_grid_import:,.0f} kWh")
print(f"Surplus (Exported/Wasted):    {mean_yearly_surplus:,.0f} kWh")
print("-" * 50)

# ==========================================
# PHASE 8: GRAPH EXPORT & INTERACTIVE DASHBOARD
# ==========================================
print("Generating and exporting high-resolution PNGs...")
sns.set_theme(style="whitegrid")

# --- Plotting Functions ---
def plot_beta(ax):
    target_hour = 12
    hour_data = df_ghi[(df_ghi['Hour'] == target_hour) & (df_ghi['R'] > 0.01)]['R']
    hour_data = np.clip(hour_data, a_min=None, a_max=0.9999)
    sns.histplot(hour_data, bins=30, stat="density", alpha=0.6, label='Historical Normalized GHI', ax=ax)
    
    x = np.linspace(0, 1, 100)
    a = beta_params[target_hour]['alpha']
    b = beta_params[target_hour]['beta']
    y = stats.beta.pdf(x, a, b)
    ax.plot(x, y, 'r-', lw=2, label=f'Fitted Beta(α={a:.2f}, β={b:.2f})')
    
    ax.set_title(f'Beta Distribution Fit for {target_hour}:00 PM')
    ax.set_xlabel('Normalized Irradiance R(t)')
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

def plot_lpsp(ax):
    sns.histplot(lpsp_array, bins=40, kde=True, color="teal", ax=ax)
    ax.set_title(f"Distribution of LPSP: {config['name']} (2500 Iterations)")
    ax.set_xlabel("Loss of Power Supply Probability (LPSP)")
    ax.set_ylabel("Frequency")
    ax.axvline(mean_lpsp, color='red', linestyle='dashed', linewidth=2, label=f'Mean LPSP: {mean_lpsp*100:.3f}%')
    ax.legend(loc='upper right')

def plot_ribbon(ax):
    pv_percentiles = np.percentile(sim_PV_power, [5, 50, 95], axis=0)
    avg_load_curve = np.mean(load_matrix, axis=0)[:24]
    hours_plot = np.arange(24)
    
    ax.plot(hours_plot, avg_load_curve, label=f"{config['type']} Load Profile", color='black', lw=2, linestyle='--')
    ax.plot(hours_plot, pv_percentiles[1, :24], label='Median PV Output', color='darkorange', lw=2)
    ax.fill_between(hours_plot, pv_percentiles[0, :24], pv_percentiles[2, :24], color='orange', alpha=0.3, label='90% PV Confidence Interval')
    
    ax.set_title(f"System Dynamics (Typical Day) - {config['name']}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(hours_plot)
    ax.legend(loc='upper right')

def plot_energy_mix(ax):
    labels = ['Supplied by Microgrid', 'Bought from Main Grid']
    sizes = [mean_yearly_microgrid_supply, mean_yearly_grid_import]
    colors = ['#4CAF50', '#F44336'] 
    explode = (0.05, 0) 
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
           shadow=True, startangle=90, textprops={'fontsize': 12})
    ax.set_aspect('equal') 
    ax.set_title(f"Annual Energy Mix - {config['name']}\nTotal Load: {mean_yearly_load:,.0f} kWh", pad=20)


# --- Automated Background Export ---
# 1. Beta Distribution
fig_exp, ax_exp = plt.subplots(figsize=(10, 6))
plot_beta(ax_exp)
plt.tight_layout()
fig_exp.savefig(f"1_Beta_Distribution_Scenario_{SELECTED_SCENARIO}.png", dpi=300)
plt.close(fig_exp)

# 2. LPSP Histogram
fig_exp, ax_exp = plt.subplots(figsize=(10, 6))
plot_lpsp(ax_exp)
plt.tight_layout()
fig_exp.savefig(f"2_LPSP_Distribution_Scenario_{SELECTED_SCENARIO}.png", dpi=300)
plt.close(fig_exp)

# 3. Ribbon Plot
fig_exp, ax_exp = plt.subplots(figsize=(12, 6))
plot_ribbon(ax_exp)
plt.tight_layout()
fig_exp.savefig(f"3_System_Dynamics_Scenario_{SELECTED_SCENARIO}.png", dpi=300)
plt.close(fig_exp)

# 4. Energy Mix Pie Chart
fig_exp, ax_exp = plt.subplots(figsize=(10, 6))
plot_energy_mix(ax_exp)
fig_exp.savefig(f"4_Energy_Mix_Scenario_{SELECTED_SCENARIO}.png", dpi=300)
plt.close(fig_exp)

print(f"-> Successfully saved 4 PNG files for Scenario {SELECTED_SCENARIO} in the current directory!")


# --- Interactive Dashboard Setup ---
print("Launching Interactive Dashboard...")
fig, ax = plt.subplots(figsize=(12, 6.5))
plt.subplots_adjust(bottom=0.2)
current_plot_idx = 0

plot_funcs = [plot_ribbon, plot_lpsp, plot_energy_mix, plot_beta]

def update_plot():
    ax.clear()
    ax.set_aspect('auto')
    plot_funcs[current_plot_idx](ax)
    fig.canvas.draw()

# --- Button Callback Functions ---
def next_plot(event):
    global current_plot_idx
    current_plot_idx = (current_plot_idx + 1) % len(plot_funcs)
    update_plot()

def prev_plot(event):
    global current_plot_idx
    current_plot_idx = (current_plot_idx - 1) % len(plot_funcs)
    update_plot()

# Add Buttons
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next \u2192')
bprev = Button(axprev, '\u2190 Prev')

bnext.on_clicked(next_plot)
bprev.on_clicked(prev_plot)

# Initialize the first plot
update_plot()
plt.show()