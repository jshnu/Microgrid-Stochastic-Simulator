# Microgrid-Stochastic-Simulator
Python-based Monte-Carlo simulation framework for evaluating urban microgrid reliability under stochastic solar uncertainty. Code for the Research paper 'Monte-Carlo Based Uncertainty Analysis of Rooftop Solar Generation for Microgrid Reliability in Urban India' by Annapureddy et al.

---

## Project Architecture & Methodology

Conventional microgrid plan-and-design workflows rely on deterministic annual-energy averages, which frequently obscure the real-world operational challenges of solar intermittency. This framework treats Global Horizontal Irradiance (GHI) as a deeply stochastic process by:
1. Fetching a high-resolution 4-year continuous historical data series (2022–2025) via the NASA POWER API.
2. Generating hourly empirical distributions using Maximum Likelihood Estimation (MLE) to fit continuous Beta probability functions.
3. Accounting for urban morphology and sun-angle dynamics via a time-dependent, stochastic uniform Shadow Modification Factor (SMF).
4. Evaluating system operational constraints using an hour-by-hour sequential battery State of Charge (SoC) tracking algorithm.

System reliability and sizing dependencies are analyzed across four architectural design configurations (Budget Residential, Standard Modern, Future-Proof Premium, and Commercial Office) using key probabilistic metrics: Loss of Power Supply Probability (LPSP), Probability of Deficit, and Expected Energy Not Served (EENS).

---

## Repository Structure
The repository is organized as follows:
```text
├── Convergence_Test.py    # Standalone script validating mathematical convergence (N=50 to N=5000)
├── Implementation.py      # Master simulation engine containing data processing, scenario configurations, and plotting
├── requirements.txt       # Hardcoded, version-locked environment dependencies
└── results/               # Compiled verification artifacts and output files
    ├── MC_Convergence_Graph.png # Output verification of the iteration convergence curve
    ├── Console_Outputs.docx    # Document containing detailed raw console printouts for all scenarios
    ├── A/                 # Visual artifacts generated for Scenario A (Budget Residential)
    ├── B/                 # Visual artifacts generated for Scenario B (Standard Modern Home)
    ├── C/                 # Visual artifacts generated for Scenario C (Future-Proof Premium)
    └── D/                 # Visual artifacts generated for Scenario D (Commercial IT Office)
```
---

## Installation & Environment Setup

To guarantee mathematical consistency and eliminate API execution conflicts, the execution pipeline requires a version-locked environment. Follow these setup steps:


1. Configure a Virtual Environment (Recommended):
[Bash]
```text
python -m venv myenv
# On Windows activation:
myenv\Scripts\activate
# On macOS/Linux activation:
source myenv/bin/activate
```
2.Install Package Dependencies:
```text
[Bash]
pip install -r requirements.txt
```

# Execution Instructions
```text
1. Verification of Simulation Convergence BoundsBefore executing full system scenario profiles, run the tracking script to evaluate the computational performance boundaries and prove the structural convergence limit of the model (N = 2500):
[Bash]
python Convergence_Test.py
This routine tests historical dataset scaling across iterations ranging from N=50 to N=5000, saving the structural verification plot as MC_Convergence_Graph.png.

2. Running Microgrid Architecture ScenariosThe master simulation execution profile is fully modularized. To evaluate a specific infrastructure tier:Open Implementation.py in a text editor or IDE.
Locate the master scenario select toggle under Phase 4:
[Python]
# Select 'A', 'B', 'C', or 'D'
SELECTED_SCENARIO = 'D'  
Execute the simulator from the terminal console:Bashpython Implementation.py
[Bash]
python Implementation.py
```
note: [Bash] indicates the command is run in Bash
