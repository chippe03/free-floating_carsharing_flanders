# Intercity FFCS-OA Relocation Simulation

Simulation and analysis pipeline for the thesis *"Free-floating car-sharing in Flanders: a case study"*.

Evaluates three operator-based relocation policies — reactive, nightly, and proactive — against a no-relocation baseline, across an eight-city network (Antwerp, Bruges, Brussels, Ghent, Kortrijk, Leuven, Mechelen, Ostend). Fleet size, policy parameters, and worker count are optimised sequentially. Results are compared under both a cash-based profit objective and a goodwill objective that penalises unmet demand as foregone revenue.

---

## Project structure

```
config/
  cities.yaml                     City demand weights and other information
  simulation.yaml                 Default configurations of the simulation
  financials.yaml                 Cost and revenue parameters
  calculate_city_weights.py       Utility: compute demand weights from raw data
  get_city_coordinates.py         Utility: fetch city coordinates
data/
  metrics/                        Output directory — one subfolder per pipeline step
  processed/                      Travel data used throughout the simulation
  raw/
src/
  analysis/                       Metrics and financial model
  api/
  demand/
  entities/
  optimization/                   Separate optimisation procedure for each parameter
  relocation/                     Policy implementations (baseline, reactive, nightly, proactive)
  simulation/                     Discrete-event simulation state and engine
  __init__.py
  config.py                       Loads and exposes all config YAML files
  final_param.py                  Single source of truth for all optimal configurations
  0_baseline_characterization.py
  1_cash_fleet_sweep.py
  2_parameter_sweep.py
  3_goodwill_fleet_sweep.py
  4_worker_sweep.py
  5_sequential_validation.py
  6_full_comparison.py
  7_sensitivity_goodwill.py
  7_sensitivity_demand.py
  8_robustness.py
```

---

## `final_param.py`

The single source of truth for all optimal configurations. Every script from Step 3 onwards imports from here rather than hardcoding values, so changing a fleet size or parameter in one place propagates everywhere.
Each entry contains: `fleet`, `workers`, `params`, `policy_name`, `label`, `color`.

---

## Pipeline

The eight steps are designed to run in order. Each step reads from the previous step's output directory. All steps support `--plot` (skip simulation, re-render figures from saved JSON) and `--skip-existing` (resume an interrupted run).

### Step 0 — Baseline characterisation
```
python 0_baseline_characterization.py
```
Runs the no-relocation baseline for 300 days at fleet=650. Establishes that demand imbalances self-correct to a stable equilibrium (no accumulating trend). Produces daily stability plots, hourly/weekday demand patterns, and per-city breakdowns.

Output: `data/metrics/0_baseline/`

---

### Step 1 — Cash-based fleet sweep
```
python 1_cash_fleet_sweep.py
```
Sweeps all four policies over fleet sizes 300–800 (step 25) with `goodwill_multiplier=0` (cash only, no lost-revenue penalty). Identifies the profit-optimal fleet size per policy under cash accounting, and records the fleet size at which the baseline first reaches ≤5% unmet demand as a service-quality reference point.

**Key finding:** cash profit cannot differentiate relocation from no-relocation, motivating the goodwill objective in Step 2 onwards.

Output: `data/metrics/1_cash_fleet/`

---

### Step 2 — Parameter sweep
```
python 2_parameter_sweep.py
```
Optimises each relocation policy's parameters at its Step 1 cash-optimal fleet size using `goodwill_multiplier=1.0`:
- **Reactive:** alpha × check_interval grid
- **Nightly:** trigger_hour
- **Proactive:** planning_period × lookahead × check_interval grid

Sweeps are append-mode (safe to interrupt and resume). Saves `optimal_params.json` for downstream steps. Produces cash-vs-goodwill sensitivity figures demonstrating that the goodwill objective creates a meaningfully larger optimisation signal.

Output: `data/metrics/2_parameters/`

---

### Step 3 — Goodwill fleet sweep
```
python 3_goodwill_fleet_sweep.py
```
Repeats the Step 1 fleet sweep (300–800, step 25) with `goodwill_multiplier=1.0` and the goodwill-optimal parameters from Step 2. Parameters are loaded from `final_param.OPTIMAL`. Produces comparison figures overlaying the cash (Step 1) and goodwill (Step 3) curves to show how the objective shifts the profit-optimal fleet size.

Output: `data/metrics/3_goodwill_fleet/`

---

### Step 4 — Worker count sweep
```
python 4_worker_sweep.py [--policies reactive nightly proactive]
```
Sweeps worker pool size per policy at the Step 3 goodwill-optimal fleet, with goodwill-optimal parameters. Sweep ranges:
- Reactive / Proactive: 25–4000 workers
- Nightly: 25–800 workers

Produces a diminishing-returns plot (`fig_4_W4`) with dots coloured green/red by marginal profit per additional worker. **Recommended worker counts are identified manually** from this plot and recorded in `RECOMMENDED` in the script; these are saved as `optimal_workers.json` for downstream steps.

Output: `data/metrics/4_workers/`

---

### Step 5 — Sequential validation
```
python 5_sequential_validation.py
```
Validates the sequential optimisation approach (fleet → workers) by re-sweeping the fleet at the Step 4 recommended worker counts for the reactive and proactive policies. Nightly is excluded (profit declines monotonically with workers regardless of fleet).

If the profit-optimal fleet at recommended workers matches the Step 3 optimum, the sequential approach is confirmed. If it shifts, the corrected fleet size is reported. **Note:** `STEP3_OPT_FLEET` in this script is hardcoded from the Step 3 JSON output, not from `final_param.OPTIMAL` (which already contains the post-validation corrected fleet).

Output: `data/metrics/5_sequential/`

---

### Step 6 — Full policy comparison
```
python 6_full_comparison.py
```
Evaluates all four policies at their individually optimal configurations (fleet from Step 5, parameters from Step 2, workers from Step 4), loaded from `final_param.OPTIMAL`. For each policy, searches for a matching cached result from earlier steps before running a fresh simulation; use `--rerun` to force fresh runs.

Produces the primary policy ranking under both the goodwill and cash objectives, plus spatial/temporal analysis of service improvements (per-city unmet rates by hour and day of week, difference heatmaps vs baseline).

Output: `data/metrics/6_comparison/`

---

### Step 7a — Demand sensitivity
```
python 7_sensitivity_demand.py [--reoptimise]
```
Repeats the Step 6 evaluation across demand scaling factors `{0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00}` × base arrival rate. Tests whether policy rankings hold at lower or higher demand levels.

Two modes:
- **Fixed fleet** (default): each policy runs at its `final_param.OPTIMAL` fleet across all demand levels.
- **Reoptimised fleet** (`--reoptimise`): baseline fleet is re-swept per demand level to find the profit-optimal fleet, and all policies are evaluated at that fleet.

Output: `data/metrics/7_demand/`

---

### Step 7b — Goodwill sensitivity
```
python 7_sensitivity_goodwill.py
```
Derives profit at any goodwill multiplier analytically from the Step 6 results — **no new simulations**. Uses the identity:

```
profit(m) = cash_profit − m × lost_revenue_per_day
```

Produces the full profit-vs-multiplier sensitivity curve for each policy and identifies the breakeven goodwill multiplier at which each relocation policy becomes preferable to the baseline.

Reads from: `data/metrics/6_comparison/optimal_results.json`  
Output: `data/metrics/7_goodwill/`

---

### Step 8 — Robustness check
```
python 8_robustness.py [--seeds 10]
```
Runs each policy 10 times at its Step 6 optimal configuration using different random seeds (demand realisations). The same 10 seeds are used across all policies, enabling paired comparison. Summarises the profit distribution by mean, standard deviation, and 95% confidence interval (t-distribution). Checks whether policy rankings from Step 6 are stable across seeds.

Output: `data/metrics/8_robustness/`

---

## Common flags

| Flag | Effect |
|---|---|
| `--plot` | Skip simulation; load saved JSON and re-render figures |
| `--skip-existing` | Resume an interrupted sweep (append-mode) |
| `--rerun` | Force fresh simulations even if cached results exist (Step 6 only) |
| `--policies` | Run a subset of policies (Steps 4, 5, 6) |
| `--seeds N` | Number of random seeds to use (Step 8) |
| `--multipliers` | Demand multipliers to test (Step 7b) |
| `--reoptimise` | Re-optimise fleet per demand level (Step 7b) |

---

## Output layout

```
data/metrics/
  0_baseline/
    baseline_events.json
    plots/
  1_cash_fleet/
    sweep_{policy}.json          # one file per policy
    plots/
  2_parameters/
    param_{policy}.json          # full parameter sweep per policy
    optimal_params.json          # goodwill-optimal params (saved by Step 2)
    plots/
  3_goodwill_fleet/
    fleet_sweep_{policy}.json
    plots/
  4_workers/
    workers_{policy}.json
    optimal_workers.json         # recommended worker counts (manual)
    optimal_workers_auto.json    # profit-argmax worker counts (automatic)
    plots/
  5_sequential/
    validation_{policy}.json
    plots/
  6_comparison/
    optimal_results.json         # one result dict per policy
    {policy}/
      events_{policy}.json       # full event log
      plots/
    comparison/                  # cross-policy spatial figures
  7_goodwill/
    plots/
  7_demand/
    demand_sensitivity.json
    demand_sensitivity_reopt.json
    plots/
  8_robustness/
    robustness_results.json
    plots/
```

---