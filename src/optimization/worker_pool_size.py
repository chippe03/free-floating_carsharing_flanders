"""
Worker count sweep: profit maximising.

For a fixed fleet size and policy (with fixed parameters), sweeps
`total_workers` and finds the count that maximises gross profit per day.


Main entry points
-----------------
run_worker_sweep(...)          -> list[dict]   1-D sweep over worker count
run_worker_fleet_grid(...)     -> list[dict]   2-D grid: workers × fleet size
run_worker_param_grid(...)     -> list[dict]   2-D grid: workers × policy param
find_optimal(results)          -> dict         config with highest profit/day

plot_profit_by_workers(...)                    profit + cost breakdown vs workers
plot_worker_fleet_heatmap(...) 	               2-D heatmap workers × fleet
plot_worker_param_heatmap(...) 	               2-D heatmap workers × param
plot_utilisation(...)                          worker utilisation efficiency
plot_marginal_worker(...)                      marginal profit per extra worker
plot_all(...)                                  render everything at once
"""

import json
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from collections import defaultdict

from src.simulation.engine import SimulationEngine
from src.analysis.metrics import Metrics
from src.analysis.financials import FinancialResults, FinancialModel
from src.relocation.create_policy import load_policy
from src import config


# ------------------------------------------------------------------ #
#  Config patcher                                                    #
# ------------------------------------------------------------------ #

@contextmanager
def _patch_workers(n_workers: int):
    """
    Temporarily override config.simulation["workers"]["total_workers"].
    Restored on exit even if an exception is raised.
    """
    original = config.simulation["workers"]["total_workers"]
    config.simulation["workers"]["total_workers"] = n_workers
    try:
        yield
    finally:
        config.simulation["workers"]["total_workers"] = original


# ------------------------------------------------------------------ #
#  Shared result builder                                             #
# ------------------------------------------------------------------ #

def _build_result(
    event_log:    list[dict],
    policy_name:  str,
    fleet_size:   int,
    n_workers:    int,
    model:        FinancialModel,
    **extra_params,
) -> dict:
    m   = Metrics(event_log, label=policy_name, fleet_size=fleet_size)
    fin = FinancialResults(event_log, fleet_size=fleet_size, model=model)
    rel = m.relocation_stats()
    wor = m.worker_stats()

    # worker utilisation: relocations executed per worker per day
    n_days      = fin.n_days
    utilisation = (wor["relocations_executed"] / (n_workers * n_days)
                   if n_workers > 0 and n_days > 0 else 0.0)

    return {
        # identity
        "policy":     policy_name,
        "fleet_size": fleet_size,
        "n_workers":  n_workers,
        **extra_params,

        # operational
        "unmet_rate":              round(m.unmet_rate, 4),
        "unmet_total":             m.total_unmet,
        "total_requests":          m.total_requests,
        "total_served":            m.total_served,
        "relocations":             m.total_relocations,
        "relocations_executed":    wor["relocations_executed"],
        "relocations_queued":      wor["relocations_queued"],
        "relocations_forgotten":   wor["relocations_forgotten"],
        "peak_workers":            wor["peak_workers_active"],
        "worker_utilisation":      round(utilisation, 4),
        "relocation_min":          rel["total_min"],
        "relocation_km":           rel["total_km"],

        # financial: revenue
        "total_revenue_eur":           round(fin.total_revenue, 2),
        "lost_revenue_eur":            round(fin.lost_revenue, 2),
        "avg_revenue_per_trip_eur":    round(fin.avg_revenue_per_trip, 2),

        # financial: costs
        "vehicle_cost_eur":            round(fin.total_vehicle_cost, 2),
        "fuel_cost_eur":               round(fin.total_fuel_cost, 2),
        "relocation_cost_eur":         round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":      round(fin.total_repositioning_cost, 2),
        "total_cost_eur":              round(fin.total_cost, 2),

        # financial: profit (PRIMARY objective)
        "gross_profit_eur":            round(fin.gross_profit, 2),
        "gross_profit_per_day_eur":    round(fin.gross_profit_per_day, 2),
        "cost_per_trip_eur":           round(fin.cost_per_trip, 2),
    }


def _run_single(
    policy_name:   str,
    fleet_size:    int,
    n_workers:     int,
    n_days:        int,
    model:         FinancialModel,
    policy_kwargs: dict,
    param_label:   str,
    **extra_params,
) -> dict:
    with _patch_workers(n_workers):
        policy = load_policy(policy_name=policy_name, **policy_kwargs)
        engine = SimulationEngine(
            n_days              = n_days,
            policy              = policy,
            fleet_size_override = fleet_size,
            verbose             = False,
        )
        engine.run(verbose=False)

    result = _build_result(
        event_log   = engine.event_log,
        policy_name = policy_name,
        fleet_size  = fleet_size,
        n_workers   = n_workers,
        model       = model,
        **extra_params,
    )

    profit_sign = "+" if result["gross_profit_per_day_eur"] >= 0 else ""
    print(
        f"  {param_label:<50}  "
        f"unmet={result['unmet_rate']:>5.1%}  "
        f"reloc={result['relocations']:>4}  "
        f"util={result['worker_utilisation']:>5.1%}  "
        f"profit/day={profit_sign}{result['gross_profit_per_day_eur']:>8.0f}€"
    )
    return result


# ------------------------------------------------------------------ #
#  1-D sweep: workers                                                #
# ------------------------------------------------------------------ #

def run_worker_sweep(
    policy_name:            str,
    fleet_size:             int,
    n_days:                 int   = None,
    worker_min:             int   = 1,
    worker_max:             int   = 30,
    worker_step:            int   = 1,
    check_interval_minutes: int   = None,

    # Reactive
    alpha:                  int   = None,
    # Proactive
    planning_period_minutes:  int = None,
    lookahead_period_minutes: int = None,
    time_step_minutes:        int = None,
    history_log_path:         str = None,
    # Nightly
    trigger_hour:             int = None,

    financial_model: FinancialModel = None,
) -> list[dict]:
    """
    Sweep total_workers for a single policy with fixed fleet size and parameters.
    Finds the worker count that maximises gross profit per day.
    """
    model   = financial_model or FinancialModel()
    workers = range(worker_min, worker_max + 1, worker_step)

    print(f"\nWorker sweep  -  policy: {policy_name}  fleet: {fleet_size}")
    print(f"Testing {len(list(workers))} worker counts "
          f"({worker_min} to {worker_max}, step {worker_step})\n")

    policy_kwargs = {}
    if check_interval_minutes:   policy_kwargs["check_interval_minutes"]   = check_interval_minutes
    if alpha is not None:        policy_kwargs["alpha"]                     = alpha
    if planning_period_minutes:  policy_kwargs["planning_period_minutes"]  = planning_period_minutes
    if lookahead_period_minutes: policy_kwargs["lookahead_period_minutes"] = lookahead_period_minutes
    if time_step_minutes:        policy_kwargs["time_step_minutes"]        = time_step_minutes
    if history_log_path:         policy_kwargs["history_log_path"]         = history_log_path
    if trigger_hour is not None: policy_kwargs["trigger_hour"]             = trigger_hour

    results = []
    for n_workers in workers:
        result = _run_single(
            policy_name   = policy_name,
            fleet_size    = fleet_size,
            n_workers     = n_workers,
            n_days        = n_days,
            model         = model,
            policy_kwargs = policy_kwargs,
            param_label   = f"workers={n_workers}",
        )
        results.append(result)

    opt = find_optimal(results)
    print(f"\n  ✓ Optimal: workers={opt['n_workers']}  "
          f"profit/day=€{opt['gross_profit_per_day_eur']:,.0f}  "
          f"unmet={opt['unmet_rate']:.1%}  "
          f"utilisation={opt['worker_utilisation']:.1%}\n")
    return results


# ------------------------------------------------------------------ #
#  2-D grid: workers × fleet size                                    #
# ------------------------------------------------------------------ #

def run_worker_fleet_grid(
    policy_name:            str,
    n_days:                 int   = None,
    worker_values:          list  = None,
    fleet_values:           list  = None,
    check_interval_minutes: int   = None,
    alpha:                  int   = None,
    planning_period_minutes:  int = None,
    lookahead_period_minutes: int = None,
    time_step_minutes:        int = None,
    history_log_path:         str = None,
    trigger_hour:             int = None,
    financial_model: FinancialModel = None,
) -> list[dict]:
    """
    2-D grid sweep over (n_workers, fleet_size).
    Finds the jointly optimal staffing and fleet combination.
    """
    model   = financial_model or FinancialModel()
    workers = worker_values or list(range(2, 20, 2))
    fleets  = fleet_values  or list(range(40, 160, 20))

    total = len(workers) * len(fleets)
    print(f"\nWorker × Fleet grid  -  policy: {policy_name}")
    print(f"Grid: {len(workers)} worker values × {len(fleets)} fleet sizes = {total} runs\n")

    policy_kwargs = {}
    if check_interval_minutes:   policy_kwargs["check_interval_minutes"]   = check_interval_minutes
    if alpha is not None:        policy_kwargs["alpha"]                     = alpha
    if planning_period_minutes:  policy_kwargs["planning_period_minutes"]  = planning_period_minutes
    if lookahead_period_minutes: policy_kwargs["lookahead_period_minutes"] = lookahead_period_minutes
    if time_step_minutes:        policy_kwargs["time_step_minutes"]        = time_step_minutes
    if history_log_path:         policy_kwargs["history_log_path"]         = history_log_path
    if trigger_hour is not None: policy_kwargs["trigger_hour"]             = trigger_hour

    results = []
    for n_workers in workers:
        for fleet_size in fleets:
            result = _run_single(
                policy_name   = policy_name,
                fleet_size    = fleet_size,
                n_workers     = n_workers,
                n_days        = n_days,
                model         = model,
                policy_kwargs = policy_kwargs,
                param_label   = f"workers={n_workers:>3}  fleet={fleet_size:>3}",
            )
            results.append(result)

    opt = find_optimal(results)
    print(f"\n  ✓ Optimal: workers={opt['n_workers']}  fleet={opt['fleet_size']}  "
          f"profit/day=€{opt['gross_profit_per_day_eur']:,.0f}\n")
    return results


# ------------------------------------------------------------------ #
#  2-D grid: workers × policy param                                  #
# ------------------------------------------------------------------ #

def run_worker_param_grid(
    policy_name:            str,
    fleet_size:             int,
    param_name:             str,
    param_values:           list,
    n_days:                 int   = None,
    worker_values:          list  = None,
    check_interval_minutes: int   = None,
    history_log_path:       str   = None,
    financial_model: FinancialModel = None,
) -> list[dict]:
    """
    2-D grid over (n_workers, one policy parameter).
    Examples:
        param_name="alpha",                    param_values=[1,2,3,4,5]
        param_name="trigger_hour",             param_values=[0,1,2,3,4,5]
        param_name="planning_period_minutes",  param_values=[60,120,180,240]
    """
    model   = financial_model or FinancialModel()
    workers = worker_values or list(range(2, 20, 2))
    total   = len(workers) * len(param_values)

    print(f"\nWorker × {param_name} grid  -  policy: {policy_name}  fleet: {fleet_size}")
    print(f"Grid: {len(workers)} worker values × {len(param_values)} param values = {total} runs\n")

    base_kwargs = {}
    if check_interval_minutes: base_kwargs["check_interval_minutes"] = check_interval_minutes
    if history_log_path:       base_kwargs["history_log_path"]       = history_log_path

    results = []
    for n_workers in workers:
        for pval in param_values:
            kwargs = {**base_kwargs, param_name: pval}
            result = _run_single(
                policy_name   = policy_name,
                fleet_size    = fleet_size,
                n_workers     = n_workers,
                n_days        = n_days,
                model         = model,
                policy_kwargs = kwargs,
                param_label   = f"workers={n_workers:>3}  {param_name}={pval}",
                **{param_name: pval},
            )
            results.append(result)

    opt = find_optimal(results)
    print(f"\n  ✓ Optimal: workers={opt['n_workers']}  {param_name}={opt[param_name]}  "
          f"profit/day=€{opt['gross_profit_per_day_eur']:,.0f}\n")
    return results


# ------------------------------------------------------------------ #
#  Optimal finder                                                    #
# ------------------------------------------------------------------ #

def find_optimal(results: list[dict]) -> dict:
    """Return the result with the highest gross profit per day."""
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


# ------------------------------------------------------------------ #
#  Persist                                                           #
# ------------------------------------------------------------------ #

def save_sweep(results: list[dict], filename: str):
    output_dir = Path("./data/metrics/workers")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sweep saved to {path}")


def load_sweep(filename: str) -> list[dict]:
    path = Path("./data/metrics/workers") / f"{filename}.json"
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Shared style helpers                                              #
# ------------------------------------------------------------------ #

_COLORS = {
    "reactive":  "#1f77b4",
    "proactive": "#2ca02c",
    "nightly":   "#ff7f0e",
}

def _color(policy): return _COLORS.get(policy.lower(), "#888888")

def _eur_fmt(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
    )

def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


# ------------------------------------------------------------------ #
#  Plots                                                             #
# ------------------------------------------------------------------ #

def plot_profit_by_workers(
    results:   list[dict],
    save_path: str = None,
):
    """
    Top panel:    gross profit/day vs worker count, with unmet rate on secondary axis.
    Bottom panel: stacked cost breakdown vs worker count.
    The profit-optimal worker count is marked on both panels.
    """
    results  = sorted(results, key=lambda r: r["n_workers"])
    workers  = [r["n_workers"]                  for r in results]
    profit   = [r["gross_profit_per_day_eur"]   for r in results]
    unmet    = [r["unmet_rate"] * 100           for r in results]
    policy   = results[0]["policy"]
    opt      = find_optimal(results)

    cost_components = [
        ("Vehicle ownership",  "vehicle_cost_eur",       "#4c72b0"),
        ("Fuel (trips)",       "fuel_cost_eur",           "#55a868"),
        ("Relocation",         "relocation_cost_eur",     "#c44e52"),
        ("Repositioning",      "repositioning_cost_eur",  "#dd8452"),
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]}
    )

    # --- top: profit + unmet ---
    ax2 = ax_top.twinx()
    ax_top.plot(workers, profit, color=_color(policy), linewidth=2.5,
                marker="o", markersize=7, label="Profit/day (€)", zorder=5)
    ax2.plot(workers, unmet, color="#d62728", linewidth=1.5,
             marker="s", markersize=5, linestyle="--", label="Unmet rate (%)")
    ax_top.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_top.axvline(opt["n_workers"], color=_color(policy),
                   linestyle=":", linewidth=1.5, alpha=0.8)
    ax_top.annotate(
        f"Optimal\n{opt['n_workers']} workers\n€{opt['gross_profit_per_day_eur']:,.0f}/day",
        xy     = (opt["n_workers"], opt["gross_profit_per_day_eur"]),
        xytext = (opt["n_workers"] + 0.5, max(profit) * 0.85),
        fontsize   = 8,
        color      = _color(policy),
        arrowprops = dict(arrowstyle="->", color=_color(policy)),
    )
    _eur_fmt(ax_top)
    ax_top.set_ylabel("Gross profit per day (€)", fontsize=11)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=11, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    lines  = ax_top.lines + ax2.lines
    labels = [l.get_label() for l in lines]
    ax_top.legend(lines, labels, fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(
        f"{policy.capitalize()} policy - profit & costs vs worker count  "
        f"(fleet={results[0]['fleet_size']})",
        fontsize=13
    )

    # --- bottom: stacked costs ---
    bottoms = np.zeros(len(results))
    for comp_label, key, color in cost_components:
        vals = np.array([r.get(key, 0) for r in results])
        ax_bot.bar(workers, vals, bottom=bottoms, label=comp_label,
                   color=color, alpha=0.85, width=max(0.6, worker_step_from(results) * 0.7))
        bottoms += vals

    ax_bot.axvline(opt["n_workers"], color=_color(policy),
                   linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"Optimal ({opt['n_workers']} workers)")
    ax_bot.set_xlabel("Number of workers", fontsize=12)
    ax_bot.set_ylabel("Total cost (€)", fontsize=11)
    _eur_fmt(ax_bot)
    ax_bot.legend(fontsize=9)
    ax_bot.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def worker_step_from(results):
    workers = sorted(set(r["n_workers"] for r in results))
    return workers[1] - workers[0] if len(workers) > 1 else 1


def plot_utilisation(
    results:   list[dict],
    save_path: str = None,
):
    """
    Three-panel view of worker efficiency:
      Left:   relocations executed and queued/forgotten vs worker count
      Centre: utilisation rate (relocations/worker/day) vs worker count
      Right:  relocation cost per executed relocation vs worker count

    Helps identify whether workers are bottlenecked (too few) or idle (too many).
    """
    results = sorted(results, key=lambda r: r["n_workers"])
    workers  = [r["n_workers"]                              for r in results]
    executed = [r["relocations_executed"]                   for r in results]
    queued   = [r["relocations_queued"]                     for r in results]
    forgot   = [r["relocations_forgotten"]                  for r in results]
    util     = [r["worker_utilisation"] * 100               for r in results]
    cost_per = [
        (r["relocation_cost_eur"] / r["relocations_executed"]
         if r["relocations_executed"] > 0 else 0)
        for r in results
    ]
    policy = results[0]["policy"]
    opt    = find_optimal(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # left: volume
    ax = axes[0]
    ax.bar(workers, executed, label="Executed",  color=_color(policy), alpha=0.8)
    ax.bar(workers, queued,   label="Queued",    color="#ff7f0e", alpha=0.7,
           bottom=executed)
    ax.bar(workers, forgot,   label="Forgotten", color="#d62728", alpha=0.7,
           bottom=[e + q for e, q in zip(executed, queued)])
    ax.axvline(opt["n_workers"], color="black", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Workers", fontsize=11)
    ax.set_ylabel("Relocations", fontsize=11)
    ax.set_title("Relocation volume", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # centre: utilisation
    ax = axes[1]
    ax.plot(workers, util, color=_color(policy), linewidth=2,
            marker="o", markersize=6)
    ax.axvline(opt["n_workers"], color="black", linestyle=":", linewidth=1.5,
               label=f"Optimal ({opt['n_workers']})")
    ax.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="100% utilisation")
    ax.set_xlabel("Workers", fontsize=11)
    ax.set_ylabel("Utilisation (%)", fontsize=11)
    ax.set_title("Worker utilisation\n(relocations/worker/day)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # right: cost per relocation
    ax = axes[2]
    ax.plot(workers, cost_per, color="#c44e52", linewidth=2,
            marker="^", markersize=6)
    ax.axvline(opt["n_workers"], color="black", linestyle=":", linewidth=1.5,
               label=f"Optimal ({opt['n_workers']})")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.1f}")
    )
    ax.set_xlabel("Workers", fontsize=11)
    ax.set_ylabel("Cost per relocation (€)", fontsize=11)
    ax.set_title("Relocation cost efficiency", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        f"{policy.capitalize()} policy - worker utilisation analysis  "
        f"(fleet={results[0]['fleet_size']})",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_marginal_worker(
    results:   list[dict],
    save_path: str = None,
):
    """
    Marginal gross profit gained by hiring one additional worker.
    Computed via central finite differences.
    Crosses zero at the profit-optimal worker count.
    Also shows marginal relocation cost to reveal where cost overtakes gain.
    """
    results  = sorted(results, key=lambda r: r["n_workers"])
    workers  = [r["n_workers"]                  for r in results]
    profit   = [r["gross_profit_per_day_eur"]   for r in results]
    rel_cost = [r["relocation_cost_eur"]        for r in results]
    policy   = results[0]["policy"]

    mid_w, d_profit, d_cost = [], [], []
    for i in range(1, len(workers) - 1):
        dw = workers[i + 1] - workers[i - 1]
        if dw == 0:
            continue
        d_profit.append((profit[i + 1]   - profit[i - 1])   / dw)
        d_cost.append(  (rel_cost[i + 1] - rel_cost[i - 1]) / dw)
        mid_w.append(workers[i])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mid_w, d_profit, color=_color(policy), linewidth=2,
            marker="o", markersize=6, label="Marginal profit/day (€)")
    ax.plot(mid_w, d_cost,   color="#c44e52", linewidth=1.5,
            marker="^", markersize=5, linestyle="--",
            label="Marginal relocation cost (€)")

    ax.set_xlabel("Number of workers", fontsize=12)
    ax.set_ylabel("Marginal value per additional worker (€/day)", fontsize=12)
    ax.set_title(
        f"{policy.capitalize()} policy - diminishing returns: "
        f"marginal profit per extra worker",
        fontsize=13
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.1f}")
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_worker_fleet_heatmap(
    results:   list[dict],
    metric:    str = "gross_profit_per_day_eur",
    save_path: str = None,
):
    """
    2-D heatmap of a metric over (n_workers, fleet_size).
    Best cell is outlined in navy.

    metric options:
        "gross_profit_per_day_eur"   ← default
        "unmet_rate"
        "relocation_cost_eur"
        "worker_utilisation"
    """
    worker_vals = sorted(set(r["n_workers"]  for r in results))
    fleet_vals  = sorted(set(r["fleet_size"] for r in results))

    lookup = {(r["n_workers"], r["fleet_size"]): r[metric] for r in results}

    matrix = np.array([
        [lookup.get((w, f), np.nan) for f in fleet_vals]
        for w in worker_vals
    ])

    higher_is_better = metric not in ("unmet_rate", "relocation_cost_eur",
                                      "repositioning_cost_eur", "total_cost_eur")
    best_idx = np.unravel_index(
        np.nanargmax(matrix) if higher_is_better else np.nanargmin(matrix),
        matrix.shape
    )

    fig, ax = plt.subplots(figsize=(max(8, len(fleet_vals) * 1.2),
                                    max(5, len(worker_vals) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   origin="lower",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=10)

    ax.set_xticks(range(len(fleet_vals)))
    ax.set_xticklabels(fleet_vals)
    ax.set_yticks(range(len(worker_vals)))
    ax.set_yticklabels(worker_vals)
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Number of workers", fontsize=12)

    for i in range(len(worker_vals)):
        for j in range(len(fleet_vals)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text = (f"€{val:,.0f}" if "eur" in metric
                    else f"{val:.1%}" if "rate" in metric or "utilisation" in metric
                    else f"{val:.2f}")
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6.5, color="black")

    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=2, edgecolor="navy", facecolor="none"
    ))

    opt = results[0]
    for r in results:
        if r["n_workers"] == worker_vals[best_idx[0]] and \
           r["fleet_size"] == fleet_vals[best_idx[1]]:
            opt = r
            break

    policy = results[0]["policy"]
    ax.set_title(
        f"{policy.capitalize()} policy - {metric.replace('_', ' ').title()}\n"
        f"Optimal: {opt['n_workers']} workers, {opt['fleet_size']} vehicles  "
        f"→ €{opt['gross_profit_per_day_eur']:,.0f}/day",
        fontsize=12
    )
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_worker_param_heatmap(
    results:    list[dict],
    param_name: str,
    metric:     str = "gross_profit_per_day_eur",
    save_path:  str = None,
):
    """
    2-D heatmap of a metric over (n_workers, policy_param).
    param_name must match the key stored in result dicts,
    e.g. "alpha", "trigger_hour", "planning_period_minutes".
    """
    worker_vals = sorted(set(r["n_workers"]    for r in results))
    param_vals  = sorted(set(r[param_name]     for r in results))

    lookup = {(r["n_workers"], r[param_name]): r[metric] for r in results}

    matrix = np.array([
        [lookup.get((w, p), np.nan) for p in param_vals]
        for w in worker_vals
    ])

    higher_is_better = metric not in ("unmet_rate", "relocation_cost_eur",
                                      "repositioning_cost_eur", "total_cost_eur")
    best_idx = np.unravel_index(
        np.nanargmax(matrix) if higher_is_better else np.nanargmin(matrix),
        matrix.shape
    )

    fig, ax = plt.subplots(figsize=(max(8, len(param_vals) * 1.5),
                                    max(5, len(worker_vals) * 0.6)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   origin="lower",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(metric.replace("_", " ").title(), fontsize=10)

    ax.set_xticks(range(len(param_vals)))
    ax.set_xticklabels(param_vals)
    ax.set_yticks(range(len(worker_vals)))
    ax.set_yticklabels(worker_vals)
    ax.set_xlabel(param_name.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Number of workers", fontsize=12)

    for i in range(len(worker_vals)):
        for j in range(len(param_vals)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text = (f"€{val:,.0f}" if "eur" in metric
                    else f"{val:.1%}" if "rate" in metric or "utilisation" in metric
                    else f"{val:.2f}")
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6.5, color="black")

    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=2, edgecolor="navy", facecolor="none"
    ))

    opt_w = worker_vals[best_idx[0]]
    opt_p = param_vals[best_idx[1]]
    policy = results[0]["policy"]
    ax.set_title(
        f"{policy.capitalize()} policy - {metric.replace('_', ' ').title()}\n"
        f"Optimal: {opt_w} workers, {param_name}={opt_p}",
        fontsize=12
    )
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ------------------------------------------------------------------ #
#  Run all plots at once                                             #
# ------------------------------------------------------------------ #

def plot_all(
    filename:       str,
    param_name:     str  = None,
    output_dir:     str  = None,
):
    """
    Load a saved sweep and render all relevant plots.

    filename:    saved sweep JSON (without .json)
    param_name:  if the sweep is a 2-D workers × param grid,
                 provide the param key to render the param heatmap
    """
    def path(name):
        return str(Path(output_dir) / name) if output_dir else None

    results = load_sweep(filename)

    # detect sweep type from result keys
    has_fleet_grid = len(set(r["fleet_size"] for r in results)) > 1
    has_param_grid = param_name and param_name in results[0]

    if has_fleet_grid:
        plot_worker_fleet_heatmap(results,
                                  save_path=path("worker_fleet_heatmap.png"))
        plot_worker_fleet_heatmap(results, metric="unmet_rate",
                                  save_path=path("worker_fleet_heatmap_unmet.png"))
    elif has_param_grid:
        plot_worker_param_heatmap(results, param_name,
                                  save_path=path(f"worker_{param_name}_heatmap.png"))
    else:
        # 1-D worker sweep
        plot_profit_by_workers(results,
                               save_path=path("worker_profit.png"))
        plot_utilisation(results,
                         save_path=path("worker_utilisation.png"))
        plot_marginal_worker(results,
                             save_path=path("worker_marginal.png"))


if __name__ == "__main__":
    FLEET_SIZE = 650
    N_DAYS     = 30
    model      = FinancialModel()

    # --- 1-D: worker count only (reactive) ---
    for policy in ["reactive", "proactive", "nightly"]:
        results_1d = run_worker_sweep(
            policy_name  = policy,
            fleet_size   = FLEET_SIZE,
            n_days       = N_DAYS,
            worker_min   = 50,
            worker_max   = 500,
            worker_step  = 50,
            financial_model = model,
        )
        save_sweep(results_1d, f"worker_sweep_{policy}")
        plot_all(f"worker_sweep_{policy}", output_dir="./data/metrics/workers/plots")

    # # --- 2-D: workers × fleet (nightly) ---
    # results_2d = run_worker_fleet_grid(
    #     policy_name    = "nightly",
    #     n_days         = N_DAYS,
    #     worker_values  = list(range(2, 16, 2)),
    #     fleet_values   = list(range(40, 140, 20)),
    #     trigger_hour   = 2,
    #     financial_model = model,
    # )
    # save_sweep(results_2d, "worker_fleet_grid_nightly")
    # plot_all("worker_fleet_grid_nightly", output_dir="./data/metrics/workers/plots")

    # # --- 2-D: workers × alpha (reactive) ---
    # results_param = run_worker_param_grid(
    #     policy_name   = "reactive",
    #     fleet_size    = FLEET_SIZE,
    #     param_name    = "alpha",
    #     param_values  = [1, 2, 3, 4, 5, 6, 8, 10],
    #     n_days        = N_DAYS,
    #     worker_values = list(range(2, 16, 2)),
    #     financial_model = model,
    # )
    # save_sweep(results_param, "worker_alpha_grid_reactive")
    # plot_all("worker_alpha_grid_reactive", param_name="alpha",
    #          output_dir="./data/metrics/workers/plots")