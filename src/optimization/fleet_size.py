"""
Fleet size sweep: profit maximising.
 
Runs the simulation across a range of fleet sizes for a given policy,
records both operational and financial KPIs, and exposes a suite of
plots to understand the profit curve and its drivers.


Main entry points
-----------------
run_fleet_sweep(...)   -> list[dict]   run sweep, return results
find_optimal(results)  -> dict         fleet size that maximises gross profit/day

plot_profit(...)                 gross profit vs fleet size
plot_unmet_demand(...)           unmet rate vs fleet size
plot_cost_breakdown(...)         stacked cost components vs fleet size
plot_revenue_vs_cost(...)        revenue & total cost on same axes
plot_relocation_cost(...)        relocation cost vs fleet size
plot_marginal_profit(...)        marginal gross profit gained per one extra vehicle
plot_unmet_vs_relocation(...)    service quality vs relocation cost
plot_lost_rev_vs_vehicles(...)   lost revenue vs vehicle ownership
"""


import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.simulation.engine import SimulationEngine
from src.analysis.metrics  import Metrics
from src.analysis.financials import FinancialResults, FinancialModel
from src.relocation.create_policy import load_policy
from src import config


# ------------------------------------------------------------------ #
#  Sweep                                                             #
# ------------------------------------------------------------------ #

def run_fleet_sweep(
        policy_name: str = "reactive",    n_days:    int = None,    check_interval_minutes: int = None,
        fleet_min:   int = 20,            fleet_max: int = 200,     fleet_step:             int = 10,
        verbose:    bool = False,

        # Reactive
        alpha:                   int = None,

        # PROACTIVE
        planning_period_minutes: int = None,    lookahead_period_minutes: int = None, 
        time_step_minutes:       int = None,    history_log_path:         str = None,

        # NIGHTLY
        trigger_hour:            int = None,

        goodwill_multiplier:   float = None,
) -> list[dict]:
    """
    Run the simulation for each fleet size and return a list of result dicts.
 
    Each result contains:
      - fleet / policy metadata
      - operational KPIs  (unmet rate, relocations, …)
      - financial KPIs    (revenue, costs broken down, gross profit)
    """
    model   = FinancialModel(goodwill_multiplier)
    sizes   = range(fleet_min, fleet_max + 1, fleet_step)

    print(f"\nFleet size sweep - policy: {policy_name}")
    print(f"Testing {len(list(sizes))} fleet sizes "
          f"({fleet_min} to {fleet_max} step {fleet_step})\n")
    
    results = []

    for fleet_size in sizes:
        policy = load_policy(
            policy_name             = policy_name,              check_interval_minutes   = check_interval_minutes,
            alpha                   = alpha,
            planning_period_minutes = planning_period_minutes,  lookahead_period_minutes = lookahead_period_minutes,
            time_step_minutes       = time_step_minutes,        history_log_path         = history_log_path,
            trigger_hour            = trigger_hour,
        )

        engine = SimulationEngine(
            n_days              = n_days,
            policy              = policy,
            fleet_size_override = fleet_size,
            verbose             = False
        )
        engine.run(verbose=False)

        # --- operational metrics ---
        m   = Metrics(
            engine.event_log,
            label      = f"{policy_name}_fleet{fleet_size}",
            n_days     = n_days,
            fleet_size = fleet_size
        )
        rel = m.relocation_stats()
        wor = m.worker_stats()
 
        # --- financial metrics ---
        fin = FinancialResults(
            engine.event_log,
            label      = f"{policy_name}_fleet{fleet_size}",
            fleet_size = fleet_size,
            model      = model,
        )

        result = {
            # identity
            "fleet_size":   fleet_size,
            "policy":       policy_name,
            "goodwill_multiplier":   model.goodwill_multiplier,
 
            # operational
            "unmet_rate":           round(m.unmet_rate, 4),
            "unmet_total":          m.total_unmet,
            "total_requests":       m.total_requests,
            "total_served":         m.total_served,
            "total_duration_min":   m.total_duration_min,
            "relocations":          m.total_relocations,
            "relocation_min":       rel["total_min"],
            "relocation_km":        rel["total_km"],
            "relocations_queued":   wor["relocations_queued"],
            "relocations_forgotten":wor["relocations_forgotten"],
            "peak_workers":         wor["peak_workers_active"],
 
            # financial — revenue
            "total_revenue_eur":        round(fin.total_revenue, 2),
            "avg_revenue_per_trip_eur": round(fin.avg_revenue_per_trip, 2),
            "lost_revenue_eur":         round(fin.lost_revenue, 2),
            "lost_revenue_goodwill_eur":round(fin.lost_revenue * model.goodwill_multiplier, 2),
 
            # financial — costs (broken down)
            "vehicle_cost_eur":         round(fin.total_vehicle_cost, 2),
            "fuel_cost_eur":            round(fin.total_fuel_cost, 2),
            "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
            "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
            "total_cost_eur":           round(fin.total_cost, 2),
 
            # financial — profit  ← PRIMARY objective
            "gross_profit_eur":         round(fin.gross_profit, 2),
            "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
            "revenue_per_vehicle_per_day_eur": round(fin.revenue_per_vehicle_per_day, 2),
            "cost_per_trip_eur":        round(fin.cost_per_trip, 2),
            "breakeven_trips_per_day":  round(fin.breakeven_trips_per_day, 1),
        }
 
        results.append(result)
 
        profit_sign = "+" if result["gross_profit_per_day_eur"] >= 0 else ""
        print(
            f"  fleet={fleet_size:>3}  "
            f"unmet={m.unmet_rate:>5.1%}  "
            f"served={m.total_served:>5}  "
            f"profit/day={profit_sign}{result['gross_profit_per_day_eur']:>8.0f}€  "
            f"reloc={m.total_relocations:>4}"
        )
 
    return results


def save_sweep(results: list[dict], filename: str):
    output_dir = Path("./data/metrics/fleet_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep saved to {path}")


def load_sweep(filename: str) -> list[dict]:
    path = Path("./data/metrics/fleet_sweep") / f"{filename}.json"
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Optimal fleet finder                                              #
# ------------------------------------------------------------------ #
 
def find_optimal(results: list[dict]) -> dict:
    """
    Return the result dict for the fleet size that maximises
    gross profit per day (primary objective).
    """
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])
 
 
def print_optimal_summary(sweep_files: dict):
    """
    Print a table comparing the profit-optimal fleet for each policy.
    """
    print(f"\n{'='*70}")
    print(f"  PROFIT-OPTIMAL FLEET SIZE PER POLICY")
    print(f"{'='*70}")
    print(f"  {'Policy':<12} {'Fleet':>6} {'Profit/day':>12} "
          f"{'Unmet':>7} {'Revenue':>12} {'Cost':>12}")
    print(f"  {'-'*68}")
 
    for label, filename in sweep_files.items():
        results = load_sweep(filename)
        opt     = find_optimal(results)
        print(
            f"  {label:<12} "
            f"{opt['fleet_size']:>6}  "
            f"€{opt['gross_profit_per_day_eur']:>10,.0f}  "
            f"{opt['unmet_rate']:>6.1%}  "
            f"€{opt['total_revenue_eur']:>10,.0f}  "
            f"€{opt['total_cost_eur']:>10,.0f}"
        )
    print(f"{'='*70}\n")



# ------------------------------------------------------------------ #
#  Shared style helpers                                              #
# ------------------------------------------------------------------ #
 
_COLORS = {
    "Baseline":  "#d62728",
    "Reactive":  "#1f77b4",
    "Proactive": "#2ca02c",
    "Nightly":   "#ff7f0e",
}
_MARKERS = {
    "Baseline": "o",
    "Reactive": "s",
    "Proactive": "^",
    "Nightly":  "D",
}
 
def _color(label):  return _COLORS.get(label, "#888888")
def _marker(label): return _MARKERS.get(label, "o")
 
def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


# ------------------------------------------------------------------ #
#  Plots                                                             #
# ------------------------------------------------------------------ #
 
def plot_profit(
    sweep_files:  dict,
    save_path:    str  = None,
):
    """
    Gross profit per day vs fleet size (PRIMARY optimisation plot).
    Annotates each curve's profit-maximising fleet size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for label, filename in sweep_files.items():
        results = load_sweep(filename)
        sizes   = [r["fleet_size"]              for r in results]
        profits = [r["gross_profit_per_day_eur"] for r in results]
 
        ax.plot(sizes, profits,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=6)
 
        opt = find_optimal(results)
        sign = "+" if opt["gross_profit_per_day_eur"] >= 0 else ""
        ax.annotate(
            f"{opt['fleet_size']} veh\n{sign}€{opt['gross_profit_per_day_eur']:,.0f}/day",
            xy     = (opt["fleet_size"], opt["gross_profit_per_day_eur"]),
            xytext = (opt["fleet_size"] + 4, opt["gross_profit_per_day_eur"] + max(profits) * 0.04),
            fontsize   = 8,
            color      = _color(label),
            arrowprops = dict(arrowstyle="->", color=_color(label)),
        )
 
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title("Fleet size vs. daily gross profit by relocation policy", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_revenue_vs_cost(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Total revenue and total cost on the same axes.
    The gap between the two curves is gross profit; crossing point = break-even.
    One subplot per policy for readability.
    """
    n = len(sweep_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
 
    for ax, (label, filename) in zip(axes, sweep_files.items()):
        results  = load_sweep(filename)
        sizes    = [r["fleet_size"]      for r in results]
        revenue  = [r["total_revenue_eur"] for r in results]
        cost     = [r["total_cost_eur"]    for r in results]
        profit   = [r["gross_profit_per_day_eur"] for r in results]
 
        ax.plot(sizes, revenue, color="#2ca02c", linewidth=2, label="Total revenue")
        ax.plot(sizes, cost,    color="#d62728", linewidth=2, label="Total cost")
        ax.fill_between(sizes, revenue, cost,
                        where=[r >= c for r, c in zip(revenue, cost)],
                        alpha=0.15, color="#2ca02c", label="Profit zone")
        ax.fill_between(sizes, revenue, cost,
                        where=[r < c for r, c in zip(revenue, cost)],
                        alpha=0.15, color="#d62728", label="Loss zone")
 
        opt = find_optimal(results)
        ax.axvline(opt["fleet_size"], color=_color(label),
                   linestyle=":", linewidth=1.5,
                   label=f"Optimal ({opt['fleet_size']} veh)")
 
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Fleet size (vehicles)", fontsize=11)
        ax.set_ylabel("EUR (total simulation period)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
 
    fig.suptitle("Revenue vs. cost by fleet size", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_cost_breakdown(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Stacked area chart showing how each cost component evolves with fleet size.
    One subplot per policy.
    Useful to see whether vehicle ownership or relocation dominates costs.
    """
    cost_components = {
        "Vehicle ownership": ("vehicle_cost_eur",         "#4c72b0"),
        "Fuel (trips)":      ("fuel_cost_eur",            "#55a868"),
        "Relocation":        ("relocation_cost_eur",      "#c44e52"),
        "Repositioning":     ("repositioning_cost_eur",   "#dd8452"),
        "Lost revenue":      ("lost_revenue_goodwill_eur","#9467bd"),
    }
 
    n = len(sweep_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
 
    for ax, (label, filename) in zip(axes, sweep_files.items()):
        results = load_sweep(filename)
        sizes   = [r["fleet_size"] for r in results]
 
        bottoms = np.zeros(len(results))
        for comp_label, (key, color) in cost_components.items():
            values = np.array([r.get(key, 0) for r in results])
            ax.bar(sizes, values, bottom=bottoms,
                   label=comp_label, color=color, alpha=0.85,
                   width=max(1, (sizes[-1] - sizes[0]) / len(sizes) * 0.8))
            bottoms += values
 
        opt = find_optimal(results)
        ax.axvline(opt["fleet_size"], color="black",
                   linestyle="--", linewidth=1.5,
                   label=f"Optimal ({opt['fleet_size']} veh)")
 
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Fleet size (vehicles)", fontsize=11)
        ax.set_ylabel("Total cost (€)", fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
 
    fig.suptitle("Cost breakdown by fleet size", fontsize=13, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_unmet_demand(
    sweep_files:  dict,
    target_unmet: float = 0.05,
    save_path:    str   = None,
):
    """
    Unmet demand rate vs fleet size, with target threshold and profit-optimal marker.
    Kept as a secondary operational lens alongside the profit plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for label, filename in sweep_files.items():
        results     = load_sweep(filename)
        fleet_sizes = [r["fleet_size"]       for r in results]
        unmet_rates = [r["unmet_rate"] * 100 for r in results]
 
        ax.plot(fleet_sizes, unmet_rates,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=6)
 
        # mark profit-optimal fleet (may differ from unmet-rate knee)
        opt = find_optimal(results)
        opt_unmet = opt["unmet_rate"] * 100
        ax.annotate(
            f"Profit opt.\n{opt['fleet_size']} veh",
            xy     = (opt["fleet_size"], opt_unmet),
            xytext = (opt["fleet_size"] + 4, opt_unmet + 1.5),
            fontsize   = 8,
            color      = _color(label),
            arrowprops = dict(arrowstyle="->", color=_color(label)),
        )
 
    ax.axhline(target_unmet * 100, color="gray", linestyle="--",
               linewidth=1.5, label=f"Target ({target_unmet:.0%} unmet)")
 
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Fleet size vs. unmet demand rate by relocation policy", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_relocation_cost(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Total relocation cost vs fleet size.
    Skips the baseline (no relocations).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for label, filename in sweep_files.items():
        results  = load_sweep(filename)
        sizes    = [r["fleet_size"]                  for r in results]
        rel_cost = [r.get("relocation_cost_eur", 0) for r in results]
 
        if all(v == 0 for v in rel_cost):
            continue  # skip baseline
 
        ax.plot(sizes, rel_cost,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=6)
 
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Total relocation cost", fontsize=12)
    ax.set_title("Fleet size vs. relocation cost by policy", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_unmet_vs_relocation(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Scatter of unmet-demand rate vs relocation cost, coloured by fleet size.
    Shows the efficiency frontier across policies.
    Colourbar missing in original — added here.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sc_last = None
 
    for label, filename in sweep_files.items():
        results = load_sweep(filename)
        unmet   = [r["unmet_rate"] * 100      for r in results]
        rel_min = [r.get("relocation_min", 0) for r in results]
        sizes   = [r["fleet_size"]            for r in results]
 
        if all(v == 0 for v in rel_min):
            continue
 
        color = _color(label)
        ax.plot(rel_min, unmet, color=color, linewidth=1.5, alpha=0.5, zorder=2)
        sc_last = ax.scatter(rel_min, unmet, c=sizes, cmap="viridis",
                             s=60, zorder=3, label=label,
                             edgecolors=color, linewidths=1.5)
 
    if sc_last is not None:
        cbar = fig.colorbar(sc_last, ax=ax, pad=0.02)
        cbar.set_label("Fleet size (vehicles)", fontsize=10)
 
    ax.set_xlabel("Total relocation time (min)", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Service quality vs. relocation cost — Pareto frontier", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_marginal_profit(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Marginal gross profit gained by adding one extra vehicle (finite differences).
    Turns negative at the profit-optimal fleet size: see exactly where
    diminishing returns set in and the curve crosses zero.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for label, filename in sweep_files.items():
        results = sorted(load_sweep(filename), key=lambda r: r["fleet_size"])
        sizes   = [r["fleet_size"]              for r in results]
        profits = [r["gross_profit_per_day_eur"] for r in results]
 
        # central finite difference
        marginal = []
        steps    = []
        for i in range(1, len(profits) - 1):
            dp = (profits[i + 1] - profits[i - 1]) / (sizes[i + 1] - sizes[i - 1])
            marginal.append(dp)
            steps.append(sizes[i])
 
        ax.plot(steps, marginal,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5)
 
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5,
               label="Marginal profit = 0 (optimum)")
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Marginal profit per additional vehicle (€/day)", fontsize=12)
    ax.set_title("Diminishing returns: marginal profit vs fleet size", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.1f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()

 
def plot_lost_rev_vs_vehicles(
    sweep_files: dict,
    save_path:   str = None,
):
    """
    Lost revenue (from unmet demand) vs vehicle ownership cost on the same axes.
    As fleet grows: lost revenue falls, vehicle cost rises.
    Their intersection is the cost-minimising sweet spot, which closely
    tracks the profit-maximising fleet size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for label, filename in sweep_files.items():
        results = sorted(load_sweep(filename), key=lambda r: r["fleet_size"])
        sizes   = [r["fleet_size"]          for r in results]
        lost    = [r["lost_revenue_eur"]    for r in results]
        veh     = [r["vehicle_cost_eur"]    for r in results]
 
        c = _color(label)
        ax.plot(sizes, lost, color=c, linewidth=2,
                marker=_marker(label), markersize=5,
                label=f"{label} — lost revenue")
        ax.plot(sizes, veh,  color=c, linewidth=2,
                linestyle="--", markersize=5,
                label=f"{label} — vehicle cost")
 
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("EUR (total simulation period)", fontsize=12)
    ax.set_title("Core trade-off: lost revenue vs vehicle ownership cost", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()
 
 
# ------------------------------------------------------------------ #
#  Run all plots at once                                             #
# ------------------------------------------------------------------ #
 
def plot_all(
    sweep_files:  dict,
    target_unmet: float = 0.05,
    output_dir:   str   = None,
):
    """
    Render all eight plots. If output_dir is given, each plot is saved there.
    """
    def path(name):
        if output_dir is None:
            return None
        return str(Path(output_dir) / name)
 
    print_optimal_summary(sweep_files)
 
    plot_profit(                 sweep_files, save_path=path("profit_vs_fleet.png"))
    plot_revenue_vs_cost(        sweep_files, save_path=path("revenue_vs_cost.png"))
    plot_cost_breakdown(         sweep_files, save_path=path("cost_breakdown.png"))
    plot_unmet_demand(           sweep_files, target_unmet=target_unmet,
                                    save_path=path("unmet_vs_fleet.png"))
    plot_relocation_cost(        sweep_files, save_path=path("relocation_cost.png"))
    plot_unmet_vs_relocation(    sweep_files, save_path=path("pareto.png"))
    plot_marginal_profit(        sweep_files, save_path=path("marginal_profit.png"))
    plot_lost_rev_vs_vehicles(   sweep_files, save_path=path("tradeoff.png"))


if __name__ == "__main__":
 
    policies = {
        # "Baseline":  dict(policy_name="baseline"),
        "Reactive":  dict(policy_name="reactive"),
        # "Proactive": dict(policy_name="proactive"),
        # "Nightly":   dict(policy_name="nightly"),
    }
 
    sweep_files = {}
    model = FinancialModel()
 
    for label, kwargs in policies.items():
        filename = f"fleet_sweep_{label.lower()}_smaller_steps"
        results  = run_fleet_sweep(
            fleet_min  = 500,
            fleet_max  = 800,
            fleet_step = 50,
            n_days     = 30,
            **kwargs,
        )
        save_sweep(results, filename)
        sweep_files[label] = filename
 
    
    plot_all(sweep_files, target_unmet=0.05, output_dir="./data/metrics/fleet_sweep/plots")