"""
1_cash_fleet_sweep.py  -  Wide cash-based fleet sweep (Step 1).

Runs the simulation across a wide fleet range (300-800; step 25) for all
four policies with cash-based profit (goodwill multiplier = 0, i.e. no lost
revenue). Default policy parameters are used; no tuning at this stage.

This step reveals that cash-based profit cannot differentiate between
relocation and no-relocation configurations, motivating the introduction
of a goodwill objective from Step 2 onwards.

Key quantities recorded per (policy, fleet_size):
  - Profit-optimal fleet size per policy
  - Smallest fleet size at which the no-relocation baseline reaches the
    5% unmet demand service target  (service quality reference point)

Produces five figures:
  fig_1_profit_vs_fleet       Gross profit per day vs fleet size — all policies (PRIMARY)
  fig_1_unmet_vs_fleet        Unmet demand rate vs fleet size — all policies
  fig_1_profit_and_unmet      Gross profit + unmet rate on dual y-axis — all policies
  fig_1_relocation_cost       Relocation cost vs fleet size (baseline excluded)
  fig_1_profit_difference     Profit advantage of each relocation policy over baseline

Usage
-----
  python 1_cash_fleet_sweep.py                  # run sweep + all plots
  python 1_cash_fleet_sweep.py --plot           # plot only (data must exist)
  python 1_cash_fleet_sweep.py --skip-existing  # skip policies whose JSON exists
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.simulation.engine import SimulationEngine
from src.analysis.metrics import Metrics
from src.analysis.financials import FinancialResults, FinancialModel
from src.relocation.create_policy import load_policy
from src.optimization import worker_pool_size as ws
from src import config


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

FLEET_MIN    = 300
FLEET_MAX    = 800
FLEET_STEP   = 25
N_DAYS       = 30
TARGET_UNMET = 0.05

# Worker counts during sweep — fixed reasonable values, no optimisation yet
WORKER_COUNTS = {
    "baseline":  1,
    "reactive":  300,
    "nightly":   300,
    "proactive": 100,
}

DATA_DIR  = Path("./data/metrics/1_cash_fleet")
PLOTS_DIR = DATA_DIR / "plots"

# Policy configs — default parameters, no tuning at this stage
POLICIES = {
    "Baseline":  dict(policy_name="baseline"),
    "Reactive":  dict(policy_name="reactive"),
    "Nightly":   dict(policy_name="nightly"),
    "Proactive": dict(policy_name="proactive"),
}

# Colors and markers drawn from final_param.py OPTIMAL dict
_COLORS = {
    "Baseline":  "#00FF99",
    "Reactive":  "#00CCFF",
    "Nightly":   "#9933FF",
    "Proactive": "#FF6EC7",
}
_MARKERS = {
    "Baseline":  "o",
    "Reactive":  "s",
    "Nightly":   "D",
    "Proactive": "^",
}

def _color(label):  return _COLORS.get(label, "#888888")
def _marker(label): return _MARKERS.get(label, "o")


# ════════════════════════════════════════════════════════════════════════════
#  IO HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(data, name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → saved {path}")


def _load_json(name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def _exists(name: str) -> bool:
    return (DATA_DIR / f"{name}.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP
# ════════════════════════════════════════════════════════════════════════════

def run_sweep(label: str, policy_kwargs: dict, model: FinancialModel) -> list[dict]:
    """Run fleet sweep for one policy; return results list."""
    policy_name = policy_kwargs["policy_name"]
    n_workers   = WORKER_COUNTS.get(policy_name, 30)
    sizes       = range(FLEET_MIN, FLEET_MAX + 1, FLEET_STEP)

    print(f"\n  {label}  ({len(list(sizes))} fleet sizes, {n_workers} workers)")

    results = []
    with ws._patch_workers(n_workers):
        for fleet_size in sizes:
            policy = load_policy(**policy_kwargs)
            engine = SimulationEngine(
                n_days              = N_DAYS,
                policy              = policy,
                fleet_size_override = fleet_size,
                verbose             = False,
            )
            engine.run(verbose=False)

            m   = Metrics(engine.event_log, fleet_size=fleet_size)
            fin = FinancialResults(engine.event_log, fleet_size=fleet_size, model=model)
            rel = m.relocation_stats()
            wor = m.worker_stats()

            result = {
                # identity
                "fleet_size":               fleet_size,
                "policy":                   policy_name,
                "label":                    label,
                "goodwill_multiplier":      model.goodwill_multiplier,

                # operational
                "unmet_rate":               round(m.unmet_rate, 4),
                "unmet_total":              m.total_unmet,
                "total_requests":           m.total_requests,
                "total_served":             m.total_served,
                "relocations":              m.total_relocations,
                "relocation_min":           rel["total_min"],
                "relocation_km":            rel["total_km"],
                "relocations_queued":       wor["relocations_queued"],
                "relocations_forgotten":    wor["relocations_forgotten"],
                "peak_workers":             wor["peak_workers_active"],

                # financial — revenue
                "total_revenue_eur":        round(fin.total_revenue, 2),
                "avg_revenue_per_trip_eur": round(fin.avg_revenue_per_trip, 2),
                "lost_revenue_eur":         round(fin.lost_revenue, 2),

                # financial — costs
                "vehicle_cost_eur":         round(fin.total_vehicle_cost, 2),
                "fuel_cost_eur":            round(fin.total_fuel_cost, 2),
                "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
                "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
                "total_cost_eur":           round(fin.total_cost, 2),

                # financial — profit
                "gross_profit_eur":         round(fin.gross_profit, 2),
                "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
                "cost_per_trip_eur":        round(fin.cost_per_trip, 2),
            }

            results.append(result)

            sign = "+" if result["gross_profit_per_day_eur"] >= 0 else ""
            print(
                f"    fleet={fleet_size:>3}  "
                f"unmet={m.unmet_rate:>5.1%}  "
                f"served={m.total_served:>5}  "
                f"profit/day={sign}{result['gross_profit_per_day_eur']:>8.0f}€  "
                f"reloc={m.total_relocations:>4}"
            )

    return results


def find_optimal(results: list[dict]) -> dict:
    """Return the result row with the highest gross profit per day."""
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


def find_knee(results: list[dict], target_unmet: float = TARGET_UNMET) -> dict | None:
    """Return the smallest fleet where unmet rate first reaches target_unmet."""
    for r in sorted(results, key=lambda x: x["fleet_size"]):
        if r["unmet_rate"] <= target_unmet:
            return r
    return None


def run_all_sweeps(skip_existing: bool = True) -> dict:
    """Run fleet sweep for all four policies. Returns {label: results_list}."""
    _ensure_dirs()
    model = FinancialModel()   # goodwill_multiplier=0.0 from config

    print(f"\n{'═'*60}")
    print(f"  STEP 1 — Wide fleet sweep (cash-based, no lost revenue)")
    print(f"  Fleet: {FLEET_MIN}–{FLEET_MAX}, step {FLEET_STEP}")
    print(f"  N_DAYS: {N_DAYS}")
    print(f"  goodwill_multiplier: {model.goodwill_multiplier}")
    print(f"{'═'*60}")

    all_results = {}

    for label, kwargs in POLICIES.items():
        name = f"sweep_{label.lower()}"
        if skip_existing and _exists(name):
            print(f"\n  {label}: skipping (data exists)")
            all_results[label] = _load_json(name)
            continue
        results = run_sweep(label, kwargs, model)
        _save_json(results, name)
        all_results[label] = results

    # summary table
    print(f"\n{'═'*70}")
    print(f"  PROFIT-OPTIMAL FLEET SIZE PER POLICY  (cash-based)")
    print(f"{'═'*70}")
    print(f"  {'Policy':<12} {'Opt fleet':>9} {'Profit/day':>12} "
          f"{'Unmet':>7} {'Reloc cost':>12}")
    print(f"  {'-'*65}")
    for label, results in all_results.items():
        opt  = find_optimal(results)
        knee = find_knee(results, TARGET_UNMET)
        print(
            f"  {label:<12} "
            f"{opt['fleet_size']:>9}  "
            f"€{opt['gross_profit_per_day_eur']:>10,.0f}  "
            f"{opt['unmet_rate']:>6.1%}  "
            f"€{opt['relocation_cost_eur']:>10,.0f}"
        )
        if knee:
            print(f"  {'':12} "
                  f"{TARGET_UNMET:.0%} unmet knee: {knee['fleet_size']} vehicles")
    print(f"{'═'*70}\n")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_profit(all_results: dict):
    """
    fig_1_profit_vs_fleet — Gross profit per day vs fleet size, all policies.
    PRIMARY plot. Annotates profit optimum per policy.
    Marks break-even (€0) and the baseline's 5%-unmet knee as a fleet reference.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, results in all_results.items():
        sizes   = [r["fleet_size"]               for r in results]
        profits = [r["gross_profit_per_day_eur"]  for r in results]

        ax.plot(sizes, profits,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5)

        opt  = find_optimal(results)
        sign = "+" if opt["gross_profit_per_day_eur"] >= 0 else ""
        ax.annotate(
            f"{opt['fleet_size']} veh\n{sign}€{opt['gross_profit_per_day_eur']:,.0f}/day",
            xy     = (opt["fleet_size"], opt["gross_profit_per_day_eur"]),
            xytext = (opt["fleet_size"], opt["gross_profit_per_day_eur"] - 2000),
            fontsize   = 8,
            color      = _color(label),
            arrowprops = dict(arrowstyle="->", color=_color(label), lw=1.2),
            va = "center",
        )

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title("Fleet size vs daily gross profit\n"
                 "(cash-based: no lost revenue included)",
                 fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig_1_profit_vs_fleet.png")
    plt.show()
    plt.close(fig)


def plot_unmet(all_results: dict):
    """
    fig_1_unmet_vs_fleet — Unmet demand rate vs fleet size, all policies.
    Shows at which fleet size each policy reaches the 5% target.
    Stars (★) mark the profit-optimal fleet size per policy.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, results in all_results.items():
        sizes = [r["fleet_size"]       for r in results]
        unmet = [r["unmet_rate"] * 100 for r in results]

        ax.plot(sizes, unmet,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5)

        opt       = find_optimal(results)
        opt_unmet = opt["unmet_rate"] * 100
        ax.scatter([opt["fleet_size"]], [opt_unmet],
                   color=_color(label), s=120, zorder=5,
                   marker="*", edgecolors="black", linewidths=0.5)

    ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--",
               linewidth=1.5, label=f"Target: {TARGET_UNMET:.0%} unmet")

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Fleet size vs unmet demand rate\n"
                 "(★ = profit-optimal fleet size per policy)",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(fig, "fig_1_unmet_vs_fleet.png")
    plt.show()
    plt.close(fig)


def plot_profit_and_unmet(all_results: dict):
    """
    fig_1_profit_and_unmet — Gross profit (left axis) and unmet demand rate
    (right axis) on the same figure. Solid lines = profit, dashed = unmet rate.
    Shows the trade-off between financial performance and service quality.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    for label, results in all_results.items():
        sizes   = [r["fleet_size"]               for r in results]
        profits = [r["gross_profit_per_day_eur"]  for r in results]
        unmet   = [r["unmet_rate"] * 100          for r in results]

        ax1.plot(sizes, profits,
                 label=f"{label} — profit",
                 color=_color(label),
                 marker=_marker(label), linewidth=2, markersize=5)
        ax2.plot(sizes, unmet,
                 color=_color(label), linewidth=1.2,
                 linestyle="--", alpha=0.6)

    ax1.axhline(0, color="black", linestyle=":", linewidth=1.5,
                alpha=0.5, label="Break-even (€0)")
    ax2.axhline(TARGET_UNMET * 100, color="gray", linestyle="-.",
                linewidth=1.2, alpha=0.7,
                label=f"{TARGET_UNMET:.0%} unmet target")

    ax1.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%) — dashed lines", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(bottom=0)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax1.set_title("Fleet size vs profit and service quality — all policies\n"
                  "(solid lines: profit/day, dashed lines: unmet rate)",
                  fontsize=13)

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig_1_profit_and_unmet.png")
    plt.show()
    plt.close(fig)


def plot_relocation_cost(all_results: dict):
    """
    fig_1_relocation_cost — Total relocation cost vs fleet size (baseline excluded).
    Dotted verticals mark the profit-optimal fleet size per policy.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    any_plotted = False
    for label, results in all_results.items():
        sizes    = [r["fleet_size"]          for r in results]
        rel_cost = [r["relocation_cost_eur"] for r in results]

        if all(v == 0 for v in rel_cost):
            continue   # skip baseline

        ax.plot(sizes, rel_cost,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5)
        any_plotted = True

        opt = find_optimal(results)
        ax.axvline(opt["fleet_size"], color=_color(label),
                   linestyle=":", linewidth=1.2, alpha=0.6)

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Total relocation cost (€)", fontsize=12)
    ax.set_title("Fleet size vs relocation cost by policy\n"
                 "(dotted verticals: profit-optimal fleet per policy)",
                 fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(fig, "fig_1_relocation_cost.png")
    plt.show()
    plt.close(fig)


def plot_profit_difference(all_results: dict):
    """
    fig_1_profit_difference — Profit difference vs baseline at each fleet size.
    Zero line = baseline. Positive = policy beats baseline on cash profit.
    Directly shows that relocation does not add cash value, motivating Step 2.
    """
    baseline_results = all_results.get("Baseline")
    if not baseline_results:
        print("  No baseline data — skipping profit difference plot")
        return

    baseline_by_fleet = {r["fleet_size"]: r["gross_profit_per_day_eur"]
                         for r in baseline_results}

    fig, ax = plt.subplots(figsize=(12, 7))

    for label, results in all_results.items():
        if label == "Baseline":
            continue

        sizes = [r["fleet_size"] for r in results
                 if r["fleet_size"] in baseline_by_fleet]
        diff  = [r["gross_profit_per_day_eur"] - baseline_by_fleet[r["fleet_size"]]
                 for r in results
                 if r["fleet_size"] in baseline_by_fleet]

        ax.plot(sizes, diff,
                label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=6)

        opt = find_optimal(results)
        if opt["fleet_size"] in baseline_by_fleet:
            opt_diff = (opt["gross_profit_per_day_eur"]
                        - baseline_by_fleet[opt["fleet_size"]])
            sign = "+" if opt_diff >= 0 else ""
            ax.annotate(
                f"{opt['fleet_size']} veh\n{sign}€{opt_diff:,.0f}/day",
                xy     = (opt["fleet_size"], opt_diff),
                xytext = (opt["fleet_size"] + 10, opt_diff),
                fontsize   = 8,
                color      = _color(label),
                arrowprops = dict(arrowstyle="->", color=_color(label), lw=1.2),
                va = "center",
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=2,
               label="Baseline (reference)")

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Profit difference vs baseline (€/day)", fontsize=12)
    ax.set_title("Cash profit advantage of relocation policies over baseline\n"
                 "(near-zero differences motivate goodwill objective in Step 2)",
                 fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig_1_profit_difference.png")
    plt.show()
    plt.close(fig)


def plot_all(all_results: dict):
    """Render all Step 1 figures."""
    print(f"\n{'═'*60}")
    print(f"  Generating Step 1 plots  →  {PLOTS_DIR}")
    print(f"{'═'*60}\n")

    plot_profit(all_results)
    plot_unmet(all_results)
    plot_profit_and_unmet(all_results)
    plot_relocation_cost(all_results)
    plot_profit_difference(all_results)

    print(f"\n  All plots saved to {PLOTS_DIR}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 1: wide cash-based fleet sweep")
    parser.add_argument("--plot", action="store_true",
                        help="Plot only — skip sweep (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip policies whose JSON file already exists")
    args = parser.parse_args()

    _ensure_dirs()

    if args.plot:
        all_results = {}
        for label in POLICIES:
            name = f"sweep_{label.lower()}"
            if _exists(name):
                all_results[label] = _load_json(name)
            else:
                print(f"  WARNING: {name}.json not found — run without --plot first")
        if all_results:
            plot_all(all_results)
    else:
        all_results = run_all_sweeps(skip_existing=args.skip_existing)
        plot_all(all_results)


if __name__ == "__main__":
    main()
