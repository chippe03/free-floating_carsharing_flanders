"""
7_sensitivity_demand.py  -  Demand sensitivity analysis (Step 7b).

Evaluates all four policies at varying demand levels to assess whether the
policy rankings from Step 6 hold when realised demand is lower or higher
than the calibrated baseline.

Two modes
---------
FIXED FLEET (default):
  Each policy runs at its Step 6 optimal fleet size (from final_param.OPTIMAL)
  across all demand levels. Shows how profit and unmet demand respond to demand
  shocks without any operational adjustment.

REOPTIMISED FLEET (--reoptimise):
  For each demand level, the baseline fleet sweep is re-run to find the
  profit-optimal fleet, and all policies are evaluated at that fleet.
  Simulates a scenario where the operator adjusts fleet deployment in response
  to a structural demand change.

Parameters and colors are loaded from final_param.OPTIMAL.

Demand multipliers (default): 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00

Produces four figures:
  fig_7d_profit_vs_demand     Profit per policy vs demand multiplier
  fig_7d_unmet_vs_demand      Unmet demand rate vs demand multiplier
  fig_7d_advantage_vs_demand  Profit advantage vs baseline per policy
  fig_7d_fleet_vs_demand      Optimal fleet size vs demand (reoptimise mode only)

Usage
-----
  python 7_sensitivity_demand.py                         # fixed fleet, all multipliers
  python 7_sensitivity_demand.py --reoptimise            # re-optimise fleet per demand level
  python 7_sensitivity_demand.py --multipliers 0.5 0.75 1.0 1.25
  python 7_sensitivity_demand.py --plot                  # plots only (data must exist)
  python 7_sensitivity_demand.py --skip-existing         # resume interrupted run
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
from src import config as sim_config

from final_param import OPTIMAL


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

N_DAYS   = 30
GOODWILL = 1.0

DEFAULT_MULTIPLIERS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

DEMAND_KEY_WEEKDAY = "base_rate_weekday"
DEMAND_KEY_WEEKEND = "base_rate_weekend"

POLICY_ORDER = ["baseline", "reactive", "nightly", "proactive"]

# Fleet sweep range for reoptimisation mode
REOPT_FLEET_MIN  = 100
REOPT_FLEET_MAX  = 800
REOPT_FLEET_STEP = 50

DATA_DIR  = Path("./data/metrics/7_demand")
PLOTS_DIR = DATA_DIR / "plots"

# Fleet sizes, worker counts, params and colors from final_param.OPTIMAL
_COLORS  = {k: v["color"] for k, v in OPTIMAL.items()}
_LABELS  = {k: v["label"] for k, v in OPTIMAL.items()}


# ════════════════════════════════════════════════════════════════════════════
#  IO
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(data, name: str = "demand_sensitivity"):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


def _load(name: str = "demand_sensitivity"):
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def _exists(name: str = "demand_sensitivity") -> bool:
    return (DATA_DIR / f"{name}.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  DEMAND PATCHING
# ════════════════════════════════════════════════════════════════════════════

def _get_base_rates() -> tuple[float, float]:
    d = sim_config.simulation["demand"]
    return d[DEMAND_KEY_WEEKDAY], d[DEMAND_KEY_WEEKEND]


def _patch_demand(multiplier: float, base_weekday: float, base_weekend: float):
    sim_config.simulation["demand"][DEMAND_KEY_WEEKDAY] = base_weekday * multiplier
    sim_config.simulation["demand"][DEMAND_KEY_WEEKEND] = base_weekend * multiplier


def _restore_demand(base_weekday: float, base_weekend: float):
    sim_config.simulation["demand"][DEMAND_KEY_WEEKDAY] = base_weekday
    sim_config.simulation["demand"][DEMAND_KEY_WEEKEND] = base_weekend


# ════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_one(policy_key: str, fleet_size: int, multiplier: float,
            base_weekday: float, base_weekend: float) -> dict:
    cfg   = OPTIMAL[policy_key]
    model = FinancialModel(goodwill_multiplier=GOODWILL)

    _patch_demand(multiplier, base_weekday, base_weekend)
    try:
        policy = load_policy(policy_name=cfg["policy_name"], **cfg["params"])
        engine = SimulationEngine(
            n_days=N_DAYS, policy=policy,
            fleet_size_override=fleet_size, verbose=False,
        )
        with ws._patch_workers(cfg["workers"]):
            engine.run(verbose=False)
    finally:
        _restore_demand(base_weekday, base_weekend)

    m   = Metrics(engine.event_log, fleet_size=fleet_size)
    fin = FinancialResults(engine.event_log, fleet_size=fleet_size, model=model)

    return {
        "policy":                   policy_key,
        "label":                    cfg["label"],
        "fleet_size":               fleet_size,
        "n_workers":                cfg["workers"],
        "demand_multiplier":        multiplier,
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "total_unmet":              m.total_unmet,
        "relocations":              m.total_relocations,
        "lost_revenue_eur":         round(fin.lost_revenue, 2),
        "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
        "gross_profit_eur":         round(fin.gross_profit, 2),
        "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
    }


def find_optimal_fleet(multiplier: float, base_weekday: float, base_weekend: float) -> int:
    """Re-sweep fleet size for baseline under given demand multiplier."""
    best_fleet  = OPTIMAL["baseline"]["fleet"]
    best_profit = -float("inf")
    print(f"    Fleet sweep for demand={multiplier:.2f}x ...")

    for fleet_size in range(REOPT_FLEET_MIN, REOPT_FLEET_MAX + 1, REOPT_FLEET_STEP):
        r = run_one("baseline", fleet_size, multiplier, base_weekday, base_weekend)
        if r["gross_profit_per_day_eur"] > best_profit:
            best_profit = r["gross_profit_per_day_eur"]
            best_fleet  = fleet_size

    print(f"    Optimal fleet at {multiplier:.2f}x demand: {best_fleet} veh "
          f"(€{best_profit:,.0f}/day)")
    return best_fleet


# ════════════════════════════════════════════════════════════════════════════
#  MAIN SWEEP
# ════════════════════════════════════════════════════════════════════════════

def run_sensitivity(multipliers: list[float], reoptimise: bool = False,
                    skip_existing: bool = False) -> tuple[list[dict], str]:
    name    = "demand_sensitivity_reopt" if reoptimise else "demand_sensitivity"
    results = _load(name) if (skip_existing and _exists(name)) else []
    done    = {(r["policy"], r["demand_multiplier"]) for r in results}

    base_weekday, base_weekend = _get_base_rates()
    print(f"\n  Base demand rates: weekday={base_weekday:.4f}, weekend={base_weekend:.4f}")

    opt_fleets: dict[float, int] = {}

    for multiplier in multipliers:
        print(f"\n  Demand multiplier: {multiplier:.2f}×  "
              f"(weekday={base_weekday*multiplier:.4f}, "
              f"weekend={base_weekend*multiplier:.4f})")

        if reoptimise:
            cache_key = round(multiplier, 3)
            if cache_key not in opt_fleets:
                opt_fleets[cache_key] = find_optimal_fleet(
                    multiplier, base_weekday, base_weekend)
                _save(opt_fleets, "optimal_fleets_by_demand")
            fleet_sizes = {p: opt_fleets[cache_key] for p in POLICY_ORDER}
        else:
            fleet_sizes = {p: OPTIMAL[p]["fleet"] for p in POLICY_ORDER}

        for policy_key in POLICY_ORDER:
            if (policy_key, float(f"{multiplier:.2f}")) in done:
                print(f"    {policy_key}: skipping (already done)")
                continue

            fleet = fleet_sizes[policy_key]
            r     = run_one(policy_key, fleet, multiplier, base_weekday, base_weekend)
            sign  = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    {policy_key:<12} fleet={fleet}  "
                  f"unmet={r['unmet_rate']:.1%}  "
                  f"profit/day={sign}€{r['gross_profit_per_day_eur']:>9,.0f}")
            results.append(r)
            _save(results, name)

    return results, name


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], multipliers: list[float]):
    by_m: dict[float, dict] = {}
    for r in results:
        by_m.setdefault(r["demand_multiplier"], {})[r["policy"]] = r

    print(f"\n{'═'*85}")
    print(f"  STEP 7b — DEMAND SENSITIVITY SUMMARY (goodwill=1.0)")
    print(f"{'═'*85}")
    print(f"  {'Demand':>8}  {'Policy':<12} {'Profit/day':>12} "
          f"{'vs Baseline':>13} {'Unmet':>7} {'Fleet':>6}")
    print(f"  {'-'*82}")

    for m in sorted(by_m.keys()):
        base_profit = by_m[m].get("baseline", {}).get("gross_profit_per_day_eur", 0)
        for p in POLICY_ORDER:
            r = by_m[m].get(p)
            if not r:
                continue
            diff   = r["gross_profit_per_day_eur"] - base_profit
            sign   = "+" if diff >= 0 else ""
            marker = " ←" if p == "baseline" else ""
            print(f"  {m:>7.2f}×  {OPTIMAL[p]['label']:<12} "
                  f"€{r['gross_profit_per_day_eur']:>10,.0f}  "
                  f"{sign}€{diff:>10,.0f}  "
                  f"{r['unmet_rate']:>6.1%}  "
                  f"{r['fleet_size']:>5}{marker}")
        print()


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def _build_curves(results: list[dict]) -> dict:
    """Return {policy_key: {multiplier: result}}."""
    curves: dict[str, dict] = {p: {} for p in POLICY_ORDER}
    for r in results:
        curves[r["policy"]][r["demand_multiplier"]] = r
    return curves


def plot_profit_vs_demand(results: list[dict], multipliers: list[float]):
    """fig_7d_profit_vs_demand — profit per policy vs demand multiplier."""
    curves = _build_curves(results)
    fig, ax = plt.subplots(figsize=(12, 7))

    for p in POLICY_ORDER:
        ms   = sorted(curves[p].keys())
        prof = [curves[p][m]["gross_profit_per_day_eur"] for m in ms]
        ax.plot(ms, prof, color=_COLORS[p], linewidth=2.5,
                marker="o", markersize=7, label=_LABELS[p])

    ax.axvline(1.0, color="gray", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Calibrated demand (1.0×)")
    ax.set_xlabel("Demand multiplier", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        "Profit per day vs demand level — all policies at optimal configurations\n"
        "(goodwill=1.0; multiplier applied to both weekday and weekend base rates)",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_7d_profit_vs_demand.png")
    plt.show(); plt.close(fig)


def plot_unmet_vs_demand(results: list[dict], multipliers: list[float]):
    """fig_7d_unmet_vs_demand — unmet demand rate vs demand multiplier."""
    curves = _build_curves(results)
    fig, ax = plt.subplots(figsize=(12, 7))

    for p in POLICY_ORDER:
        ms    = sorted(curves[p].keys())
        unmet = [curves[p][m]["unmet_rate"] * 100 for m in ms]
        ax.plot(ms, unmet, color=_COLORS[p], linewidth=2.5,
                marker="o", markersize=7, label=_LABELS[p])

    ax.axhline(5.0, color="gray", linewidth=1.5, linestyle="--",
               label="5% service target")
    ax.axvline(1.0, color="gray", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Calibrated demand (1.0×)")
    ax.set_xlabel("Demand multiplier", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title(
        "Unmet demand rate vs demand level — all policies at optimal configurations",
        fontsize=12
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_7d_unmet_vs_demand.png")
    plt.show(); plt.close(fig)


def plot_advantage_vs_demand(results: list[dict], multipliers: list[float]):
    """fig_7d_advantage_vs_demand — profit advantage vs baseline per policy."""
    curves = _build_curves(results)
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.axhline(0, color="black", linewidth=2, linestyle="--",
               label="Baseline (reference)")
    ax.axvline(1.0, color="gray", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Calibrated demand (1.0×)")

    for p in ["reactive", "proactive", "nightly"]:
        ms    = sorted(curves[p].keys())
        diffs = [curves[p][m]["gross_profit_per_day_eur"] -
                 curves["baseline"].get(m, {}).get("gross_profit_per_day_eur", 0)
                 for m in ms]
        ax.plot(ms, diffs, color=_COLORS[p], linewidth=2.5,
                marker="o", markersize=7, label=_LABELS[p])
        ax.fill_between(ms, diffs, 0,
                        where=[d > 0 for d in diffs],
                        color=_COLORS[p], alpha=0.07)

    ax.set_xlabel("Demand multiplier", fontsize=12)
    ax.set_ylabel("Profit advantage vs baseline (€/day)", fontsize=12)
    ax.set_title(
        "Profit advantage over baseline vs demand level\n"
        "Positive = policy beats baseline at that demand level",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_7d_advantage_vs_demand.png")
    plt.show(); plt.close(fig)


def plot_optimal_fleet_vs_demand(results: list[dict]):
    """fig_7d_fleet_vs_demand — optimal fleet size vs demand (reoptimise mode only)."""
    baseline_results = [r for r in results if r["policy"] == "baseline"]
    if not baseline_results:
        return

    by_m: dict[float, dict] = {}
    for r in baseline_results:
        m = r["demand_multiplier"]
        if m not in by_m or r["gross_profit_per_day_eur"] > by_m[m]["gross_profit_per_day_eur"]:
            by_m[m] = r

    ms     = sorted(by_m.keys())
    fleets = [by_m[m]["fleet_size"] for m in ms]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ms, fleets, color=_COLORS["baseline"], linewidth=2.5,
            marker="o", markersize=8, label="Profit-optimal fleet (baseline)")
    ax.axvline(1.0, color="gray", linewidth=1.2, linestyle=":",
               alpha=0.7, label="Calibrated demand (1.0×)")
    ax.set_xlabel("Demand multiplier", fontsize=12)
    ax.set_ylabel("Optimal fleet size (vehicles)", fontsize=12)
    ax.set_title("Profit-optimal fleet size vs demand level", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_7d_fleet_vs_demand.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 7b: demand sensitivity analysis"
    )
    parser.add_argument("--multipliers", nargs="+", type=float,
                        default=DEFAULT_MULTIPLIERS,
                        help="Demand multipliers to test")
    parser.add_argument("--reoptimise",    action="store_true",
                        help="Re-optimise fleet size per demand level")
    parser.add_argument("--plot",          action="store_true",
                        help="Plot only (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume an interrupted sweep")
    args = parser.parse_args()

    _ensure_dirs()
    multipliers = sorted(args.multipliers)

    print(f"\n{'═'*60}")
    print(f"  STEP 7b — Demand sensitivity analysis")
    print(f"  Multipliers: {multipliers}")
    print(f"  Mode: {'fleet re-optimised per demand level' if args.reoptimise else 'fixed fleet (final_param.OPTIMAL)'}")
    print(f"  N_DAYS: {N_DAYS}  |  Goodwill: {GOODWILL}")
    print(f"{'═'*60}")

    name = "demand_sensitivity_reopt" if args.reoptimise else "demand_sensitivity"

    if args.plot and _exists(name):
        results = _load(name)
    else:
        results, name = run_sensitivity(
            multipliers,
            reoptimise=args.reoptimise,
            skip_existing=args.skip_existing,
        )

    print_summary(results, multipliers)
    plot_profit_vs_demand(results, multipliers)
    plot_unmet_vs_demand(results, multipliers)
    plot_advantage_vs_demand(results, multipliers)

    if args.reoptimise:
        plot_optimal_fleet_vs_demand(results)

    print(f"\n  Figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
