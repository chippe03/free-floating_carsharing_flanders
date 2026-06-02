"""
2_parameter_sweep.py  -  Parameter sweep with goodwill objective (Step 2).

Each relocation policy is optimised at its Step 1 cash-optimal fleet size
using a goodwill multiplier of 1.0 (lost revenue included as a penalty).

  Cash-optimal fleet sizes (from Step 1):
    Reactive:   725 vehicles
    Nightly:    600 vehicles
    Proactive:  575 vehicles

  Parameter grids swept:
    Reactive:   alpha  ×  check_interval_minutes
    Nightly:    trigger_hour  (check_interval fixed at 60 min)
    Proactive:  planning_period  ×  lookahead_period  ×  check_interval

The goodwill-optimal parameters identified here are saved to
data/metrics/2_parameters/optimal_params.json and held fixed in all
subsequent steps.

Produces ten figures:
  fig_2_reactive_alpha          Reactive: profit vs alpha - cash vs goodwill + unmet rate
  fig_2_reactive_alpha_v2       Reactive: improved version with both optimal verticals marked
  fig_2_reactive_ci             Reactive: profit vs check interval at optimal alpha
  fig_2_nightly_trigger         Nightly: profit vs trigger hour - cash vs goodwill
  fig_2_proactive_heatmaps      Proactive: T×L heatmaps - cash / goodwill / difference
  fig_2_pro_bubble              Proactive: bubble chart T×L, size=profit, colour=check interval
  fig_2_pro_parallel_goodwill   Proactive: parallel coordinates coloured by goodwill profit
  fig_2_pro_parallel_cash       Proactive: parallel coordinates coloured by cash profit
  fig_2_pro_T_ci_by_L           Proactive: grouped bar chart T×ci interaction by L
  fig_2_sensitivity_summary     Summary: sensitivity range under both objectives, all policies
  fig_2_beats_baseline          Bar: goodwill profit at optimal params vs goodwill baseline

Usage
-----
  python 2_parameter_sweep.py                 # run sweep + all plots
  python 2_parameter_sweep.py --plot          # plot only (data must exist)
  python 2_parameter_sweep.py --skip-existing # skip policies already fully swept
"""

import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from src.simulation.engine import SimulationEngine
from src.analysis.metrics import Metrics
from src.analysis.financials import FinancialResults, FinancialModel
from src.relocation.create_policy import load_policy
from src.optimization import worker_pool_size as ws
from src import config


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

N_DAYS   = 30
GOODWILL = 1.0

# Cash-optimal fleet sizes from Step 1
CASH_OPT_FLEET = {
    "reactive":  725,
    "nightly":   600,
    "proactive": 575,
}

WORKER_COUNTS = {
    "baseline":  1,
    "reactive":  300,
    "nightly":   300,
    "proactive": 100,
}

# Parameter grids
ALPHA_VALUES_SMALL      = [10, 25, 50, 75, 100, 125, 150, 175, 200]
ALPHA_FRACTIONS         = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CHECK_INTERVAL_VALUES            = [15, 30, 60, 120]   # reactive
CHECK_INTERVAL_VALUES_PROACTIVE  = [30, 60, 120]       # proactive (15 min excluded)
NIGHTLY_CHECK_INTERVAL  = 60    # fixed - only trigger_hour is swept for nightly
TRIGGER_HOURS           = list(range(0, 4))
PLANNING_PERIOD_VALUES  = [60, 120, 180, 240]
LOOKAHEAD_PERIOD_VALUES = [60, 120, 180, 240]

DATA_DIR  = Path("./data/metrics/2_parameters")
PLOTS_DIR = DATA_DIR / "plots"

# Goodwill profit of baseline at each policy's cash-optimal fleet size
# (used for reference lines in proactive interaction plots)
BASELINE_GW = {
    "reactive":  195738,   # fleet=725
    "proactive": 192998,   # fleet=575
    "nightly":   194235,   # fleet=600
}

# Colors consistent with final_param.py OPTIMAL
_COLORS = {
    "Baseline":  "#00FF99",
    "Reactive":  "#00CCFF",
    "Nightly":   "#9933FF",
    "Proactive": "#FF6EC7",
    # objective colours for cash vs goodwill comparison plots
    "cash":      "#7A1FCC",
    "goodwill":  "#0099CC",
    "unmet":     "#FF3300",
}
_MARKERS = {
    "Baseline":  "o",
    "Reactive":  "s",
    "Nightly":   "D",
    "Proactive": "^",
}

PROACTIVE_EXCLUDE = lambda r: r.get("planning_period_minutes") == r.get("check_interval_minutes")
TARGET_UNMET = 0.05


# ════════════════════════════════════════════════════════════════════════════
#  IO HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(data, name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


def _load(name: str):
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def _exists(name: str) -> bool:
    return (DATA_DIR / f"{name}.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _run(policy_name, fleet_size, goodwill=None, **policy_kwargs):
    """Run one simulation; return result dict."""
    model  = FinancialModel(goodwill_multiplier=goodwill)
    policy = load_policy(policy_name=policy_name, **policy_kwargs)
    engine = SimulationEngine(
        n_days=N_DAYS, policy=policy,
        fleet_size_override=fleet_size, verbose=False
    )
    engine.run(verbose=False)
    m   = Metrics(engine.event_log, fleet_size=fleet_size)
    fin = FinancialResults(engine.event_log, fleet_size=fleet_size, model=model)
    rel = m.relocation_stats()
    return {
        "fleet_size":               fleet_size,
        "policy":                   policy_name,
        "goodwill_multiplier":      model.goodwill_multiplier,
        **policy_kwargs,
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "total_unmet":              m.total_unmet,
        "relocations":              m.total_relocations,
        "relocation_min":           rel["total_min"],
        "relocation_km":            rel["total_km"],
        "total_revenue_eur":        round(fin.total_revenue, 2),
        "lost_revenue_eur":         round(fin.lost_revenue, 2),
        "avg_revenue_per_trip_eur": round(fin.avg_revenue_per_trip, 2),
        "vehicle_cost_eur":         round(fin.total_vehicle_cost, 2),
        "fuel_cost_eur":            round(fin.total_fuel_cost, 2),
        "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
        "total_cost_eur":           round(fin.total_cost, 2),
        "gross_profit_eur":         round(fin.gross_profit, 2),
        "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
    }


def find_optimal(results, key="gross_profit_per_day_eur"):
    return max(results, key=lambda r: r[key])


def enrich(results: list[dict]) -> list[dict]:
    """Add cash_profit_per_day_eur (goodwill profit + lost revenue) / N_DAYS."""
    for r in results:
        r["cash_profit_per_day_eur"] = (
            r["gross_profit_eur"] + r["lost_revenue_eur"]
        ) / (N_DAYS+1)
    return results


def sensitivity_range(values):
    return max(values) - min(values)


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP - REACTIVE
# ════════════════════════════════════════════════════════════════════════════

def sweep_reactive(skip_existing: bool):
    """
    Append-mode sweep: loads any existing results and only runs missing
    (alpha, check_interval) combinations. Safe to interrupt and resume.
    """
    name  = "param_reactive"
    fleet = CASH_OPT_FLEET["reactive"]

    total_weight = sum(
        c["demand_weight"] for c in config.cities["cities"].values()
        if c.get("demand_weight") is not None
    )
    max_alpha   = int(fleet / total_weight)
    frac_alphas = sorted(set(max(1, round(f * max_alpha)) for f in ALPHA_FRACTIONS))
    alphas      = sorted(set(ALPHA_VALUES_SMALL + frac_alphas))

    results = _load(name) if _exists(name) else []
    done    = {(r["alpha"], r["check_interval_minutes"]) for r in results}
    grid    = [(a, ci) for a in alphas for ci in CHECK_INTERVAL_VALUES
               if (a, ci) not in done]

    if not grid:
        print(f"  Reactive: all {len(results)} combinations done - skipping")
        opt = find_optimal(results)
        print(f"  ✓ Optimal: alpha={opt['alpha']} ci={opt['check_interval_minutes']} "
              f"€{opt['gross_profit_per_day_eur']:,.0f}/day  unmet={opt['unmet_rate']:.1%}")
        return results

    print(f"\n  Reactive: fleet={fleet}, {len(done)} done, {len(grid)} remaining")
    with ws._patch_workers(WORKER_COUNTS["reactive"]):
        for alpha, ci in grid:
            r    = _run("reactive", fleet, goodwill=GOODWILL,
                        alpha=alpha, check_interval_minutes=ci)
            sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    alpha={alpha:<4} ci={ci:<4} "
                  f"unmet={r['unmet_rate']:>5.1%} "
                  f"profit/day={sign}{r['gross_profit_per_day_eur']:>8.0f}€")
            results.append(r)
            _save(results, name)

    results = sorted(results, key=lambda r: (r["alpha"], r["check_interval_minutes"]))
    _save(results, name)
    opt = find_optimal(results)
    print(f"  ✓ Reactive optimal: alpha={opt['alpha']} ci={opt['check_interval_minutes']} "
          f"€{opt['gross_profit_per_day_eur']:,.0f}/day  unmet={opt['unmet_rate']:.1%}")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP - NIGHTLY
# ════════════════════════════════════════════════════════════════════════════

def sweep_nightly(skip_existing: bool):
    """
    Only trigger_hour is swept. check_interval is fixed at NIGHTLY_CHECK_INTERVAL
    because the LP fires exactly once per day at trigger_hour regardless of it.
    """
    name  = "param_nightly"
    fleet = CASH_OPT_FLEET["nightly"]

    existing = _load(name) if _exists(name) else []
    results  = [r for r in existing
                if r.get("check_interval_minutes") == NIGHTLY_CHECK_INTERVAL]
    if len(results) < len(existing):
        print(f"  Nightly: discarded {len(existing)-len(results)} anomalous ci results")

    done = {r["trigger_hour"] for r in results}
    grid = [h for h in TRIGGER_HOURS if h not in done]

    if not grid:
        print(f"  Nightly: all {len(results)} trigger hours done - skipping")
        opt = find_optimal(results)
        print(f"  ✓ Optimal: trigger_hour={opt['trigger_hour']:02d}:00 "
              f"€{opt['gross_profit_per_day_eur']:,.0f}/day  unmet={opt['unmet_rate']:.1%}")
        return results

    print(f"\n  Nightly: fleet={fleet}, ci fixed at {NIGHTLY_CHECK_INTERVAL}min, "
          f"{len(done)} done, {len(grid)} remaining")
    with ws._patch_workers(WORKER_COUNTS["nightly"]):
        for hour in grid:
            r    = _run("nightly", fleet, goodwill=GOODWILL,
                        trigger_hour=hour,
                        check_interval_minutes=NIGHTLY_CHECK_INTERVAL)
            sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    hour={hour:02d}:00  "
                  f"unmet={r['unmet_rate']:>5.1%} "
                  f"profit/day={sign}{r['gross_profit_per_day_eur']:>8.0f}€")
            results.append(r)
            _save(results, name)

    opt = find_optimal(results)
    print(f"  ✓ Nightly optimal: trigger_hour={opt['trigger_hour']:02d}:00 "
          f"€{opt['gross_profit_per_day_eur']:,.0f}/day  unmet={opt['unmet_rate']:.1%}")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP - PROACTIVE
# ════════════════════════════════════════════════════════════════════════════

def sweep_proactive(skip_existing: bool):
    """Full grid: planning × lookahead × check_interval. T=ci combinations excluded."""
    name = "param_proactive"
    if skip_existing and _exists(name):
        print(f"  Proactive: skipping (file exists)")
        return _load(name)

    fleet   = CASH_OPT_FLEET["proactive"]
    results = _load(name) if _exists(name) else []
    done    = {(r["planning_period_minutes"], r["lookahead_period_minutes"],
                r["check_interval_minutes"]) for r in results}

    full_grid = [
        (p, l, ci)
        for p, l, ci in itertools.product(
            PLANNING_PERIOD_VALUES, LOOKAHEAD_PERIOD_VALUES,
            CHECK_INTERVAL_VALUES_PROACTIVE)
        if p % ci == 0 and l % ci == 0
    ]
    grid = [(p, l, ci) for p, l, ci in full_grid if (p, l, ci) not in done]

    if not grid:
        print(f"  Proactive: all combinations done - skipping")
        return results

    print(f"\n  Proactive: fleet={fleet}, grid={len(full_grid)}, "
          f"{len(done)} done, {len(grid)} remaining")
    with ws._patch_workers(WORKER_COUNTS["proactive"]):
        for planning, lookahead, ci in grid:
            r    = _run("proactive", fleet, goodwill=GOODWILL,
                        planning_period_minutes=planning,
                        lookahead_period_minutes=lookahead,
                        time_step_minutes=ci,
                        check_interval_minutes=ci)
            sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    T={planning:<4} L={lookahead:<4} ci={ci:<4} "
                  f"unmet={r['unmet_rate']:>5.1%} "
                  f"profit/day={sign}{r['gross_profit_per_day_eur']:>8.0f}€")
            results.append(r)
            _save(results, name)

    opt = find_optimal(results)
    print(f"  ✓ Proactive optimal: T={opt['planning_period_minutes']} "
          f"L={opt['lookahead_period_minutes']} ci={opt['check_interval_minutes']} "
          f"€{opt['gross_profit_per_day_eur']:,.0f}/day  unmet={opt['unmet_rate']:.1%}")
    _save(results, name)
    return results


# ════════════════════════════════════════════════════════════════════════════
#  RUN ALL SWEEPS + SAVE OPTIMAL PARAMS
# ════════════════════════════════════════════════════════════════════════════

def run_sweeps(skip_existing: bool):
    print(f"\n{'═'*60}")
    print(f"  STEP 2 - Parameter sweep (goodwill_multiplier={GOODWILL})")
    print(f"  N_DAYS: {N_DAYS}")
    print(f"{'═'*60}")
    t0 = time.time()

    r_reactive  = sweep_reactive(skip_existing)
    r_nightly   = sweep_nightly(skip_existing)
    r_proactive = sweep_proactive(skip_existing)

    optimal_params = {
        "reactive":  {k: find_optimal(r_reactive)[k]
                      for k in ["alpha", "check_interval_minutes"]},
        "nightly":   {"trigger_hour":           find_optimal(r_nightly)["trigger_hour"],
                      "check_interval_minutes": NIGHTLY_CHECK_INTERVAL},
        "proactive": {k: find_optimal(r_proactive)[k]
                      for k in ["planning_period_minutes",
                                "lookahead_period_minutes",
                                "check_interval_minutes"]},
    }
    _save(optimal_params, "optimal_params")

    elapsed = int(time.time() - t0)
    print(f"\n  Sweeps complete in {elapsed//60}m {elapsed%60}s")
    print(f"  Optimal params saved → {DATA_DIR / 'optimal_params.json'}")
    return r_reactive, r_nightly, r_proactive


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(reactive, nightly, proactive):
    valid_pro = [r for r in proactive if not PROACTIVE_EXCLUDE(r)]

    print(f"\n{'═'*70}")
    print(f"  STEP 2 SUMMARY - Cash vs Goodwill Parameter Optimisation")
    print(f"{'═'*70}")

    for name, data in [("Reactive", reactive), ("Nightly", nightly),
                        ("Proactive", valid_pro)]:
        cash_vals = [r["cash_profit_per_day_eur"]  for r in data]
        gw_vals   = [r["gross_profit_per_day_eur"] for r in data]
        opt_gw    = find_optimal(data, "gross_profit_per_day_eur")
        opt_cash  = find_optimal(data, "cash_profit_per_day_eur")

        print(f"\n  {name}")
        print(f"    Cash range:     €{sensitivity_range(cash_vals):>6,.0f}/day  "
              f"(€{min(cash_vals):,.0f} – €{max(cash_vals):,.0f})")
        print(f"    Goodwill range: €{sensitivity_range(gw_vals):>6,.0f}/day  "
              f"(€{min(gw_vals):,.0f} – €{max(gw_vals):,.0f})")
        print(f"    Goodwill / Cash signal ratio: "
              f"{sensitivity_range(gw_vals)/max(sensitivity_range(cash_vals),1):.1f}×")

        if name == "Reactive":
            print(f"    Cash-optimal:     alpha={opt_cash['alpha']}  "
                  f"ci={opt_cash['check_interval_minutes']}")
            print(f"    Goodwill-optimal: alpha={opt_gw['alpha']}  "
                  f"ci={opt_gw['check_interval_minutes']}")
            same = (opt_cash["alpha"] == opt_gw["alpha"] and
                    opt_cash["check_interval_minutes"] == opt_gw["check_interval_minutes"])
            print(f"    Same optimum under both objectives: {same}")
        elif name == "Nightly":
            print(f"    Cash-optimal:     trigger_hour={opt_cash['trigger_hour']:02d}:00")
            print(f"    Goodwill-optimal: trigger_hour={opt_gw['trigger_hour']:02d}:00")
        elif name == "Proactive":
            print(f"    Cash-optimal:     T={opt_cash['planning_period_minutes']} "
                  f"L={opt_cash['lookahead_period_minutes']} "
                  f"ci={opt_cash['check_interval_minutes']}")
            print(f"    Goodwill-optimal: T={opt_gw['planning_period_minutes']} "
                  f"L={opt_gw['lookahead_period_minutes']} "
                  f"ci={opt_gw['check_interval_minutes']}")
    print(f"\n{'═'*70}\n")


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS - REACTIVE
# ════════════════════════════════════════════════════════════════════════════

def plot_reactive_alpha(results: list[dict]):
    """fig_2_reactive_alpha - profit vs alpha, cash vs goodwill, dual axis."""
    best_ci = find_optimal(results)["check_interval_minutes"]
    subset  = sorted([r for r in results if r["check_interval_minutes"] == best_ci],
                     key=lambda r: r["alpha"])

    alphas       = [r["alpha"]                    for r in subset]
    cash_profits = [r["cash_profit_per_day_eur"]  for r in subset]
    gw_profits   = [r["gross_profit_per_day_eur"] for r in subset]
    unmet        = [r["unmet_rate"] * 100          for r in subset]
    opt_gw       = find_optimal(subset, "gross_profit_per_day_eur")
    opt_cash     = find_optimal(subset, "cash_profit_per_day_eur")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))

    l1, = ax1.plot(alphas, gw_profits,   color=_COLORS["goodwill"], linewidth=2.5,
                   marker="o", markersize=7, label="Profit incl. lost revenue (goodwill=1.0)")
    l2, = ax1.plot(alphas, cash_profits, color=_COLORS["cash"], linewidth=2,
                   marker="s", markersize=6, linestyle="--",
                   label="Cash-based profit (goodwill=0)")
    l3, = ax3.plot(alphas, unmet, color=_COLORS["unmet"], linewidth=1.5,
                   marker="^", markersize=5, linestyle=":", label="Unmet demand rate (%)",
                   alpha=0.8)

    ax1.axvline(opt_gw["alpha"],   color=_COLORS["goodwill"], linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.axvline(opt_cash["alpha"], color=_COLORS["cash"],     linestyle=":", linewidth=1.5, alpha=0.5)
    ax1.annotate(f"Goodwill opt.\nα={opt_gw['alpha']}",
                 xy=(opt_gw["alpha"], opt_gw["gross_profit_per_day_eur"]),
                 xytext=(opt_gw["alpha"]+15, opt_gw["gross_profit_per_day_eur"]-800),
                 fontsize=8, color=_COLORS["goodwill"],
                 arrowprops=dict(arrowstyle="->", color=_COLORS["goodwill"]))

    gw_range   = sensitivity_range(gw_profits)
    cash_range = sensitivity_range(cash_profits)
    ax1.text(0.02, 0.97,
             f"Goodwill sensitivity range: €{gw_range:,.0f}/day\n"
             f"Cash-based sensitivity range: €{cash_range:,.0f}/day",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="lightgray", alpha=0.9))

    ax1.set_xlabel("Alpha (relocation aggressiveness)", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax3.set_ylabel("Unmet demand rate (%)", fontsize=11, color=_COLORS["unmet"])
    ax3.tick_params(axis="y", labelcolor=_COLORS["unmet"])
    ax3.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(
        f"Reactive policy - profit vs alpha under two objectives (ci={best_ci} min)\n"
        f"Cash profit is flat; goodwill profit has a clear optimum at α={opt_gw['alpha']}",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_reactive_alpha.png"); plt.show(); plt.close(fig)


def plot_reactive_alpha_v2(results: list[dict]):
    """fig_2_reactive_alpha_v2 - improved version with both optimal verticals annotated."""
    best_ci  = find_optimal(results)["check_interval_minutes"]
    subset   = sorted([r for r in results if r["check_interval_minutes"] == best_ci],
                      key=lambda r: r["alpha"])

    alphas   = [r["alpha"]                    for r in subset]
    gw       = [r["gross_profit_per_day_eur"] for r in subset]
    cash     = [r["cash_profit_per_day_eur"]  for r in subset]
    unmet    = [r["unmet_rate"] * 100          for r in subset]
    opt_gw   = find_optimal(subset, "gross_profit_per_day_eur")
    opt_cash = find_optimal(subset, "cash_profit_per_day_eur")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(alphas, gw,   color=_COLORS["goodwill"], linewidth=2.5,
                   marker="o", markersize=6, label="Profit incl. lost revenue (goodwill=1.0)", zorder=4)
    l2, = ax1.plot(alphas, cash, color=_COLORS["cash"], linewidth=2,
                   marker="s", markersize=5, linestyle="--", label="Cash-based profit (goodwill=0)", zorder=3)
    l3, = ax2.plot(alphas, unmet, color=_COLORS["unmet"], linewidth=1.5,
                   marker="^", markersize=4, linestyle=":", label="Unmet demand rate (%)", alpha=0.8)

    ax1.axvline(opt_gw["alpha"],   color=_COLORS["goodwill"], linestyle=":", linewidth=2,   alpha=0.8, zorder=2)
    ax1.axvline(opt_cash["alpha"], color=_COLORS["cash"],     linestyle=":", linewidth=1.5, alpha=0.6, zorder=2)

    ax1.annotate(
        f"Goodwill opt: α={opt_gw['alpha']}\n€{opt_gw['gross_profit_per_day_eur']:,.0f}/day",
        xy=(opt_gw["alpha"], opt_gw["gross_profit_per_day_eur"]),
        xytext=(opt_gw["alpha"] + 10, min(gw) + (max(gw)-min(gw)) * 0.6),
        fontsize=8.5, color=_COLORS["goodwill"],
        arrowprops=dict(arrowstyle="->", color=_COLORS["goodwill"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=_COLORS["goodwill"], alpha=0.8)
    )
    ax1.annotate(
        f"Cash opt: α={opt_cash['alpha']}\n€{opt_cash['cash_profit_per_day_eur']:,.0f}/day",
        xy=(opt_cash["alpha"], opt_cash["cash_profit_per_day_eur"]),
        xytext=(opt_cash["alpha"] + 15, max(cash) - (max(cash)-min(cash)) * 0.3),
        fontsize=8.5, color=_COLORS["cash"],
        arrowprops=dict(arrowstyle="->", color=_COLORS["cash"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=_COLORS["cash"], alpha=0.8)
    )

    gw_range   = sensitivity_range(gw)
    cash_range = sensitivity_range(cash)
    ax1.text(0.02, 0.97,
             f"Goodwill sensitivity range: €{gw_range:,.0f}/day\n"
             f"Cash-based sensitivity range: €{cash_range:,.0f}/day",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="lightgray", alpha=0.9))

    ax1.set_xlabel("Alpha (relocation aggressiveness)", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=11, color=_COLORS["unmet"])
    ax2.tick_params(axis="y", labelcolor=_COLORS["unmet"])
    ax2.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(
        f"Reactive policy - profit vs alpha under two objectives (check_interval={best_ci} min)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_reactive_alpha_v2.png"); plt.show(); plt.close(fig)


def plot_reactive_ci(results: list[dict]):
    """fig_2_reactive_ci - profit vs check interval at goodwill-optimal alpha."""
    opt_alpha = find_optimal(results)["alpha"]
    subset    = sorted([r for r in results if r["alpha"] == opt_alpha],
                       key=lambda r: r["check_interval_minutes"])
    if len(subset) < 2:
        print("  Skipping fig_2_reactive_ci: not enough ci values at optimal alpha")
        return

    cis          = [r["check_interval_minutes"]   for r in subset]
    cash_profits = [r["cash_profit_per_day_eur"]  for r in subset]
    gw_profits   = [r["gross_profit_per_day_eur"] for r in subset]
    unmet        = [r["unmet_rate"] * 100          for r in subset]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(cis))
    w = 4
    bars1 = ax1.bar(x - w/2, gw_profits,   width=w, color=_COLORS["goodwill"], alpha=0.85, label="Goodwill profit")
    bars2 = ax1.bar(x + w/2, cash_profits, width=w, color=_COLORS["cash"],     alpha=0.75, label="Cash profit")
    ax2.plot(x, unmet, color=_COLORS["unmet"], linewidth=2,
             marker="o", markersize=7, label="Unmet rate (%)", zorder=5)

    for bar, val in zip(bars1, gw_profits):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 50,
                 f"€{val:,.0f}", ha="center", va="bottom", fontsize=7.5, color=_COLORS["goodwill"])
    for bar, val in zip(bars2, cash_profits):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 50,
                 f"€{val:,.0f}", ha="center", va="bottom", fontsize=7.5, color=_COLORS["cash"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{ci} min" for ci in cis], fontsize=11)
    ax1.set_xlabel("Check interval (minutes)", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=11, color=_COLORS["unmet"])
    ax2.tick_params(axis="y", labelcolor=_COLORS["unmet"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10)
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.set_title(
        f"Reactive policy - profit vs check interval at optimal α={opt_alpha}\n",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_reactive_ci.png"); plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS - NIGHTLY
# ════════════════════════════════════════════════════════════════════════════

def plot_nightly_trigger(results: list[dict]):
    """fig_2_nightly_trigger - profit vs trigger hour, cash vs goodwill."""
    results      = sorted(results, key=lambda r: r["trigger_hour"])
    hours        = [r["trigger_hour"]             for r in results]
    cash_profits = [r["cash_profit_per_day_eur"]  for r in results]
    gw_profits   = [r["gross_profit_per_day_eur"] for r in results]
    unmet        = [r["unmet_rate"] * 100          for r in results]
    opt_gw       = find_optimal(results, "gross_profit_per_day_eur")
    opt_cash     = find_optimal(results, "cash_profit_per_day_eur")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(hours))
    w = 0.3
    ax1.bar(x - w/2, gw_profits,   width=w, color=_COLORS["goodwill"], alpha=0.85, label="Goodwill profit")
    ax1.bar(x + w/2, cash_profits, width=w, color=_COLORS["cash"],     alpha=0.75, label="Cash profit")
    ax2.plot(x, unmet, color=_COLORS["unmet"], linewidth=2,
             marker="o", markersize=7, label="Unmet rate (%)", zorder=5)

    for i, h in enumerate(hours):
        window = max(0, 6 - h)
        ax1.text(i, min(gw_profits) - 500, f"{window}h window",
                 ha="center", va="top", fontsize=7.5, color="gray")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=11)
    ax1.set_xlabel("Trigger hour", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=11, color=_COLORS["unmet"])
    ax2.tick_params(axis="y", labelcolor=_COLORS["unmet"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    gw_range   = sensitivity_range(gw_profits)
    cash_range = sensitivity_range(cash_profits)
    ax1.text(0.98, 0.97,
             f"Goodwill range: €{gw_range:,.0f}/day\nCash range: €{cash_range:,.0f}/day",
             transform=ax1.transAxes, fontsize=9, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="lightgray", alpha=0.9))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=10)
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.set_title(
        f"Nightly policy - profit vs trigger hour under two objectives\n"
        f"(grey = worker window; goodwill opt = {opt_gw['trigger_hour']:02d}:00, "
        f"cash opt = {opt_cash['trigger_hour']:02d}:00)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_nightly_trigger.png"); plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS - PROACTIVE
# ════════════════════════════════════════════════════════════════════════════

def _heatmap(results, metric, ci_filter, title, ax, vmin=None, vmax=None,
             cmap="RdYlGn", annotate=True):
    """Draw a planning×lookahead heatmap on ax for a given metric and ci."""
    subset = [r for r in results
              if r["check_interval_minutes"] == ci_filter and not PROACTIVE_EXCLUDE(r)]

    planning_vals  = sorted(set(r["planning_period_minutes"]  for r in subset))
    lookahead_vals = sorted(set(r["lookahead_period_minutes"] for r in subset))
    lookup = {(r["planning_period_minutes"], r["lookahead_period_minutes"]): r[metric]
              for r in subset}
    matrix = np.array([[lookup.get((p, l), np.nan) for l in lookahead_vals]
                       for p in planning_vals])

    vmin = vmin if vmin is not None else np.nanmin(matrix)
    vmax = vmax if vmax is not None else np.nanmax(matrix)
    im   = ax.imshow(matrix, aspect="auto", cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(lookahead_vals)))
    ax.set_xticklabels([f"{l}min" for l in lookahead_vals], fontsize=9)
    ax.set_yticks(range(len(planning_vals)))
    ax.set_yticklabels([f"{p}min" for p in planning_vals], fontsize=9)
    ax.set_xlabel("Lookahead period L (min)", fontsize=10)
    ax.set_ylabel("Planning period T (min)", fontsize=10)
    ax.set_title(title, fontsize=11)

    valid = ~np.isnan(matrix)
    if valid.any():
        best_idx = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
            linewidth=2.5, edgecolor="navy", facecolor="none"
        ))

    if annotate:
        for i in range(len(planning_vals)):
            for j in range(len(lookahead_vals)):
                val = matrix[i, j]
                if np.isnan(val): continue
                text = f"€{val:,.0f}" if "eur" in metric else f"{val:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color="black")
    return im


def plot_proactive_heatmaps(results: list[dict]):
    """fig_2_proactive_heatmaps - T×L heatmaps for cash, goodwill, and difference."""
    valid   = [r for r in results if not PROACTIVE_EXCLUDE(r)]
    best_ci = find_optimal(valid, "gross_profit_per_day_eur")["check_interval_minutes"]
    subset  = [r for r in valid if r["check_interval_minutes"] == best_ci]

    all_cash = [r["cash_profit_per_day_eur"]  for r in subset]
    all_gw   = [r["gross_profit_per_day_eur"] for r in subset]
    vmin = min(min(all_cash), min(all_gw))
    vmax = max(max(all_cash), max(all_gw))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = _heatmap(results, "cash_profit_per_day_eur", best_ci,
                   f"Cash-based profit (goodwill=0)\nci={best_ci}min",
                   axes[0], vmin=vmin, vmax=vmax)
    im2 = _heatmap(results, "gross_profit_per_day_eur", best_ci,
                   f"Profit incl. lost revenue (goodwill=1.0)\nci={best_ci}min",
                   axes[1], vmin=vmin, vmax=vmax)

    # difference panel
    planning_vals  = sorted(set(r["planning_period_minutes"]  for r in subset))
    lookahead_vals = sorted(set(r["lookahead_period_minutes"] for r in subset))
    lookup_cash = {(r["planning_period_minutes"], r["lookahead_period_minutes"]): r["cash_profit_per_day_eur"] for r in subset}
    lookup_gw   = {(r["planning_period_minutes"], r["lookahead_period_minutes"]): r["gross_profit_per_day_eur"] for r in subset}
    diff_matrix = np.array([
        [lookup_gw.get((p,l), np.nan) - lookup_cash.get((p,l), np.nan) for l in lookahead_vals]
        for p in planning_vals
    ])
    im3 = axes[2].imshow(diff_matrix, aspect="auto", cmap="PuOr", origin="lower",
                          vmin=-np.nanmax(np.abs(diff_matrix)),
                          vmax= np.nanmax(np.abs(diff_matrix)))
    axes[2].set_xticks(range(len(lookahead_vals)))
    axes[2].set_xticklabels([f"{l}min" for l in lookahead_vals], fontsize=9)
    axes[2].set_yticks(range(len(planning_vals)))
    axes[2].set_yticklabels([f"{p}min" for p in planning_vals], fontsize=9)
    axes[2].set_xlabel("Lookahead period L (min)", fontsize=10)
    axes[2].set_ylabel("Planning period T (min)", fontsize=10)
    axes[2].set_title("Difference: goodwill − cash\n(purple = goodwill preferred)", fontsize=11)
    for i in range(len(planning_vals)):
        for j in range(len(lookahead_vals)):
            val = diff_matrix[i, j]
            if not np.isnan(val):
                axes[2].text(j, i, f"€{val:,.0f}", ha="center", va="center",
                              fontsize=6.5, color="black")

    fig.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.8).set_label("€/day", fontsize=9)
    fig.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.8).set_label("€/day", fontsize=9)
    fig.colorbar(im3, ax=axes[2], pad=0.02, shrink=0.8).set_label("€/day diff", fontsize=9)

    fig.suptitle(
        "Proactive policy - parameter sensitivity under cash vs goodwill objective\n"
        "(navy outline = optimum per panel; T=ci configurations excluded as degenerate)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_proactive_heatmaps.png"); plt.show(); plt.close(fig)


def plot_proactive_bubble(results: list[dict]):
    """fig_2_pro_bubble - bubble chart T×L, size=profit above baseline, colour=ci."""
    valid   = [r for r in results if not PROACTIVE_EXCLUDE(r)]
    T_vals  = sorted(set(r["planning_period_minutes"]  for r in valid))
    L_vals  = sorted(set(r["lookahead_period_minutes"] for r in valid))
    ci_vals = sorted(set(r["check_interval_minutes"]   for r in valid))

    ci_colors = {30: "#1f77b4", 60: "#2ca02c", 120: "#d62728"}
    ci_jitter = {30: -8, 60: 0, 120: 8}
    baseline_gw = BASELINE_GW["proactive"]

    profits = [r["gross_profit_per_day_eur"] for r in valid]
    p_max   = max(profits)

    fig, ax = plt.subplots(figsize=(11, 7))
    for r in valid:
        T   = r["planning_period_minutes"]
        L   = r["lookahead_period_minutes"]
        ci  = r["check_interval_minutes"]
        p   = r["gross_profit_per_day_eur"]
        u   = r["unmet_rate"] * 100
        is_best = p == max(profits)
        above = max(0, p - baseline_gw)
        size  = 50 + (above / max(p_max - baseline_gw, 1)) * 750
        x_jit = T + ci_jitter.get(ci, 0)

        ax.scatter(x_jit, L, s=size, color=ci_colors.get(ci, "gray"),
                   alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
        if size > 150:
            ax.text(x_jit, L, f"€{p:,.0f}\n({u:.1f}%)",
                    ha="center", va="center", fontsize=6.5,
                    color="yellow" if is_best else "white",
                    fontweight="bold" if is_best else "normal")
        if is_best:
            ax.scatter(x_jit, L, s=size * 1.2, facecolors="none",
                       edgecolors="navy", linewidths=2.5, zorder=4)

    for ci, color in ci_colors.items():
        ax.scatter([], [], s=150, color=color, alpha=0.85, label=f"ci = {ci} min")
    ax.scatter([], [], s=400, color="gray", alpha=0.5, label="Bubble size ∝ profit above baseline")
    ax.axhline(-999, color="gray", linestyle="--", linewidth=1, alpha=0,
               label=f"Baseline: €{baseline_gw:,}/day")

    ax.set_xlabel("Planning period T (minutes)", fontsize=12)
    ax.set_ylabel("Lookahead period L (minutes)", fontsize=12)
    ax.set_xticks(T_vals)
    ax.set_xticklabels([f"T={t}" for t in T_vals], fontsize=10)
    ax.set_yticks(L_vals)
    ax.set_yticklabels([f"L={l}" for l in L_vals], fontsize=10)
    ax.set_xlim(min(T_vals) - 30, max(T_vals) + 30)
    ax.set_ylim(min(L_vals) - 30, max(L_vals) + 30)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "Proactive policy - all parameter combinations (goodwill=1.0)\n"
        "Bubble size ∝ profit above baseline. Navy outline = global optimum. Colour = check interval.",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_pro_bubble.png"); plt.show(); plt.close(fig)


def plot_proactive_parallel(results: list[dict], metric: str, label: str, filename: str):
    """Parallel coordinates across all proactive parameters, coloured by metric."""
    valid = [r for r in results if not PROACTIVE_EXCLUDE(r)]

    profits = [r[metric] for r in valid]
    p_min, p_max = min(profits), max(profits)
    p_q75 = np.percentile(profits, 75)

    T_vals  = sorted(set(r["planning_period_minutes"]  for r in valid))
    L_vals  = sorted(set(r["lookahead_period_minutes"] for r in valid))
    ci_vals = sorted(set(r["check_interval_minutes"]   for r in valid))

    def norm(val, vals):
        idx = sorted(vals).index(val)
        return idx / max(len(vals) - 1, 1)

    def norm_range(val, lo, hi):
        return (val - lo) / (hi - lo) if hi > lo else 0.5

    axes_labels = ["T (min)", "L (min)", "Δt (min)", "Profit/day (€)", "Unmet (%)"]
    xs = list(range(len(axes_labels)))

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap     = cm.RdYlGn
    norm_map = mcolors.Normalize(vmin=p_min, vmax=p_max)

    for r in sorted(valid, key=lambda x: x[metric]):
        T   = r["planning_period_minutes"]
        L   = r["lookahead_period_minutes"]
        ci  = r["check_interval_minutes"]
        p   = r[metric]
        u   = r["unmet_rate"] * 100
        ys  = [
            norm(T,  T_vals),
            norm(L,  L_vals),
            norm(ci, ci_vals),
            norm_range(p, p_min, p_max),
            1 - norm_range(u, 2, 6),
        ]
        color  = cmap(norm_map(p))
        lw     = 2.5 if p >= p_q75 else 0.8
        alpha  = 0.9 if p >= p_q75 else 0.35
        zorder = 3   if p >= p_q75 else 1
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

    for i, lbl in enumerate(axes_labels):
        ax.axvline(i, color="gray", linewidth=1, alpha=0.5)
        ax.text(i, -0.06, lbl, ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())

    tick_labels = [
        [str(v) for v in T_vals],
        [str(v) for v in L_vals],
        [str(v) for v in ci_vals],
        [f"€{p_min:,.0f}", f"€{(p_min+p_max)/2:,.0f}", f"€{p_max:,.0f}"],
        ["6%", "4%", "2%"],
    ]
    tick_positions = [
        [norm(v, T_vals)  for v in T_vals],
        [norm(v, L_vals)  for v in L_vals],
        [norm(v, ci_vals) for v in ci_vals],
        [0, 0.5, 1],
        [0, 0.5, 1],
    ]
    for i, (positions, labels) in enumerate(zip(tick_positions, tick_labels)):
        for pos, lbl in zip(positions, labels):
            ax.text(i + 0.02, pos, lbl, ha="left", va="center", fontsize=7.5)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_map)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, shrink=0.7)
    cbar.formatter = mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
    cbar.update_ticks()

    ax.set_xlim(-0.2, len(axes_labels) - 0.8)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"Proactive policy - parallel coordinates ({label})\n"
        "Bold lines = top quartile. T=Δt degenerate combinations excluded.",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, filename); plt.show(); plt.close(fig)


def plot_proactive_T_ci_by_L(results: list[dict]):
    """fig_2_pro_T_ci_by_L - grouped bars: T×ci interaction, one subplot per L."""
    valid   = [r for r in results if not PROACTIVE_EXCLUDE(r)]
    T_vals  = sorted(set(r["planning_period_minutes"]  for r in valid))
    L_vals  = sorted(set(r["lookahead_period_minutes"] for r in valid))
    ci_vals = sorted(set(r["check_interval_minutes"]   for r in valid))
    ci_colors = {30: "#1f77b4", 60: "#2ca02c", 120: "#d62728"}
    baseline_gw = BASELINE_GW["proactive"]

    ncols = 2
    nrows = (len(L_vals) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows), sharey=True)
    axes = [ax for row in axes for ax in row]

    for idx, L in enumerate(L_vals):
        ax = axes[idx]
        lookup = {(r["planning_period_minutes"], r["check_interval_minutes"]): r["gross_profit_per_day_eur"]
                  for r in valid if r["lookahead_period_minutes"] == L}

        x = np.arange(len(T_vals))
        w = 0.25
        for i, ci in enumerate(ci_vals):
            profits = [lookup.get((T, ci), np.nan) for T in T_vals]
            offset  = (i - len(ci_vals) / 2 + 0.5) * w
            bars    = ax.bar(x + offset, profits, w,
                             color=ci_colors.get(ci, "gray"), alpha=0.85, label=f"Δt={ci} min")
            for bar, p in zip(bars, profits):
                if not np.isnan(p):
                    ax.text(bar.get_x() + bar.get_width()/2, p + 30,
                            f"{p:,.0f}", ha="center", va="bottom",
                            fontsize=6, rotation=90, color=ci_colors.get(ci, "gray"))

        ax.axhline(baseline_gw, color="gray", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"Baseline €{baseline_gw:,}/day")
        ax.set_title(f"L = {L} min", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"T={T}" for T in T_vals], fontsize=9)
        ax.set_ylabel("Goodwill profit/day (€)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
        ax.grid(True, alpha=0.2, axis="y")
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Proactive policy - T × Δt interaction by lookahead period L\n"
        "(dashed line = goodwill-adjusted baseline; T=Δt combinations excluded)",
        fontsize=13
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_pro_T_ci_by_L.png"); plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS - CROSS-POLICY SUMMARIES
# ════════════════════════════════════════════════════════════════════════════

def plot_sensitivity_summary(reactive, nightly, proactive):
    """fig_2_sensitivity_summary - sensitivity range comparison, all policies."""
    valid_pro = [r for r in proactive if not PROACTIVE_EXCLUDE(r)]

    policies    = ["Reactive", "Nightly", "Proactive"]
    datasets    = [reactive, nightly, valid_pro]
    cash_ranges = [sensitivity_range([r["cash_profit_per_day_eur"]  for r in d]) for d in datasets]
    gw_ranges   = [sensitivity_range([r["gross_profit_per_day_eur"] for r in d]) for d in datasets]

    x = np.arange(len(policies))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, gw_ranges,   width=w, color=_COLORS["goodwill"], alpha=0.85,
                label="Goodwill profit (goodwill=1.0)")
    b2 = ax.bar(x + w/2, cash_ranges, width=w, color=_COLORS["cash"],     alpha=0.75,
                label="Cash-based profit (goodwill=0)")

    for bar, val in zip(b1, gw_ranges):
        ax.text(bar.get_x() + bar.get_width()/2, val + 30, f"€{val:,.0f}",
                ha="center", va="bottom", fontsize=9, color=_COLORS["goodwill"], fontweight="bold")
    for bar, val in zip(b2, cash_ranges):
        ax.text(bar.get_x() + bar.get_width()/2, val + 30, f"€{val:,.0f}",
                ha="center", va="bottom", fontsize=9, color=_COLORS["cash"])

    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=12)
    ax.set_ylabel("Parameter sensitivity range (€/day)\n"
                  "(max − min profit across all parameter combinations)", fontsize=11)
    ax.set_title(
        "Parameter sensitivity range under cash vs goodwill objective\n"
        "Larger range = more useful signal for parameter optimisation",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(fig, "fig_2_sensitivity_summary.png"); plt.show(); plt.close(fig)


def plot_beats_baseline():
    """
    fig_2_beats_baseline - does relocation beat the baseline under goodwill=1.0?
    Two panels: absolute profits (left) and profit difference vs baseline (right).
    Values are from the parameter sweep results at each policy's cash-optimal fleet.
    """
    policies = ["Reactive", "Proactive", "Nightly"]
    colors   = [_COLORS["Reactive"], _COLORS["Proactive"], _COLORS["Nightly"]]

    policy_gw   = {"Reactive": 197252, "Proactive": 195271, "Nightly": 187827}
    baseline_gw = {"Reactive": 195738, "Proactive": 192998, "Nightly": 194235}
    fleet_labels = {"Reactive": "(fleet=725)", "Proactive": "(fleet=575)", "Nightly": "(fleet=600)"}
    diffs = {p: policy_gw[p] - baseline_gw[p] for p in policies}

    fig, (ax_abs, ax_diff) = plt.subplots(1, 2, figsize=(13, 6))

    x = np.arange(len(policies))
    w = 0.35
    bars_p = ax_abs.bar(x - w/2, [policy_gw[p]   for p in policies],
                        width=w, color=colors, alpha=0.85, label="Policy (goodwill=1.0)")
    bars_b = ax_abs.bar(x + w/2, [baseline_gw[p] for p in policies],
                        width=w, color="lightgray", alpha=0.9,
                        edgecolor="gray", linewidth=1, label="Baseline (goodwill=1.0)")

    for bar, p in zip(bars_p, policies):
        ax_abs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"€{policy_gw[p]:,.0f}", ha="center", va="bottom",
                    fontsize=8.5, color=colors[policies.index(p)], fontweight="bold")
    for bar, p in zip(bars_b, policies):
        ax_abs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f"€{baseline_gw[p]:,.0f}\n{fleet_labels[p]}",
                    ha="center", va="bottom", fontsize=7.5, color="gray")

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(policies, fontsize=12)
    ax_abs.set_ylabel("Gross profit per day (€)", fontsize=11)
    ax_abs.set_title("Goodwill profit at optimal parameters\nvs goodwill-adjusted baseline", fontsize=11)
    ax_abs.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax_abs.legend(fontsize=10)
    ax_abs.grid(True, alpha=0.25, axis="y")
    all_vals = list(policy_gw.values()) + list(baseline_gw.values())
    ax_abs.set_ylim(min(all_vals) - 3000, max(all_vals) + 3000)

    diff_vals   = [diffs[p] for p in policies]
    bar_colors  = [_COLORS["goodwill"] if d > 0 else _COLORS["unmet"] for d in diff_vals]
    bars = ax_diff.bar(policies, diff_vals, color=bar_colors, alpha=0.85, width=0.5)
    ax_diff.axhline(0, color="black", linewidth=2, linestyle="--")
    for bar, p, val in zip(bars, policies, diff_vals):
        sign   = "+" if val >= 0 else ""
        va     = "bottom" if val >= 0 else "top"
        offset = 30 if val >= 0 else -30
        ax_diff.text(bar.get_x() + bar.get_width()/2, val + offset,
                     f"{sign}€{val:,.0f}/day",
                     ha="center", va=va, fontsize=11,
                     color=_COLORS["goodwill"] if val > 0 else _COLORS["unmet"],
                     fontweight="bold")

    ax_diff.set_ylabel("Profit difference vs baseline (€/day)", fontsize=11)
    ax_diff.set_title("Does relocation beat the baseline?\n(goodwill=1.0 objective)", fontsize=11)
    ax_diff.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax_diff.grid(True, alpha=0.25, axis="y")
    ax_diff.set_xticklabels(policies, fontsize=12)

    fig.suptitle(
        "Relocation policies vs baseline under goodwill=1.0 objective\n"
        "at each policy's optimal parameter configuration",
        fontsize=13
    )
    plt.tight_layout()
    _savefig(fig, "fig_2_beats_baseline.png"); plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOT ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_plots(reactive, nightly, proactive):
    reactive  = enrich(reactive)
    nightly   = enrich(nightly)
    proactive = enrich(proactive)

    print(f"\n{'═'*60}")
    print(f"  Generating Step 2 plots  →  {PLOTS_DIR}")
    print(f"{'═'*60}\n")

    print_summary(reactive, nightly, proactive)

    print("  fig_2_reactive_alpha ...")
    plot_reactive_alpha(reactive)
    print("  fig_2_reactive_alpha_v2 ...")
    plot_reactive_alpha_v2(reactive)
    print("  fig_2_reactive_ci ...")
    plot_reactive_ci(reactive)
    print("  fig_2_nightly_trigger ...")
    plot_nightly_trigger(nightly)
    print("  fig_2_proactive_heatmaps ...")
    plot_proactive_heatmaps(proactive)
    print("  fig_2_pro_bubble ...")
    plot_proactive_bubble(proactive)
    print("  fig_2_pro_parallel_goodwill ...")
    plot_proactive_parallel(proactive, "gross_profit_per_day_eur",
                            "Goodwill profit/day", "fig_2_pro_parallel_goodwill.png")
    print("  fig_2_pro_parallel_cash ...")
    plot_proactive_parallel(proactive, "cash_profit_per_day_eur",
                            "Cash profit/day", "fig_2_pro_parallel_cash.png")
    print("  fig_2_pro_T_ci_by_L ...")
    plot_proactive_T_ci_by_L(proactive)
    print("  fig_2_sensitivity_summary ...")
    plot_sensitivity_summary(reactive, nightly, proactive)
    print("  fig_2_beats_baseline ...")
    plot_beats_baseline()

    print(f"\n  All plots saved to {PLOTS_DIR}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 2: parameter sweep with goodwill objective")
    parser.add_argument("--plot", action="store_true",
                        help="Plot only - skip sweep (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip policies whose JSON already exists")
    args = parser.parse_args()

    _ensure_dirs()

    if args.plot:
        print("Loading existing parameter sweep data ...")
        reactive  = _load("param_reactive")
        nightly   = _load("param_nightly")
        proactive = _load("param_proactive")
        print(f"  Reactive:  {len(reactive)} combinations")
        print(f"  Nightly:   {len(nightly)} combinations")
        print(f"  Proactive: {len(proactive)} combinations")
    else:
        reactive, nightly, proactive = run_sweeps(args.skip_existing)

    run_plots(reactive, nightly, proactive)


if __name__ == "__main__":
    main()
