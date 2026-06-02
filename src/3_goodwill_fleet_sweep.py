"""
3_goodwill_fleet_sweep.py  -  Goodwill-based fleet sweep (Step 3).

Repeats the Step 1 fleet sweep (300-800 vehicles; step 25) using:
  - goodwill_multiplier = 1.0  (lost revenue included as a penalty)
  - goodwill-optimal parameters from Step 2 (loaded from final_param.OPTIMAL)
  - default worker counts (same as Step 1)

Produces six figures:
  fig_3_profit_goodwill         Gross profit per day vs fleet size — all policies (PRIMARY)
  fig_3_profit_difference       Profit advantage of relocation policies over baseline
  fig_3_objective_comparison    Side-by-side: cash objective (Step 1) vs goodwill (Step 3)
  fig_3_unmet_comparison        Unmet rate vs fleet size with cash and goodwill optima marked
  fig_3_relocation_cost         Relocation cost vs fleet size (relocation policies only)
  fig_3_unmet_vs_fleet          Unmet demand rate vs fleet size — all policies

Usage
-----
  python 3_goodwill_fleet_sweep.py                 # run sweep + all plots
  python 3_goodwill_fleet_sweep.py --plot          # plot only (data must exist)
  python 3_goodwill_fleet_sweep.py --skip-existing # skip policies whose JSON exists
"""

import argparse
import json
import time
from pathlib import Path

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

# Goodwill-optimal parameters — held fixed from Step 2 onwards
from final_param import OPTIMAL


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

FLEET_MIN    = 300
FLEET_MAX    = 800
FLEET_STEP   = 25
N_DAYS       = 30
GOODWILL     = 1.0
TARGET_UNMET = 0.05

WORKER_COUNTS = {
    "baseline":  1,
    "reactive":  300,
    "nightly":   300,
    "proactive": 100,
}

DATA_DIR      = Path("./data/metrics/3_goodwill_fleet")
PLOTS_DIR     = DATA_DIR / "plots"
STEP1_DIR     = Path("./data/metrics/1_cash_fleet")   # for comparison plots

# Policy configs built from OPTIMAL (final_param.py)
# Each entry in OPTIMAL has: fleet, workers, params, policy_name, label, color
POLICIES = {
    entry["label"]: {
        "policy_name": entry["policy_name"],
        **entry["params"],
    }
    for entry in OPTIMAL.values()
}

# Colors and markers consistent with final_param.py OPTIMAL
_COLORS  = {entry["label"]: entry["color"] for entry in OPTIMAL.values()}
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


def _save(data, name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


def _load(name: str):
    with open(DATA_DIR / f"{name}.json") as f:
        return json.load(f)


def _load_step1(name: str):
    path = STEP1_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _exists(name: str) -> bool:
    return (DATA_DIR / f"{name}.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ════════════════════════════════════════════════════════════════════════════

def _run(policy_name, fleet_size, **policy_kwargs):
    """Run one simulation with goodwill=1.0; return result dict."""
    model  = FinancialModel(goodwill_multiplier=GOODWILL)
    policy = load_policy(policy_name=policy_name, **policy_kwargs)
    engine = SimulationEngine(
        n_days=N_DAYS, policy=policy,
        fleet_size_override=fleet_size, verbose=False
    )
    engine.run(verbose=False)
    m   = Metrics(engine.event_log, fleet_size=fleet_size)
    fin = FinancialResults(engine.event_log, fleet_size=fleet_size, model=model)
    rel = m.relocation_stats()
    wor = m.worker_stats()
    return {
        "fleet_size":               fleet_size,
        "policy":                   policy_name,
        "goodwill_multiplier":      GOODWILL,
        **policy_kwargs,
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "total_unmet":              m.total_unmet,
        "relocations":              m.total_relocations,
        "relocation_min":           rel["total_min"],
        "relocation_km":            rel["total_km"],
        "relocations_queued":       wor["relocations_queued"],
        "relocations_forgotten":    wor["relocations_forgotten"],
        "peak_workers":             wor["peak_workers_active"],
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


def find_optimal(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


def find_knee(results: list[dict], target: float = TARGET_UNMET) -> dict | None:
    for r in sorted(results, key=lambda x: x["fleet_size"]):
        if r["unmet_rate"] <= target:
            return r
    return None


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP
# ════════════════════════════════════════════════════════════════════════════

def run_sweep(label: str, policy_kwargs: dict) -> list[dict]:
    """Run fleet sweep for one policy under goodwill=1.0."""
    policy_name = policy_kwargs["policy_name"]
    n_workers   = WORKER_COUNTS.get(policy_name, 30)
    sizes       = range(FLEET_MIN, FLEET_MAX + 1, FLEET_STEP)

    print(f"\n  {label}  ({len(list(sizes))} fleet sizes, {n_workers} workers, goodwill={GOODWILL})")

    results = []
    pkwargs = {k: v for k, v in policy_kwargs.items() if k != "policy_name"}
    with ws._patch_workers(n_workers):
        for fleet_size in sizes:
            r    = _run(policy_name, fleet_size, **pkwargs)
            sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    fleet={fleet_size:>3}  "
                  f"unmet={r['unmet_rate']:>5.1%}  "
                  f"served={r['total_served']:>5}  "
                  f"profit/day={sign}{r['gross_profit_per_day_eur']:>8.0f}€  "
                  f"reloc={r['relocations']:>4}")
            results.append(r)
    return results


def run_all_sweeps(skip_existing: bool) -> dict:
    _ensure_dirs()

    print(f"\n{'═'*65}")
    print(f"  STEP 3 — Goodwill fleet sweep (goodwill_multiplier={GOODWILL})")
    print(f"  Fleet: {FLEET_MIN}–{FLEET_MAX}, step {FLEET_STEP}  |  N_DAYS: {N_DAYS}")
    print(f"  Parameters: from final_param.OPTIMAL")
    print(f"{'═'*65}")
    t0 = time.time()

    all_results = {}
    for label, kwargs in POLICIES.items():
        name = f"fleet_sweep_{label.lower()}"
        if skip_existing and _exists(name):
            print(f"  {label}: skipping (data exists)")
            all_results[label] = _load(name)
            continue
        results = run_sweep(label, kwargs)
        _save(results, name)
        all_results[label] = results

    elapsed = int(time.time() - t0)
    print(f"\n  Sweep complete in {elapsed//60}m {elapsed%60}s")

    print(f"\n{'═'*65}")
    print(f"  PROFIT-OPTIMAL FLEET (goodwill={GOODWILL})")
    print(f"{'═'*65}")
    for label, results in all_results.items():
        opt  = find_optimal(results)
        knee = find_knee(results)
        print(f"  {label:<12} opt={opt['fleet_size']}  "
              f"profit/day=€{opt['gross_profit_per_day_eur']:,.0f}  "
              f"unmet={opt['unmet_rate']:.1%}  "
              f"5%_knee={knee['fleet_size'] if knee else 'n/a'}")
    print(f"{'═'*65}")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_profit_goodwill(all_results: dict):
    """
    fig_3_profit_goodwill — PRIMARY plot.
    Gross profit per day vs fleet size under goodwill=1.0, all policies.
    Annotated with profit-optimal fleet per policy.
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    all_profits = [r["gross_profit_per_day_eur"]
                   for results in all_results.values() for r in results]
    y_min = min(all_profits)
    y_max = max(all_profits)
    y_pad = (y_max - y_min) * 0.08

    annotation_offsets = {
        "Baseline":  (-120, +y_pad * 1.5),
        "Reactive":  (+30,  +y_pad * 2.5),
        "Nightly":   (-120, -y_pad * 2.0),
        "Proactive": (+30,  +y_pad * 1.0),
    }

    for label, results in all_results.items():
        sizes   = [r["fleet_size"]               for r in results]
        profits = [r["gross_profit_per_day_eur"]  for r in results]
        opt     = find_optimal(results)

        ax.plot(sizes, profits, label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5, zorder=3)
        ax.scatter([opt["fleet_size"]], [opt["gross_profit_per_day_eur"]],
                   color=_color(label), s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)

        dx, dy = annotation_offsets.get(label, (30, y_pad))
        ax.annotate(
            f"{label}\n{opt['fleet_size']} veh  €{opt['gross_profit_per_day_eur']:,.0f}/day",
            xy     = (opt["fleet_size"], opt["gross_profit_per_day_eur"]),
            xytext = (opt["fleet_size"] + dx, opt["gross_profit_per_day_eur"] + dy),
            fontsize=8.5, color=_color(label), fontweight="bold",
            ha="left" if dx > 0 else "right", va="center",
            arrowprops=dict(arrowstyle="-", color=_color(label), lw=0.9,
                            connectionstyle="arc3,rad=0"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=_color(label), alpha=0.85),
            zorder=6,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        f"Fleet size vs daily gross profit — all policies\n"
        f"(goodwill multiplier = {GOODWILL}: lost revenue included)",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(y_min - y_pad * 1.5, y_max + y_pad * 5)
    plt.tight_layout()
    _savefig(fig, "fig_3_profit_goodwill.png"); plt.show(); plt.close(fig)


def plot_profit_difference(all_results: dict):
    """
    fig_3_profit_difference — profit advantage of relocation policies vs baseline
    under goodwill=1.0. Zero line = baseline.
    """
    baseline = all_results.get("Baseline")
    if not baseline:
        print("  No baseline data — skipping profit difference plot")
        return

    base_by_fleet = {r["fleet_size"]: r["gross_profit_per_day_eur"] for r in baseline}

    fig, ax = plt.subplots(figsize=(12, 7))
    for label, results in all_results.items():
        if label == "Baseline":
            continue
        sizes = [r["fleet_size"] for r in results if r["fleet_size"] in base_by_fleet]
        diffs = [r["gross_profit_per_day_eur"] - base_by_fleet[r["fleet_size"]]
                 for r in results if r["fleet_size"] in base_by_fleet]

        ax.plot(sizes, diffs, label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=6)

        opt = find_optimal(results)
        if opt["fleet_size"] in base_by_fleet:
            opt_diff = opt["gross_profit_per_day_eur"] - base_by_fleet[opt["fleet_size"]]
            sign = "+" if opt_diff >= 0 else ""
            ax.annotate(
                f"{opt['fleet_size']} veh\n{sign}€{opt_diff:,.0f}/day",
                xy=(opt["fleet_size"], opt_diff),
                xytext=(opt["fleet_size"] + 20, opt_diff),
                fontsize=8, color=_color(label), va="center",
                arrowprops=dict(arrowstyle="->", color=_color(label), lw=1.2)
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=2, label="Baseline (reference)")
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Profit difference vs baseline (€/day)", fontsize=12)
    ax.set_title(f"Profit advantage of relocation policies over baseline (goodwill={GOODWILL})",
                 fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig(fig, "fig_3_profit_difference.png"); plt.show(); plt.close(fig)


def plot_objective_comparison(step1: dict, step3: dict):
    """
    fig_3_objective_comparison — side-by-side: Step 1 (cash) vs Step 3 (goodwill).
    Shared y-axis shows how the objective shifts profit-optimal fleet sizes.
    """
    all_profits = [
        r["gross_profit_per_day_eur"]
        for data in [step1, step3]
        for results in data.values()
        for r in results
    ]
    y_min = min(all_profits)
    y_max = max(all_profits)
    y_pad = (y_max - y_min) * 0.08

    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, data, title in [
        (ax_a, step1, "Cash-based profit\n(goodwill = 0)"),
        (ax_c, step3, f"Profit incl. lost revenue\n(goodwill = {GOODWILL})"),
    ]:
        for label, results in data.items():
            sizes   = [r["fleet_size"]               for r in results]
            profits = [r["gross_profit_per_day_eur"]  for r in results]
            ax.plot(sizes, profits, label=label, color=_color(label),
                    marker=_marker(label), linewidth=2, markersize=4)
            opt = find_optimal(results)
            ax.axvline(opt["fleet_size"], color=_color(label),
                       linewidth=0.8, linestyle=":", alpha=0.6)

        ax.set_xlabel("Fleet size (vehicles)", fontsize=11)
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min - y_pad, y_max + y_pad * 1.5)

    fig.suptitle("Effect of objective function on profit-optimal fleet size per policy",
                 fontsize=13)
    plt.tight_layout()
    _savefig(fig, "fig_3_objective_comparison.png"); plt.show(); plt.close(fig)


def plot_unmet_comparison(step1: dict, step3: dict):
    """
    fig_3_unmet_comparison — unmet rate vs fleet size.
    Hollow stars = cash-optimal fleet (Step 1); filled stars = goodwill-optimal (Step 3).
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # draw curves from step3 (identical simulation unmet rates)
    for label, results in step3.items():
        sizes = [r["fleet_size"]       for r in results]
        unmet = [r["unmet_rate"] * 100 for r in results]
        ax.plot(sizes, unmet, label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=4, zorder=3)

    # cash-optimal stars (hollow)
    for label, results in step1.items():
        opt = find_optimal(results)
        # find unmet at that fleet size in step3
        step3_results = step3.get(label, [])
        match = next((r for r in step3_results if r["fleet_size"] == opt["fleet_size"]), None)
        if match:
            ax.scatter([opt["fleet_size"]], [match["unmet_rate"] * 100],
                       marker="*", s=220, zorder=6,
                       facecolors="white", edgecolors=_color(label), linewidths=1.8)

    # goodwill-optimal stars (filled)
    for label, results in step3.items():
        opt = find_optimal(results)
        ax.scatter([opt["fleet_size"]], [opt["unmet_rate"] * 100],
                   marker="*", s=220, zorder=6,
                   color=_color(label), edgecolors="black", linewidths=0.8)

    ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"{TARGET_UNMET:.0%} service target")

    star_cash = Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
                       markeredgecolor="gray", markeredgewidth=1.5, markersize=12,
                       label="Cash-optimal fleet (hollow, Step 1)")
    star_gw   = Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                       markeredgecolor="black", markeredgewidth=0.8, markersize=12,
                       label="Goodwill-optimal fleet (filled, Step 3)")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [star_cash, star_gw],
              labels  + [star_cash.get_label(), star_gw.get_label()],
              fontsize=9, loc="upper right")

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.set_title("Unmet demand rate vs fleet size — cash vs goodwill optimal fleet", fontsize=12)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_3_unmet_comparison.png"); plt.show(); plt.close(fig)


def plot_relocation_cost(all_results: dict):
    """
    fig_3_relocation_cost — relocation cost vs fleet size (baseline excluded).
    Dotted verticals mark the goodwill-optimal fleet per policy.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    any_plotted = False

    for label, results in all_results.items():
        sorted_r = sorted(results, key=lambda r: r["fleet_size"])
        sizes    = [r["fleet_size"]          for r in sorted_r]
        rel_cost = [r["relocation_cost_eur"] for r in sorted_r]

        if all(v == 0 for v in rel_cost):
            continue

        ax.plot(sizes, rel_cost, label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5, zorder=3)
        any_plotted = True

        opt      = find_optimal(results)
        opt_cost = opt["relocation_cost_eur"]
        ax.axvline(opt["fleet_size"], color=_color(label),
                   linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)
        ax.annotate(
            f"{opt['fleet_size']} veh\n€{opt_cost:,.0f}",
            xy=(opt["fleet_size"], opt_cost),
            xytext=(opt["fleet_size"] + 12, opt_cost),
            fontsize=8, color=_color(label), va="center",
            arrowprops=dict(arrowstyle="->", color=_color(label), lw=1),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=_color(label), alpha=0.85),
            zorder=5,
        )

    if not any_plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Relocation cost per day (€)", fontsize=12)
    ax.set_title(
        f"Fleet size vs daily relocation cost — relocation policies\n"
        f"(goodwill={GOODWILL}, optimal parameters; "
        f"dotted verticals = goodwill-optimal fleet per policy)",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(fig, "fig_3_relocation_cost.png"); plt.show(); plt.close(fig)


def plot_unmet_vs_fleet(all_results: dict):
    """
    fig_3_unmet_vs_fleet — unmet demand rate vs fleet size, all policies.
    Stars mark the goodwill-optimal fleet size per policy.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, results in all_results.items():
        sizes = [r["fleet_size"]       for r in results]
        unmet = [r["unmet_rate"] * 100 for r in results]
        ax.plot(sizes, unmet, label=label, color=_color(label),
                marker=_marker(label), linewidth=2, markersize=5)

        opt = find_optimal(results)
        ax.scatter([opt["fleet_size"]], [opt["unmet_rate"] * 100],
                   color=_color(label), s=120, zorder=5,
                   marker="*", edgecolors="black", linewidths=0.5)

    ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Target: {TARGET_UNMET:.0%} unmet")
    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Fleet size vs unmet demand rate — goodwill objective\n"
                 "(★ = goodwill-optimal fleet size per policy)",
                 fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _savefig(fig, "fig_3_unmet_vs_fleet.png"); plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOT ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_plots(all_results: dict):
    print(f"\n{'═'*60}")
    print(f"  Generating Step 3 plots  →  {PLOTS_DIR}")
    print(f"{'═'*60}\n")

    # load Step 1 data for comparison plots (optional)
    step1 = {}
    for label in POLICIES:
        data = _load_step1(f"sweep_{label.lower()}")
        if data:
            step1[label] = data
        else:
            print(f"  WARNING: Step 1 data missing for {label} — skipping comparison plots")

    print("  fig_3_profit_goodwill ...")
    plot_profit_goodwill(all_results)

    print("  fig_3_profit_difference ...")
    plot_profit_difference(all_results)

    print("  fig_3_unmet_vs_fleet ...")
    plot_unmet_vs_fleet(all_results)

    print("  fig_3_relocation_cost ...")
    plot_relocation_cost(all_results)

    if step1:
        print("  fig_3_objective_comparison ...")
        plot_objective_comparison(step1, all_results)
        print("  fig_3_unmet_comparison ...")
        plot_unmet_comparison(step1, all_results)
    else:
        print("  Skipping comparison plots (no Step 1 data found)")

    print(f"\n  All plots saved to {PLOTS_DIR}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 3: goodwill-based fleet sweep")
    parser.add_argument("--plot", action="store_true",
                        help="Plot only — skip sweep (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip policies whose JSON file already exists")
    args = parser.parse_args()

    _ensure_dirs()

    if args.plot:
        print("Loading existing goodwill sweep data ...")
        all_results = {}
        for label in POLICIES:
            name = f"fleet_sweep_{label.lower()}"
            if _exists(name):
                all_results[label] = _load(name)
            else:
                print(f"  WARNING: {name}.json not found — run without --plot first")
        if all_results:
            run_plots(all_results)
    else:
        all_results = run_all_sweeps(args.skip_existing)
        run_plots(all_results)


if __name__ == "__main__":
    main()
