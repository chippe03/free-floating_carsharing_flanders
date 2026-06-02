"""
6_full_comparison.py  -  Full policy comparison (Step 6).

All four policies are evaluated at their individually optimal configurations:
  - Fleet size from Step 5 (sequential validation)
  - Parameters from Step 2 (goodwill parameter sweep)
  - Worker count from Step 4

Each policy is run as a fresh simulation (goodwill_multiplier = 1.0) and the
event log is saved. Results are compared under both the goodwill objective
and the derived cash-based profit, producing the primary policy ranking.

Event logs are also used to analyse the spatial and temporal distribution of
service improvements across cities, hours of day, and days of week.

Source files are checked first for matching cached results; a fresh simulation
is run only if no cached result is found (or --rerun is passed).

Cached result search order:
  Baseline:  data/metrics/3_goodwill_fleet/fleet_sweep_baseline.json  (fleet=700)
  Reactive:  data/metrics/5_sequential/validation_reactive.json       (fleet=700, workers=250)
  Nightly:   data/metrics/4_workers/workers_nightly.json              (fleet=775, workers=25)
  Proactive: data/metrics/4_workers/workers_proactive.json            (fleet=700, workers=150)

Produces twelve figures:
  fig_6_comparison          Bar chart: all policies at optimal config (cash + goodwill)

  Per-policy spatial figures (saved to data/metrics/6_comparison/{policy}/):
    fig_city_overall        Overall unmet rate per city
    fig_city_hourly         Hourly unmet rate per city
    fig_city_weekday        Weekday unmet rate per city
    fig_city_heatmap        City × hour + city × weekday heatmap

  Comparison spatial figures (saved to data/metrics/6_comparison/comparison/):
    fig_compare_overall     Overall unmet rate per city — all policies
    fig_compare_hourly      Aggregate hourly unmet — all policies on one plot
    fig_compare_weekday     Aggregate weekday unmet — all policies on one plot
    fig_compare_heatmap_diff  Heatmap difference vs baseline per policy

Usage
-----
  python 6_full_comparison.py            # collect results + run sims + all plots
  python 6_full_comparison.py --plot     # plots only (needs saved data + logs)
  python 6_full_comparison.py --rerun    # force fresh simulations
  python 6_full_comparison.py --skip-existing  # skip policies with saved logs
  python 6_full_comparison.py --policies baseline reactive
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from src.simulation.engine import SimulationEngine
    from src.analysis.metrics import Metrics
    from src.analysis.financials import FinancialResults, FinancialModel
    from src.relocation.create_policy import load_policy
    from src.optimization import worker_pool_size as ws
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False

from final_param import OPTIMAL


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

N_DAYS       = 30
GOODWILL     = 1.0
TARGET_UNMET = 0.05

POLICY_ORDER  = ["baseline", "reactive", "nightly", "proactive"]
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKDAY_SHORT = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

DATA_DIR = Path("./data/metrics/6_comparison")
PLOTS_DIR = DATA_DIR / "plots"

# Colors and labels from final_param.OPTIMAL
_COLORS  = {k: v["color"] for k, v in OPTIMAL.items()}
_LABELS  = {k: v["label"] for k, v in OPTIMAL.items()}
_MARKERS = {"baseline": "o", "reactive": "s", "nightly": "D", "proactive": "^"}

# Cached source files to check before running fresh simulations
SOURCE_FILES = {
    "baseline": [
        ("data/metrics/3_goodwill_fleet/fleet_sweep_baseline.json",
         lambda r: r["fleet_size"] == OPTIMAL["baseline"]["fleet"]),
    ],
    "reactive": [
        ("data/metrics/5_sequential/validation_reactive.json",
         lambda r: r["fleet_size"] == OPTIMAL["reactive"]["fleet"]
                   and r.get("n_workers", 0) == OPTIMAL["reactive"]["workers"]),
        ("data/metrics/4_workers/workers_reactive.json",
         lambda r: r["fleet_size"] == OPTIMAL["reactive"]["fleet"]
                   and r.get("n_workers", 0) == OPTIMAL["reactive"]["workers"]),
    ],
    "nightly": [
        ("data/metrics/4_workers/workers_nightly.json",
         lambda r: r["fleet_size"] == OPTIMAL["nightly"]["fleet"]
                   and r.get("n_workers", 0) == OPTIMAL["nightly"]["workers"]),
    ],
    "proactive": [
        ("data/metrics/4_workers/workers_proactive.json",
         lambda r: r["fleet_size"] == OPTIMAL["proactive"]["fleet"]
                   and r.get("n_workers", 0) == OPTIMAL["proactive"]["workers"]),
    ],
}


# ════════════════════════════════════════════════════════════════════════════
#  IO
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _log_path(policy_key: str) -> Path:
    return DATA_DIR / policy_key / f"events_{policy_key}.json"


def _plot_dir(policy_key: str) -> Path:
    d = DATA_DIR / policy_key / "plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _comp_dir() -> Path:
    d = DATA_DIR / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_results(data: list):
    path = DATA_DIR / "optimal_results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


def _load_results() -> list:
    with open(DATA_DIR / "optimal_results.json") as f:
        return json.load(f)


def _results_exist() -> bool:
    return (DATA_DIR / "optimal_results.json").exists()


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════════════════════
#  DATA COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def find_in_source(policy_key: str) -> dict | None:
    """Search cached files for a matching optimal-config entry."""
    for filepath, predicate in SOURCE_FILES.get(policy_key, []):
        path = Path(filepath)
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
        match = next((r for r in entries if predicate(r)), None)
        if match:
            print(f"  {policy_key}: found cached result in {filepath}")
            return match
    return None


def run_fresh(policy_key: str) -> dict:
    """Run a fresh simulation and save the event log."""
    if not SIM_AVAILABLE:
        raise RuntimeError(
            f"Simulation unavailable and no cached result for {policy_key}. "
            "Run from the project root with simulation dependencies installed."
        )
    cfg = OPTIMAL[policy_key]
    print(f"  {policy_key}: running fresh simulation "
          f"(fleet={cfg['fleet']}, workers={cfg['workers']})")

    model  = FinancialModel(goodwill_multiplier=GOODWILL)
    policy = load_policy(policy_name=cfg["policy_name"], **cfg["params"])
    engine = SimulationEngine(
        n_days=N_DAYS, policy=policy,
        fleet_size_override=cfg["fleet"], verbose=False
    )
    with ws._patch_workers(cfg["workers"]):
        engine.run(verbose=False)

    # save event log for spatial analysis
    log_path = _log_path(policy_key)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(engine.event_log, f)
    print(f"  Event log saved: {log_path}  ({len(engine.event_log):,} events)")

    m   = Metrics(engine.event_log, fleet_size=cfg["fleet"])
    fin = FinancialResults(engine.event_log, fleet_size=cfg["fleet"], model=model)
    rel = m.relocation_stats()

    return {
        "label":                    cfg["label"],
        "policy":                   cfg["policy_name"],
        "fleet_size":               cfg["fleet"],
        "n_workers":                cfg["workers"],
        "goodwill_multiplier":      GOODWILL,
        **cfg["params"],
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "total_unmet":              m.total_unmet,
        "relocations":              m.total_relocations,
        "relocation_min":           rel["total_min"],
        "relocation_km":            rel["total_km"],
        "total_revenue_eur":        round(fin.total_revenue, 2),
        "lost_revenue_eur":         round(fin.lost_revenue, 2),
        "vehicle_cost_eur":         round(fin.total_vehicle_cost, 2),
        "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
        "total_cost_eur":           round(fin.total_cost, 2),
        "gross_profit_eur":         round(fin.gross_profit, 2),
        "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
    }


def collect_results(policies: list[str], force_rerun: bool = False) -> dict:
    results = {}
    for policy_key in policies:
        r = None if force_rerun else find_in_source(policy_key)
        if r is None:
            r = run_fresh(policy_key)
        r["label"] = OPTIMAL[policy_key]["label"]
        results[policy_key] = r
        sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
        print(f"    fleet={r['fleet_size']}  workers={r.get('n_workers','?')}  "
              f"unmet={r['unmet_rate']:.1%}  "
              f"profit/day={sign}€{r['gross_profit_per_day_eur']:>9,.0f}  "
              f"lost_rev/day=€{r['lost_revenue_eur']/(N_DAYS+1):,.0f}")
    return results


# ════════════════════════════════════════════════════════════════════════════
#  FINANCIAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def cash_profit(r: dict) -> float:
    return (r["gross_profit_eur"] + r["lost_revenue_eur"]) / (N_DAYS+1)


def profit_at(r: dict, m: float) -> float:
    return cash_profit(r) - m * r["lost_revenue_eur"] / (N_DAYS+1)


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    base      = results.get("baseline", next(iter(results.values())))
    base_cash = cash_profit(base)
    base_gw   = profit_at(base, 1.0)

    print(f"\n{'═'*90}")
    print(f"  STEP 6 — FULL POLICY COMPARISON (goodwill=1.0)")
    print(f"{'═'*90}")
    print(f"  {'Policy':<12} {'Fleet':>6} {'Workers':>8} "
          f"{'GW profit/day':>14} {'vs Base (GW)':>13} "
          f"{'Cash/day':>12} {'vs Base (cash)':>15} {'Unmet':>7}")
    print(f"  {'-'*88}")

    for policy_key in POLICY_ORDER:
        if policy_key not in results:
            continue
        r  = results[policy_key]
        gw = profit_at(r, 1.0)
        ca = cash_profit(r)
        print(f"  {r['label']:<12} {r['fleet_size']:>6} {r.get('n_workers',1):>8} "
              f"€{gw:>12,.0f}  {gw-base_gw:>+12,.0f}  "
              f"€{ca:>10,.0f}  {ca-base_cash:>+14,.0f}  "
              f"{r['unmet_rate']:>6.1%}")
    print(f"\n{'═'*90}\n")


# ════════════════════════════════════════════════════════════════════════════
#  FINANCIAL COMPARISON PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: dict):
    """
    fig_6_comparison — side-by-side bar chart under goodwill=1.0 (left)
    and cash (right). Primary policy ranking figure.
    """
    order   = [p for p in POLICY_ORDER if p in results]
    labels  = [results[p]["label"] for p in order]
    gw_vals = [profit_at(results[p], 1.0) for p in order]
    ca_vals = [cash_profit(results[p])     for p in order]
    colors  = [_COLORS[p]                  for p in order]
    x       = np.arange(len(labels))

    base_gw = profit_at(results["baseline"], 1.0) if "baseline" in results else 0
    base_ca = cash_profit(results["baseline"])     if "baseline" in results else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    for ax, vals, base, title in [
        (ax1, gw_vals, base_gw, "Goodwill=1.0 (lost revenue included)"),
        (ax2, ca_vals, base_ca, "Cash-based profit (goodwill=0)"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55, zorder=3)
        ax.axhline(base, color=_COLORS["baseline"],
                   linewidth=1.5, linestyle="--", alpha=0.7)

        for bar, lbl, val in zip(bars, labels, vals):
            diff = val - base
            sign = "+" if diff >= 0 else ""
            col  = _COLORS[[k for k,v in OPTIMAL.items() if v["label"]==lbl][0]]
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 50,
                    f"€{val:,.0f}/day",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=col)
            if lbl != "Baseline":
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 550,
                        f"({sign}€{diff:,.0f})",
                        ha="center", va="bottom", fontsize=8, color=col)

        for bar, p in zip(bars, order):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    min(vals) - 500,
                    f"unmet: {results[p]['unmet_rate']:.1%}",
                    ha="center", va="top", fontsize=8, color="gray")

        y_min = min(vals); y_max = max(vals)
        y_pad = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - y_pad * 3, y_max + y_pad * 4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    for ax in [ax1, ax2]:
        for i, p in enumerate(order):
            r = results[p]
            ax.text(i, ax.get_ylim()[0] * 1.002,
                    f"fleet={r['fleet_size']}\nworkers={r.get('n_workers',1)}",
                    ha="center", va="top", fontsize=7, color="gray")

    fig.suptitle(
        "Full policy comparison at individual optimal configurations\n"
        "(each policy at its own goodwill-optimal fleet and recommended worker count)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, PLOTS_DIR / "fig_6_comparison.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  SPATIAL / TEMPORAL AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def aggregate(log: list[dict]):
    by_city_hour    = defaultdict(lambda: defaultdict(lambda: {"demand": 0, "unmet": 0}))
    by_city_weekday = defaultdict(lambda: defaultdict(lambda: {"demand": 0, "unmet": 0}))
    city_totals     = defaultdict(lambda: {"demand": 0, "unmet": 0})

    for e in log:
        if e["event"] not in ("TRIP_ASSIGNED", "TRIP_UNMET"):
            continue
        city     = e.get("origin", "Unknown")
        hour     = int(e.get("hour", 0))
        weekday  = e.get("weekday", "Unknown")
        is_unmet = e["event"] == "TRIP_UNMET"

        by_city_hour[city][hour]["demand"] += 1
        by_city_weekday[city][weekday]["demand"] += 1
        city_totals[city]["demand"] += 1
        if is_unmet:
            by_city_hour[city][hour]["unmet"] += 1
            by_city_weekday[city][weekday]["unmet"] += 1
            city_totals[city]["unmet"] += 1

    return dict(by_city_hour), dict(by_city_weekday), dict(city_totals)


def _unmet_rate(d: dict, key) -> float:
    b = d.get(key, {"demand": 0, "unmet": 0})
    return b["unmet"] / b["demand"] * 100 if b["demand"] > 0 else 0.0


def print_city_summary(policy_key: str, city_totals: dict):
    print(f"\n  {OPTIMAL[policy_key]['label']} — unmet rate per city:")
    for city in sorted(city_totals, key=lambda c:
                       city_totals[c]["unmet"]/max(city_totals[c]["demand"],1),
                       reverse=True):
        t    = city_totals[city]
        rate = t["unmet"] / max(t["demand"], 1) * 100
        print(f"    {city:<15} {rate:>5.1f}%  ({t['unmet']:,} / {t['demand']:,})")
    if "kortrijk" in city_totals:
        ku = city_totals["kortrijk"]["unmet"]
        tu = sum(c["unmet"] for c in city_totals.values())
        if tu > 0:
            print(f"    Kortrijk share of total unmet: {ku/tu:.1%}")


# ════════════════════════════════════════════════════════════════════════════
#  PER-POLICY SPATIAL FIGURES
# ════════════════════════════════════════════════════════════════════════════

def plot_city_overall(policy_key: str, city_totals: dict):
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"]/max(city_totals[c]["demand"],1),
                    reverse=True)
    rates  = [city_totals[c]["unmet"]/max(city_totals[c]["demand"],1)*100 for c in cities]
    colors = ["#c44e52" if r >= TARGET_UNMET*100 else "#4c72b0" for r in rates]

    fig, ax = plt.subplots(figsize=(max(8, len(cities)*0.9), 5))
    bars = ax.bar(cities, rates, color=colors, alpha=0.85, width=0.6)
    ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--",
               linewidth=1.5, label="Service target (5%)")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average unmet demand rate (%)", fontsize=12)
    ax.set_xlabel("City (origin)", fontsize=12)
    ax.set_title(f"Overall unmet demand rate per city — {OPTIMAL[policy_key]['label']}",
                 fontsize=12)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.25, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=10); plt.tight_layout()
    _savefig(fig, _plot_dir(policy_key) / "fig_city_overall.png")
    plt.show(); plt.close(fig)


def plot_hourly_per_city(policy_key: str, by_city_hour: dict, city_totals: dict):
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"]/max(city_totals[c]["demand"],1),
                    reverse=True)
    n = len(cities)
    ncols = min(3, n); nrows = (n+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), sharey=False)
    if n==1: axes=[axes]
    elif nrows==1: axes=list(axes)
    else: axes=[ax for row in axes for ax in row]
    for i, city in enumerate(cities):
        ax = axes[i]
        rates  = [_unmet_rate(by_city_hour.get(city,{}), h) for h in range(24)]
        colors = ["#c44e52" if r>=TARGET_UNMET*100 else "#4c72b0" for r in rates]
        ax.bar(range(24), rates, color=colors, alpha=0.85, width=0.8)
        ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--", linewidth=1.2)
        overall = city_totals[city]["unmet"]/max(city_totals[city]["demand"],1)*100
        ax.set_title(f"{city}\n(avg {overall:.1f}% unmet)", fontsize=10)
        ax.set_xticks(range(0,24,4))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0,24,4)], fontsize=7)
        ax.set_ylim(bottom=0); ax.set_ylabel("Unmet rate (%)", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"Unmet demand rate by hour of day — {OPTIMAL[policy_key]['label']}\n"
                 "(red bars exceed 5% target; cities ordered by overall unmet rate)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, _plot_dir(policy_key) / "fig_city_hourly.png")
    plt.show(); plt.close(fig)


def plot_weekday_per_city(policy_key: str, by_city_weekday: dict, city_totals: dict):
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"]/max(city_totals[c]["demand"],1),
                    reverse=True)
    n = len(cities)
    ncols = min(3, n); nrows = (n+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows), sharey=False)
    if n==1: axes=[axes]
    elif nrows==1: axes=list(axes)
    else: axes=[ax for row in axes for ax in row]
    for i, city in enumerate(cities):
        ax    = axes[i]
        rates = [_unmet_rate(by_city_weekday.get(city,{}), d) for d in WEEKDAY_ORDER]
        colors = ["#c44e52" if r>=TARGET_UNMET*100 else "#4c72b0" for r in rates]
        bars = ax.bar(range(7), rates, color=colors, alpha=0.85, width=0.65)
        ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--", linewidth=1.2)
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, val+0.1,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5)
        overall = city_totals[city]["unmet"]/max(city_totals[city]["demand"],1)*100
        ax.set_title(f"{city}\n(avg {overall:.1f}% unmet)", fontsize=10)
        ax.set_xticks(range(7)); ax.set_xticklabels(WEEKDAY_SHORT, fontsize=8)
        ax.set_ylim(bottom=0); ax.set_ylabel("Unmet rate (%)", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
    for j in range(i+1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"Unmet demand rate by day of week — {OPTIMAL[policy_key]['label']}\n"
                 "(red bars exceed 5% target; cities ordered by overall unmet rate)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _savefig(fig, _plot_dir(policy_key) / "fig_city_weekday.png")
    plt.show(); plt.close(fig)


def plot_heatmap(policy_key: str, by_city_hour: dict, by_city_weekday: dict,
                 city_totals: dict):
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"]/max(city_totals[c]["demand"],1),
                    reverse=True)
    n = len(cities)
    hour_matrix = np.array([[_unmet_rate(by_city_hour.get(c,{}),h) for h in range(24)]
                             for c in cities])
    wd_matrix   = np.array([[_unmet_rate(by_city_weekday.get(c,{}),d) for d in WEEKDAY_ORDER]
                             for c in cities])
    vmax = max(hour_matrix.max(), wd_matrix.max())
    fig, (ax_h, ax_w) = plt.subplots(1, 2, figsize=(16, max(4, n*0.6+2)),
                                      gridspec_kw={"width_ratios": [3,1]})
    im1 = ax_h.imshow(hour_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax_h.set_xticks(range(24))
    ax_h.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
    ax_h.set_yticks(range(n)); ax_h.set_yticklabels(cities, fontsize=9)
    ax_h.set_xlabel("Hour of day", fontsize=10)
    ax_h.set_title("Unmet demand rate by hour of day (%)", fontsize=11)
    for i in range(n):
        for j in range(24):
            val = hour_matrix[i,j]
            if val >= TARGET_UNMET*100:
                ax_h.text(j,i,f"{val:.0f}",ha="center",va="center",fontsize=5.5,
                          color="white" if val>10 else "black", fontweight="bold")
    ax_h.contour(hour_matrix, levels=[TARGET_UNMET*100],
                  colors=["navy"], linewidths=1.0, linestyles="--")
    cbar1 = fig.colorbar(im1, ax=ax_h, pad=0.01, shrink=0.8)
    cbar1.set_label("Unmet rate (%)", fontsize=9)
    cbar1.ax.axhline(TARGET_UNMET*100, color="navy", linewidth=1.5, linestyle="--")
    im2 = ax_w.imshow(wd_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax_w.set_xticks(range(7)); ax_w.set_xticklabels(WEEKDAY_SHORT, fontsize=8)
    ax_w.set_yticks(range(n)); ax_w.set_yticklabels([""]*n)
    ax_w.set_xlabel("Day of week", fontsize=10)
    ax_w.set_title("By weekday (%)", fontsize=11)
    for i in range(n):
        for j in range(7):
            val = wd_matrix[i,j]
            if val > 0:
                ax_w.text(j,i,f"{val:.1f}",ha="center",va="center",fontsize=7,
                          color="white" if val>10 else "black")
    cbar2 = fig.colorbar(im2, ax=ax_w, pad=0.01, shrink=0.8)
    cbar2.set_label("Unmet rate (%)", fontsize=9)
    fig.suptitle(f"{OPTIMAL[policy_key]['label']} — unmet demand rate by city, hour, and weekday\n"
                 "(dashed contour: 5% target; cities sorted by overall unmet rate)",
                 fontsize=12)
    plt.tight_layout()
    _savefig(fig, _plot_dir(policy_key) / "fig_city_heatmap.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  COMPARISON SPATIAL FIGURES
# ════════════════════════════════════════════════════════════════════════════

def plot_compare_overall(all_totals: dict):
    policies = list(all_totals.keys())
    cities   = sorted(all_totals.get("baseline", next(iter(all_totals.values()))).keys(),
                      key=lambda c: all_totals.get("baseline",
                          next(iter(all_totals.values())))[c]["unmet"] /
                                    max(all_totals.get("baseline",
                          next(iter(all_totals.values())))[c]["demand"],1),
                      reverse=True)
    n = len(cities); m = len(policies)
    x = np.arange(n); width = 0.8/m

    fig, ax = plt.subplots(figsize=(max(10, n*1.2), 6))
    for i, policy_key in enumerate(policies):
        tots  = all_totals[policy_key]
        rates = [tots.get(c,{"demand":1,"unmet":0})["unmet"] /
                 max(tots.get(c,{"demand":1,"unmet":0})["demand"],1)*100
                 for c in cities]
        ax.bar(x + i*width - (m-1)*width/2, rates, width*0.9,
               label=OPTIMAL[policy_key]["label"],
               color=_COLORS[policy_key], alpha=0.85)

    ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--",
               linewidth=1.5, label="5% target")
    ax.set_xticks(x); ax.set_xticklabels(cities, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Overall unmet demand rate per city — all policies at optimal configurations",
                 fontsize=12)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, _comp_dir() / "fig_compare_overall.png")
    plt.show(); plt.close(fig)


def plot_compare_hourly(all_data: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    for policy_key, (by_city_hour, _, _ct) in all_data.items():
        hour_demand = defaultdict(int); hour_unmet = defaultdict(int)
        for city, hdata in by_city_hour.items():
            for h in range(24):
                b = hdata.get(h, {"demand":0,"unmet":0})
                hour_demand[h] += b["demand"]; hour_unmet[h] += b["unmet"]
        rates = [hour_unmet[h]/max(hour_demand[h],1)*100 for h in range(24)]
        ax.plot(range(24), rates, color=_COLORS[policy_key], linewidth=2.5,
                marker="o", markersize=4, label=OPTIMAL[policy_key]["label"])
    ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--",
               linewidth=1.2, label="5% target")
    ax.set_xticks(range(0,24,2))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0,24,2)], fontsize=9)
    ax.set_xlabel("Hour of day", fontsize=12); ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Aggregate hourly unmet demand rate — all policies at optimal configurations",
                 fontsize=12)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, _comp_dir() / "fig_compare_hourly.png")
    plt.show(); plt.close(fig)


def plot_compare_weekday(all_data: dict):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(7); m = len(all_data); width = 0.7/m
    for i, (policy_key, (_, by_city_weekday, _ct)) in enumerate(all_data.items()):
        wd_demand = defaultdict(int); wd_unmet = defaultdict(int)
        for city, wdata in by_city_weekday.items():
            for d in WEEKDAY_ORDER:
                b = wdata.get(d,{"demand":0,"unmet":0})
                wd_demand[d]+=b["demand"]; wd_unmet[d]+=b["unmet"]
        rates = [wd_unmet[d]/max(wd_demand[d],1)*100 for d in WEEKDAY_ORDER]
        ax.bar(x + i*width - (m-1)*width/2, rates, width*0.9,
               label=OPTIMAL[policy_key]["label"],
               color=_COLORS[policy_key], alpha=0.85)
    ax.axhline(TARGET_UNMET*100, color="gray", linestyle="--",
               linewidth=1.5, label="5% target")
    ax.set_xticks(x); ax.set_xticklabels(WEEKDAY_SHORT, fontsize=11)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title("Aggregate day of week unmet demand rate — all policies at optimal configurations",
                 fontsize=12)
    ax.set_ylim(bottom=0); ax.legend(fontsize=10); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, _comp_dir() / "fig_compare_weekday.png")
    plt.show(); plt.close(fig)


def plot_compare_heatmap_diff(all_data: dict):
    if "baseline" not in all_data:
        print("  Skipping heatmap diff: no baseline data")
        return
    base_hour, _, base_totals = all_data["baseline"]
    policies = [p for p in all_data if p != "baseline"]
    cities   = sorted(base_totals.keys(),
                      key=lambda c: base_totals[c]["unmet"]/max(base_totals[c]["demand"],1),
                      reverse=True)
    n = len(cities)
    base_mat = np.array([[_unmet_rate(base_hour.get(c,{}),h) for h in range(24)]
                          for c in cities])
    fig, axes = plt.subplots(1, len(policies), figsize=(8*len(policies), max(4,n*0.6+2)))
    if len(policies)==1: axes=[axes]
    for ax, policy_key in zip(axes, policies):
        by_city_hour = all_data[policy_key][0]
        pol_mat  = np.array([[_unmet_rate(by_city_hour.get(c,{}),h) for h in range(24)]
                              for c in cities])
        diff_mat = pol_mat - base_mat
        vext = max(abs(diff_mat).max(), 1)
        im = ax.imshow(diff_mat, aspect="auto", cmap="RdYlGn_r", vmin=-vext, vmax=vext)
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
        ax.set_yticks(range(n)); ax.set_yticklabels(cities, fontsize=9)
        ax.set_xlabel("Hour of day", fontsize=10)
        ax.set_title(f"{OPTIMAL[policy_key]['label']} − Baseline\n"
                     f"(green = improvement, red = worse)", fontsize=11)
        for i in range(n):
            for j in range(24):
                val = diff_mat[i,j]
                if abs(val) >= 1.0:
                    ax.text(j,i,f"{val:+.0f}",ha="center",va="center",
                            fontsize=5, color="black")
        cbar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8)
        cbar.set_label("Δ unmet rate (pp)", fontsize=9)
    fig.suptitle("Change in hourly unmet demand rate vs baseline (percentage points)\n"
                 "Each relocation policy at its optimal configuration", fontsize=12)
    plt.tight_layout()
    _savefig(fig, _comp_dir() / "fig_compare_heatmap_diff.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 6: full policy comparison")
    parser.add_argument("--plot",          action="store_true",
                        help="Generate plots from saved data only")
    parser.add_argument("--rerun",         action="store_true",
                        help="Force fresh simulations even if cached data exists")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip policies whose event log already exists")
    parser.add_argument("--policies",      nargs="+",
                        default=POLICY_ORDER, choices=POLICY_ORDER)
    args = parser.parse_args()

    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 6 — Full policy comparison")
    print(f"  Policies: {args.policies}")
    print(f"{'═'*60}")

    # ── collect financial results ────────────────────────────────────────
    if args.plot and _results_exist():
        results_list = _load_results()
        results = {r["policy"]: r for r in results_list}
    else:
        print("\nCollecting optimal-configuration results...")
        results = collect_results(args.policies, force_rerun=args.rerun)
        _save_results(list(results.values()))

    print_summary(results)
    plot_comparison(results)

    # ── spatial analysis ────────────────────────────────────────────────
    all_data   = {}
    all_totals = {}

    for policy_key in args.policies:
        log_path = _log_path(policy_key)

        if args.plot or (args.skip_existing and log_path.exists()):
            if log_path.exists():
                print(f"\n  Loading saved log for {policy_key}...")
                with open(log_path) as f:
                    log = json.load(f)
            else:
                print(f"  WARNING: no log for {policy_key} — run without --plot first")
                continue
        else:
            if not SIM_AVAILABLE:
                print(f"  Skipping spatial analysis for {policy_key}: sim unavailable")
                continue
            cfg = OPTIMAL[policy_key]
            model  = FinancialModel(goodwill_multiplier=GOODWILL)
            policy = load_policy(policy_name=cfg["policy_name"], **cfg["params"])
            engine = SimulationEngine(
                n_days=N_DAYS, policy=policy,
                fleet_size_override=cfg["fleet"], verbose=False
            )
            with ws._patch_workers(cfg["workers"]):
                engine.run(verbose=False)
            log = engine.event_log
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(log, f)
            print(f"  Event log saved: {log_path}  ({len(log):,} events)")

        by_city_hour, by_city_weekday, city_totals = aggregate(log)
        print_city_summary(policy_key, city_totals)
        all_data[policy_key]   = (by_city_hour, by_city_weekday, city_totals)
        all_totals[policy_key] = city_totals

        plot_city_overall(policy_key, city_totals)
        plot_hourly_per_city(policy_key, by_city_hour, city_totals)
        plot_weekday_per_city(policy_key, by_city_weekday, city_totals)
        plot_heatmap(policy_key, by_city_hour, by_city_weekday, city_totals)

    if len(all_data) > 1:
        print("\n  Generating comparison figures...")
        plot_compare_overall(all_totals)
        plot_compare_hourly(all_data)
        plot_compare_weekday(all_data)
        if "baseline" in all_data:
            plot_compare_heatmap_diff(all_data)

    print(f"\n  Done. All figures saved under {DATA_DIR}")


if __name__ == "__main__":
    main()
