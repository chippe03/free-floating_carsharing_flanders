"""
0_baseline_characterization.py  -  Baseline characterization (Step 0).

Examines the no-relocation baseline over an extended simulation horizon of
300 days with a fleet of 650 vehicles. Runs the simulation, saves the full
event log, then produces all diagnostic plots.

Produces seven figures:
  fig_baseline_unmet_daily        Unmet demand rate aggregated per day over time
  fig_baseline_unmet_hourly       Unmet demand rate aggregated per hour over time
  fig_baseline_stability          Long-run daily unmet rate with trend line
  fig_baseline_demand_pattern     Avg unmet rate by hour and by day of week (aggregate)
  fig_city_hourly                 Avg unmet rate by hour of day, one subplot per city
  fig_city_weekday                Avg unmet rate by day of week, one subplot per city
  fig_city_heatmap                Heatmap: city × hour and city × weekday
  fig_city_overall                Overall avg unmet rate per city (bar chart)

Usage
-----
  python 0_baseline_characterization.py            # run simulation + all plots
  python 0_baseline_characterization.py --plot     # plots only (requires saved log)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress

from src import config

# ── constants ────────────────────────────────────────────────────────────────

N_DAYS        = 300
FLEET_SIZE    = 650
TARGET_UNMET  = 0.05

LOGS_DIR      = Path("./data/metrics/0_baseline")
PLOTS_DIR     = LOGS_DIR / "plots"
LOG_PATH      = LOGS_DIR / "baseline_events.json"

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ── simulation ───────────────────────────────────────────────────────────────

def run_baseline_log(n_days: int, fleet_size: int) -> list[dict]:
    from src.relocation.create_policy import load_policy
    from src.simulation.engine import SimulationEngine

    print(f"Running baseline simulation: {n_days} days, fleet={fleet_size} ...")
    policy = load_policy(policy_name="baseline")
    engine = SimulationEngine(
        n_days              = n_days,
        policy              = policy,
        fleet_size_override = fleet_size,
        verbose             = False,
    )
    engine.run(verbose=False)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(engine.event_log, f, indent=2)
    print(f"  Event log saved to {LOG_PATH}")
    return engine.event_log


def load_log(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ── aggregation helpers ──────────────────────────────────────────────────────

def daily_unmet(log: list[dict]):
    """Returns (days, unmet_rates, unmet_counts, demand_counts)."""
    demand = defaultdict(int)
    unmet  = defaultdict(int)
    for e in log:
        if e["event"] not in ("TRIP_ASSIGNED", "TRIP_UNMET"):
            continue
        day = e.get("day", 0)
        demand[day] += 1
        if e["event"] == "TRIP_UNMET":
            unmet[day] += 1
    days    = sorted(demand.keys())
    rates   = [unmet[d] / demand[d] if demand[d] > 0 else 0.0 for d in days]
    counts  = [unmet[d]  for d in days]
    demands = [demand[d] for d in days]
    return days, rates, counts, demands


def hourly_profile(log: list[dict]):
    """Average unmet rate by hour of day across all days."""
    demand = defaultdict(int)
    unmet  = defaultdict(int)
    for e in log:
        if e["event"] not in ("TRIP_ASSIGNED", "TRIP_UNMET"):
            continue
        hour = int(e.get("hour", 0))
        demand[hour] += 1
        if e["event"] == "TRIP_UNMET":
            unmet[hour] += 1
    hours = sorted(demand.keys())
    rates = [unmet[h] / demand[h] * 100 if demand[h] > 0 else 0.0 for h in hours]
    return hours, rates


def weekday_profile(log: list[dict]):
    """Average unmet rate by day of week."""
    demand = defaultdict(int)
    unmet  = defaultdict(int)
    for e in log:
        if e["event"] not in ("TRIP_ASSIGNED", "TRIP_UNMET"):
            continue
        wd = e.get("weekday", "Unknown")
        demand[wd] += 1
        if e["event"] == "TRIP_UNMET":
            unmet[wd] += 1
    days  = [d for d in WEEKDAY_ORDER if d in demand]
    rates = [unmet[d] / demand[d] * 100 if demand[d] > 0 else 0.0 for d in days]
    return days, rates


def aggregate_by_city(log: list[dict]):
    """
    Returns:
      by_city_hour[city][hour]   = {"demand": int, "unmet": int}
      by_city_weekday[city][day] = {"demand": int, "unmet": int}
      city_totals[city]          = {"demand": int, "unmet": int}
    """
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


def _city_unmet_rate(d: dict, key) -> float:
    bucket = d.get(key, {"demand": 0, "unmet": 0})
    return bucket["unmet"] / bucket["demand"] * 100 if bucket["demand"] > 0 else 0.0


# ── Figure 1 & 2: unmet over time (daily and hourly windows) ─────────────────

def plot_unmet_over_time(
    event_log:    list[dict],
    window_hours: int = 24,
    save_path:    str = None,
):
    """
    Two-panel: unmet rate over time (top) and absolute unmet counts (bottom).
    window_hours=24 → daily;  window_hours=1 → hourly.
    """
    window_min = window_hours * 60
    demand = defaultdict(int)
    unmet  = defaultdict(int)

    for e in event_log:
        if e["event"] not in ("TRIP_ASSIGNED", "TRIP_UNMET"):
            continue
        t      = e.get("time", e.get("sim_time", 0))
        bucket = int(t // window_min)
        demand[bucket] += 1
        if e["event"] == "TRIP_UNMET":
            unmet[bucket] += 1

    buckets      = sorted(demand.keys())
    x_vals       = [b * window_hours for b in buckets]
    unmet_rates  = [unmet[b] / demand[b] * 100 if demand[b] > 0 else 0 for b in buckets]
    unmet_counts = [unmet[b] for b in buckets]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(x_vals, unmet_rates, color="#d62728", linewidth=1.5)
    ax1.fill_between(x_vals, unmet_rates, alpha=0.2, color="#d62728")
    ax1.axhline(5, color="gray", linestyle="--", linewidth=1, label="5% target")
    ax1.set_ylabel("Unmet demand rate (%)", fontsize=11)
    ax1.set_title(
        f"Unmet demand over time — baseline "
        f"({'daily' if window_hours == 24 else f'{window_hours}h'} aggregation)",
        fontsize=13
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.bar(x_vals, unmet_counts, width=window_hours * 0.8, color="#d62728", alpha=0.7)
    ax2.set_xlabel(
        f"Time (hours from simulation start, "
        f"{'days' if window_hours == 24 else 'hours'} on x-axis)",
        fontsize=11
    )
    ax2.set_ylabel("Unmet trips", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    day_ticks = [d * 24 for d in range(0, int(max(x_vals) / 24) + 2)]
    ax2_top.set_xticks(day_ticks)
    ax2_top.set_xticklabels([f"Day {d+1}" for d in range(len(day_ticks))], fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── Figure 3: long-run daily stability ───────────────────────────────────────

def plot_daily_stability(
    days:      list[int],
    rates:     list[float],
    counts:    list[int],
    save_path: str = None,
):
    """
    Single-panel daily unmet rate with rolling average and linear trend.
    """
    rates_pct = [r * 100 for r in rates]
    n_days    = len(days)

    slope, intercept, r, p, _ = linregress(days, rates_pct)
    trend = [intercept + slope * d for d in days]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(days, rates_pct, alpha=0.15, color="#FF3300")
    ax.plot(days, rates_pct, color="#FF3300", linewidth=0.9, alpha=0.85,
            label="Daily unmet rate")

    window = 7
    if n_days >= window:
        padded  = [rates_pct[0]] * (window - 1) + rates_pct
        rolling = [np.mean(padded[i:i + window]) for i in range(n_days)]
        ax.plot(days, rolling, color="#8B0000", linewidth=2,
                label=f"{window}-day rolling average")

    ax.plot(days, trend, color="navy", linewidth=1.5, linestyle="--",
            label=f"Trend: slope = {slope:+.4f}%/day  (p = {p:.3f})")
    ax.axhline(TARGET_UNMET * 100, color="gray", linestyle=":", linewidth=1.5,
               label=f"Service target ({TARGET_UNMET:.0%})")

    mean_rate = np.mean(rates_pct)
    std_rate  = np.std(rates_pct)
    ax.text(0.01, 0.97,
            f"Mean: {mean_rate:.1f}%   Std: {std_rate:.1f}%   "
            f"Min: {min(rates_pct):.1f}%   Max: {max(rates_pct):.1f}%",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="lightgray", alpha=0.8))

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title(
        f"Baseline policy - daily unmet demand rate over {n_days} simulation days\n",
        fontsize=12
    )
    ax.set_xlim(0, n_days - 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.25)

    week_ticks = list(range(0, n_days, 28))
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([f"Week {t // 7 + 1}" for t in week_ticks], fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)
    return slope, p


# ── Figure 4: aggregate demand pattern (hour + weekday) ──────────────────────

def plot_demand_pattern(
    hours:      list[int],
    hour_rates: list[float],
    weekdays:   list[str],
    wd_rates:   list[float],
    save_path:  str = None,
):
    """
    Two-panel: avg unmet rate by hour of day (left) and by day of week (right).
    """
    fig, (ax_h, ax_w) = plt.subplots(1, 2, figsize=(13, 5))

    bar_colors_h = ["#c44e52" if r >= TARGET_UNMET * 100 else "#4c72b0" for r in hour_rates]
    ax_h.bar(hours, hour_rates, color=bar_colors_h, alpha=0.85, width=0.8)
    ax_h.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.5,
                 label=f"Service target ({TARGET_UNMET:.0%})")
    ax_h.set_xlabel("Hour of day", fontsize=11)
    ax_h.set_ylabel("Avg. unmet demand rate (%)", fontsize=11)
    ax_h.set_title("Unmet demand by hour of day", fontsize=12)
    ax_h.set_xticks(range(0, 24, 2))
    ax_h.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, fontsize=8)
    ax_h.set_ylim(bottom=0)
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.25, axis="y")

    wd_short      = [d[:3] for d in weekdays]
    bar_colors_w  = ["#c44e52" if r >= TARGET_UNMET * 100 else "#4c72b0" for r in wd_rates]
    bars = ax_w.bar(range(len(weekdays)), wd_rates, color=bar_colors_w, alpha=0.85, width=0.65)
    ax_w.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.5,
                 label=f"Service target ({TARGET_UNMET:.0%})")
    for bar, val in zip(bars, wd_rates):
        ax_w.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.1f}%",
                  ha="center", va="bottom", fontsize=8)
    ax_w.set_xlabel("Day of week", fontsize=11)
    ax_w.set_ylabel("Avg. unmet demand rate (%)", fontsize=11)
    ax_w.set_title("Unmet demand by day of week", fontsize=12)
    ax_w.set_xticks(range(len(weekdays)))
    ax_w.set_xticklabels(wd_short, fontsize=10)
    ax_w.set_ylim(bottom=0)
    ax_w.legend(fontsize=9)
    ax_w.grid(True, alpha=0.25, axis="y")

    fig.suptitle(
        "Baseline policy - demand patterns driving service variability\n"
        "(red bars exceed 5% target)",
        fontsize=12
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── Figure 5: hourly per city ─────────────────────────────────────────────────

def plot_hourly_per_city(by_city_hour: dict, city_totals: dict, save_path: str = None):
    """One subplot per city: avg unmet rate by hour of day."""
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"] / max(city_totals[c]["demand"], 1),
                    reverse=True)
    n = len(cities)
    if n == 0:
        print("  No city data found.")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=False)

    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    hours = list(range(24))
    for i, city in enumerate(cities):
        ax     = axes[i]
        rates  = [_city_unmet_rate(by_city_hour.get(city, {}), h) for h in hours]
        colors = ["#c44e52" if r >= TARGET_UNMET * 100 else "#4c72b0" for r in rates]
        ax.bar(hours, rates, color=colors, alpha=0.85, width=0.8)
        ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.2)
        overall_rate = city_totals[city]["unmet"] / max(city_totals[city]["demand"], 1) * 100
        ax.set_title(f"{city}\n(avg {overall_rate:.1f}% unmet)", fontsize=10)
        ax.set_xticks(range(0, 24, 4))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 4)], fontsize=7)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Unmet rate (%)", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Unmet demand rate by hour of day - per city\n"
        "(red bars exceed 5% target; cities ordered by overall unmet rate)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── Figure 6: weekday per city ────────────────────────────────────────────────

def plot_weekday_per_city(by_city_weekday: dict, city_totals: dict, save_path: str = None):
    """One subplot per city: avg unmet rate by day of week."""
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"] / max(city_totals[c]["demand"], 1),
                    reverse=True)
    n = len(cities)
    if n == 0:
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), sharey=False)

    if n == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, city in enumerate(cities):
        ax     = axes[i]
        rates  = [_city_unmet_rate(by_city_weekday.get(city, {}), d) for d in WEEKDAY_ORDER]
        colors = ["#c44e52" if r >= TARGET_UNMET * 100 else "#4c72b0" for r in rates]
        bars   = ax.bar(range(7), rates, color=colors, alpha=0.85, width=0.65)
        ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.2)
        for bar, val in zip(bars, rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=6.5)
        overall_rate = city_totals[city]["unmet"] / max(city_totals[city]["demand"], 1) * 100
        ax.set_title(f"{city}\n(avg {overall_rate:.1f}% unmet)", fontsize=10)
        ax.set_xticks(range(7))
        ax.set_xticklabels(WEEKDAY_SHORT, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Unmet rate (%)", fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Unmet demand rate by day of week - per city\n"
        "(red bars exceed 5% target; cities ordered by overall unmet rate)",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── Figure 7: city × hour / city × weekday heatmap ───────────────────────────

def plot_heatmap(by_city_hour: dict, by_city_weekday: dict,
                 city_totals: dict, save_path: str = None):
    """
    Two-panel heatmap: city × hour (left) and city × weekday (right).
    Most compact view; best for the thesis appendix.
    """
    cities = sorted(city_totals.keys(),
                    key=lambda c: city_totals[c]["unmet"] / max(city_totals[c]["demand"], 1),
                    reverse=True)
    n = len(cities)

    hour_matrix = np.array([
        [_city_unmet_rate(by_city_hour.get(c, {}), h) for h in range(24)]
        for c in cities
    ])
    wd_matrix = np.array([
        [_city_unmet_rate(by_city_weekday.get(c, {}), d) for d in WEEKDAY_ORDER]
        for c in cities
    ])

    fig, (ax_h, ax_w) = plt.subplots(1, 2,
                                      figsize=(16, max(4, n * 0.6 + 2)),
                                      gridspec_kw={"width_ratios": [3, 1]})

    vmax = max(hour_matrix.max(), wd_matrix.max())
    im1  = ax_h.imshow(hour_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax_h.set_xticks(range(24))
    ax_h.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=7, rotation=45)
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels(cities, fontsize=9)
    ax_h.set_xlabel("Hour of day", fontsize=10)
    ax_h.set_title("Unmet demand rate by hour of day (%)", fontsize=11)
    for i in range(n):
        for j in range(24):
            val = hour_matrix[i, j]
            if val >= TARGET_UNMET * 100:
                ax_h.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.5,
                          color="white" if val > 10 else "black", fontweight="bold")
    ax_h.contour(hour_matrix, levels=[TARGET_UNMET * 100],
                  colors=["navy"], linewidths=1.0, linestyles="--")
    cbar1 = fig.colorbar(im1, ax=ax_h, pad=0.01, shrink=0.8)
    cbar1.set_label("Unmet rate (%)", fontsize=9)
    cbar1.ax.axhline(TARGET_UNMET * 100, color="navy", linewidth=1.5, linestyle="--")

    im2 = ax_w.imshow(wd_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax_w.set_xticks(range(7))
    ax_w.set_xticklabels(WEEKDAY_SHORT, fontsize=8)
    ax_w.set_yticks(range(n))
    ax_w.set_yticklabels([""] * n)
    ax_w.set_xlabel("Day of week", fontsize=10)
    ax_w.set_title("By weekday (%)", fontsize=11)
    for i in range(n):
        for j in range(7):
            val = wd_matrix[i, j]
            if val > 0:
                ax_w.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                          color="white" if val > 10 else "black")
    cbar2 = fig.colorbar(im2, ax=ax_w, pad=0.01, shrink=0.8)
    cbar2.set_label("Unmet rate (%)", fontsize=9)

    fig.suptitle(
        "Baseline policy - unmet demand rate by city, hour, and weekday\n"
        "(dashed contour: 5% service target; cities sorted by overall unmet rate)",
        fontsize=12
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── Figure 8: city overall bar ────────────────────────────────────────────────

def plot_city_overall(city_totals: dict, save_path: str = None):
    """Overall avg unmet rate per city."""
    cities  = sorted(city_totals.keys(),
                     key=lambda c: city_totals[c]["unmet"] / max(city_totals[c]["demand"], 1),
                     reverse=True)
    rates   = [city_totals[c]["unmet"] / max(city_totals[c]["demand"], 1) * 100 for c in cities]
    demands = [city_totals[c]["demand"] for c in cities]
    colors  = ["#c44e52" if r >= TARGET_UNMET * 100 else "#4c72b0" for r in rates]

    fig, ax = plt.subplots(figsize=(max(8, len(cities) * 0.9), 5))
    bars = ax.bar(cities, rates, color=colors, alpha=0.85, width=0.6)
    ax.axhline(TARGET_UNMET * 100, color="gray", linestyle="--", linewidth=1.5,
               label=f"Service target ({TARGET_UNMET:.0%})")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average unmet demand rate (%)", fontsize=12)
    ax.set_xlabel("City (origin)", fontsize=12)
    ax.set_title("Overall unmet demand rate per city - baseline policy", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close(fig)


# ── plotting orchestrator ─────────────────────────────────────────────────────

def run_plots(log: list[dict], out: Path):
    out.mkdir(parents=True, exist_ok=True)

    # aggregate once, reuse across all plot functions
    days, rates, counts, demands   = daily_unmet(log)
    hours, hour_rates               = hourly_profile(log)
    weekdays, wd_rates              = weekday_profile(log)
    by_city_hour, by_city_weekday, city_totals = aggregate_by_city(log)

    n_days      = len(days)
    mean_rate   = np.mean(rates) * 100
    total_unmet = sum(counts)
    print(f"\n  {n_days} simulation days")
    print(f"  Mean daily unmet rate : {mean_rate:.1f}%")
    print(f"  Total unmet trips     : {total_unmet:,}")

    print("\n  Overall unmet rate per city:")
    for city in sorted(city_totals, key=lambda c: city_totals[c]["unmet"] /
                       max(city_totals[c]["demand"], 1), reverse=True):
        t    = city_totals[city]
        rate = t["unmet"] / max(t["demand"], 1) * 100
        print(f"    {city:<15} {rate:>5.1f}%  ({t['unmet']:,} / {t['demand']:,})")

    if "kortrijk" in city_totals:
        kortrijk_unmet = city_totals["kortrijk"]["unmet"]
        total_u        = sum(c["unmet"] for c in city_totals.values())
        print(f"\n  Kortrijk share: {kortrijk_unmet / total_u:.1%}")

    # Fig 1 – daily unmet over time
    print("\nPlotting fig_baseline_unmet_daily ...")
    plot_unmet_over_time(log, window_hours=24,
                         save_path=str(out / "fig_baseline_unmet_daily.png"))

    # Fig 2 – hourly unmet over time
    print("Plotting fig_baseline_unmet_hourly ...")
    plot_unmet_over_time(log, window_hours=1,
                         save_path=str(out / "fig_baseline_unmet_hourly.png"))

    # Fig 3 – long-run stability
    print("Plotting fig_baseline_stability ...")
    slope, p = plot_daily_stability(days, rates, counts,
                                    save_path=str(out / "fig_baseline_stability.png"))
    print(f"  Trend slope: {slope:+.4f}%/day  (p = {p:.3f})")
    if p > 0.05:
        print("  → Slope not statistically significant: no accumulating imbalance confirmed.")
    else:
        print("  → Slope IS statistically significant - review claim of stability.")

    # Fig 4 – aggregate demand pattern
    print("Plotting fig_baseline_demand_pattern ...")
    plot_demand_pattern(hours, hour_rates, weekdays, wd_rates,
                        save_path=str(out / "fig_baseline_demand_pattern.png"))

    # Fig 5 – hourly per city
    print("Plotting fig_city_hourly ...")
    plot_hourly_per_city(by_city_hour, city_totals,
                         save_path=str(out / "fig_city_hourly.png"))

    # Fig 6 – weekday per city
    print("Plotting fig_city_weekday ...")
    plot_weekday_per_city(by_city_weekday, city_totals,
                          save_path=str(out / "fig_city_weekday.png"))

    # Fig 7 – heatmap
    print("Plotting fig_city_heatmap ...")
    plot_heatmap(by_city_hour, by_city_weekday, city_totals,
                 save_path=str(out / "fig_city_heatmap.png"))

    # Fig 8 – city overall
    print("Plotting fig_city_overall ...")
    plot_city_overall(city_totals,
                      save_path=str(out / "fig_city_overall.png"))

    print(f"\n  All plots saved to {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline characterization: simulate and/or plot."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Skip simulation; load existing event log and re-plot only."
    )
    parser.add_argument("--log",    default=str(LOG_PATH),  help="Path to event log JSON.")
    parser.add_argument("--output", default=str(PLOTS_DIR), help="Output directory for plots.")
    args = parser.parse_args()

    out = Path(args.output)

    if args.plot:
        print(f"Loading existing event log: {args.log}")
        log = load_log(Path(args.log))
        print(f"  {len(log):,} events loaded")
    else:
        log = run_baseline_log(n_days=N_DAYS, fleet_size=FLEET_SIZE)

    run_plots(log, out)


if __name__ == "__main__":
    main()
