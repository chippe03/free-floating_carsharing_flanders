"""
7_sensitivity_goodwill.py  -  Goodwill sensitivity analysis (Step 7a).

From each Step 6 result, profit at any goodwill multiplier is derived
analytically using the identity:
  profit(m) = cash_profit - m × lost_revenue_per_day

No new simulations are needed: the full sensitivity curve over any
multiplier range is computed directly from the two scalars stored in the
Step 6 optimal_results.json.

The breakeven goodwill multiplier is the value of m at which a relocation
policy's profit curve crosses the baseline curve — i.e. the minimum
valuation of an unmet trip at which relocation pays for itself.

Produces four figures:
  fig_7_gw_comparison     Bar chart: all policies at goodwill=1.0 and cash=0 (from Step 6)
  fig_7_gw_sensitivity    Profit vs goodwill multiplier — all policies on one plot
  fig_7_gw_difference     Profit difference vs baseline across multipliers
  fig_7_gw_breakeven      Breakeven multiplier per policy — bar chart

Usage
-----
  python 7_sensitivity_goodwill.py          # load Step 6 results + plot
  python 7_sensitivity_goodwill.py --plot   # same (always plot-only for this step)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from final_param import OPTIMAL as OPTIMAL_CFG


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

N_DAYS       = 30
POLICY_ORDER = ["baseline", "reactive", "nightly", "proactive"]

DATA_DIR  = Path("./data/metrics/7_goodwill")
PLOTS_DIR = DATA_DIR / "plots"

STEP6_RESULTS = Path("./data/metrics/6_comparison/optimal_results.json")

# Colors from final_param.OPTIMAL
_COLORS = {k: v["color"] for k, v in OPTIMAL_CFG.items()}
_LABELS = {k: v["label"] for k, v in OPTIMAL_CFG.items()}

# Goodwill multiplier range — dense near 0–2, sparser beyond
_m1 = [round(x, 2) for x in np.arange(0, 1.05, 0.05)]
_m2 = [round(x, 2) for x in np.arange(1.0, 2.1,  0.1)]
_m3 = [round(x, 1) for x in np.arange(2.0, 5.5,  0.5)]
MULTIPLIERS = sorted(set(_m1 + _m2 + _m3))


# ════════════════════════════════════════════════════════════════════════════
#  IO
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


def load_step6_results() -> dict:
    if not STEP6_RESULTS.exists():
        raise FileNotFoundError(
            f"Step 6 results not found at {STEP6_RESULTS}. "
            "Run 6_full_comparison.py first."
        )
    with open(STEP6_RESULTS) as f:
        data = json.load(f)
    return {r["policy"]: r for r in data}


# ════════════════════════════════════════════════════════════════════════════
#  SENSITIVITY HELPERS
# ════════════════════════════════════════════════════════════════════════════

def cash_profit(r: dict) -> float:
    """Cash-based profit/day: goodwill profit + lost revenue."""
    return (r["gross_profit_eur"] + r["lost_revenue_eur"]) / (N_DAYS+1)


def profit_at(r: dict, m: float) -> float:
    """Analytical profit/day at goodwill multiplier m."""
    return cash_profit(r) - m * r["lost_revenue_eur"] / (N_DAYS+1)


def breakeven(r_policy: dict, r_baseline: dict) -> float | None:
    """
    Multiplier at which policy profit crosses baseline profit.
    Returns 0.0 if the policy always beats baseline; None if it never does.
    """
    dense = np.linspace(0, 5, 5000)
    diffs = [profit_at(r_policy, m) - profit_at(r_baseline, m) for m in dense]
    if diffs[0] > 0 and all(d > 0 for d in diffs):
        return 0.0
    for i in range(len(dense) - 1):
        if diffs[i] <= 0 and diffs[i+1] > 0:
            return float(dense[i] + (0 - diffs[i]) / (diffs[i+1] - diffs[i])
                         * (dense[i+1] - dense[i]))
    return None


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict):
    base      = results["baseline"]
    base_cash = cash_profit(base)
    base_gw   = profit_at(base, 1.0)

    print(f"\n{'═'*90}")
    print(f"  STEP 7a — GOODWILL SENSITIVITY SUMMARY")
    print(f"{'═'*90}")
    print(f"  {'Policy':<12} {'Fleet':>6} {'Workers':>8} "
          f"{'GW profit/day':>14} {'vs Base (GW)':>13} "
          f"{'Cash/day':>12} {'vs Base (cash)':>15} "
          f"{'Unmet':>7} {'Breakeven':>10}")
    print(f"  {'-'*88}")

    for policy_key in POLICY_ORDER:
        if policy_key not in results:
            continue
        r      = results[policy_key]
        gw     = profit_at(r, 1.0)
        ca     = cash_profit(r)
        be     = breakeven(r, base)
        be_str = f"{be:.2f}×" if be is not None else "∞ (never)"
        print(f"  {r['label']:<12} {r['fleet_size']:>6} {r.get('n_workers',1):>8} "
              f"€{gw:>12,.0f}  {gw-base_gw:>+12,.0f}  "
              f"€{ca:>10,.0f}  {ca-base_cash:>+14,.0f}  "
              f"{r['unmet_rate']:>6.1%}  {be_str:>10}")
    print(f"\n{'═'*90}\n")


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_comparison(results: dict):
    """
    fig_7_gw_comparison — side-by-side bar chart, goodwill=1.0 (left)
    and cash (right). Reproduced from Step 6 data for convenience.
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
            col  = _COLORS[[k for k,v in OPTIMAL_CFG.items() if v["label"]==lbl][0]]
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f"€{val:,.0f}/day",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=col)
            if lbl != "Baseline":
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 550,
                        f"({sign}€{diff:,.0f})",
                        ha="center", va="bottom", fontsize=8, color=col)
        for bar, p in zip(bars, order):
            ax.text(bar.get_x() + bar.get_width()/2, min(vals) - 500,
                    f"unmet: {results[p]['unmet_rate']:.1%}",
                    ha="center", va="top", fontsize=8, color="gray")
        y_pad = (max(vals) - min(vals)) * 0.15
        ax.set_ylim(min(vals) - y_pad*3, max(vals) + y_pad*4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Full policy comparison at individual optimal configurations\n"
        "(each policy at its own goodwill-optimal fleet and recommended worker count)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_7_gw_comparison.png")
    plt.show(); plt.close(fig)


def plot_sensitivity(results: dict):
    """
    fig_7_gw_sensitivity — profit vs goodwill multiplier, all policies.
    Shows where each relocation policy crosses the baseline.
    """
    base = results["baseline"]
    fig, ax = plt.subplots(figsize=(12, 7))

    base_curve = [profit_at(base, m) for m in MULTIPLIERS]
    ax.plot(MULTIPLIERS, base_curve, color=_COLORS["baseline"],
            linewidth=2.5, label=_LABELS["baseline"], zorder=5)

    for policy_key in ["reactive", "nightly", "proactive"]:
        if policy_key not in results:
            continue
        r     = results[policy_key]
        curve = [profit_at(r, m) for m in MULTIPLIERS]
        be    = breakeven(r, base)

        ax.plot(MULTIPLIERS, curve, color=_COLORS[policy_key], linewidth=2,
                marker="o", markersize=2, label=_LABELS[policy_key], zorder=3)

        if be is not None and 0 <= be <= 5:
            be_profit = profit_at(r, be)
            ax.scatter([be], [be_profit], color=_COLORS[policy_key], s=80,
                       zorder=6, edgecolors="black", linewidths=1)
            ax.annotate(
                f"{_LABELS[policy_key]} breakeven\n{be:.2f}×",
                xy=(be, be_profit),
                xytext=(be + 0.2, be_profit),
                fontsize=8, color=_COLORS[policy_key],
                arrowprops=dict(arrowstyle="->", color=_COLORS[policy_key], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=_COLORS[policy_key], alpha=0.85),
            ).set_zorder(10)

    ax.set_xlabel("Goodwill multiplier", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        "Profit vs goodwill multiplier — all policies at optimal configurations\n"
        "● = breakeven multiplier",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25); ax.set_xlim(0, 5)
    plt.tight_layout()
    _savefig(fig, "fig_7_gw_sensitivity.png")
    plt.show(); plt.close(fig)


def plot_difference_vs_baseline(results: dict):
    """
    fig_7_gw_difference — profit difference vs baseline across multipliers.
    Positive region = policy beats baseline.
    """
    base = results["baseline"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0, color="black", linewidth=2, linestyle="--",
               label="Baseline (reference)", zorder=5)

    for policy_key in ["reactive", "nightly", "proactive"]:
        if policy_key not in results:
            continue
        r     = results[policy_key]
        diffs = [profit_at(r, m) - profit_at(base, m) for m in MULTIPLIERS]
        be    = breakeven(r, base)

        ax.plot(MULTIPLIERS, diffs, color=_COLORS[policy_key], linewidth=2.5,
                marker="o", markersize=2, label=_LABELS[policy_key], zorder=3)
        ax.fill_between(MULTIPLIERS, diffs, 0,
                        where=[d > 0 for d in diffs],
                        color=_COLORS[policy_key], alpha=0.08)
        if be is not None and 0 <= be <= 5:
            ax.axvline(be, color=_COLORS[policy_key],
                       linewidth=1.2, linestyle="--", alpha=0.6)
            ax.text(be + 0.02, -750, f"{_LABELS[policy_key][0]}: {be:.2f}×",
                    fontsize=8, color=_COLORS[policy_key], va="bottom")

    ax.set_xlabel("Goodwill multiplier", fontsize=12)
    ax.set_ylabel("Profit difference vs baseline (€/day)", fontsize=12)
    ax.set_title(
        "Profit advantage over baseline vs goodwill multiplier\n"
        "Dashed verticals = breakeven multiplier.",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25); ax.set_xlim(0, 5)
    plt.tight_layout()
    _savefig(fig, "fig_7_gw_difference.png")
    plt.show(); plt.close(fig)


def plot_breakeven_summary(results: dict):
    """
    fig_7_gw_breakeven — breakeven goodwill multiplier per policy, bar chart.
    """
    base     = results["baseline"]
    policies = [p for p in ["reactive", "nightly", "proactive"] if p in results]
    labels   = [results[p]["label"] for p in policies]
    bes      = [breakeven(results[p], base) for p in policies]
    colors   = [_COLORS[p] for p in policies]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(labels))
    for i, (lbl, be, col) in enumerate(zip(labels, bes, colors)):
        height = min(be, 5.5) if be is not None else 5.5
        ax.bar(i, height, color=col, alpha=0.85, width=0.5)
        if be is None or be > 5:
            ax.text(i, 5.3, "∞\n(never)", ha="center", va="bottom",
                    fontsize=11, color="red", fontweight="bold")
        else:
            ax.text(i, height + 0.05, f"{be:.2f}×",
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color=col)

    ax.axhline(1.0, color="navy", linewidth=1.8, linestyle="--",
               label="Multiplier=1.0 (direct fare only)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Breakeven goodwill multiplier", fontsize=12)
    ax.set_ylim(0, 6.5)
    ax.set_title(
        "Breakeven goodwill multiplier per policy\n"
        "(minimum valuation of unmet trip at which relocation pays for itself)",
        fontsize=12
    )
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, "fig_7_gw_breakeven.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 7a: goodwill sensitivity analysis (analytical, no new simulations)"
    )
    parser.add_argument("--plot", action="store_true",
                        help="(No-op: this step is always plot-only)")
    args = parser.parse_args()

    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 7a — Goodwill sensitivity analysis")
    print(f"  Loading from: {STEP6_RESULTS}")
    print(f"{'═'*60}")

    results = load_step6_results()
    print(f"  Loaded {len(results)} policy results")

    print_summary(results)

    print("  Generating figures...")
    plot_comparison(results)
    plot_sensitivity(results)
    plot_difference_vs_baseline(results)
    plot_breakeven_summary(results)

    print(f"\n  All figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
