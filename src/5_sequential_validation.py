"""
5_sequential_validation.py  -  Sequential optimisation validation (Step 5).

Because fleet size and worker count are optimised sequentially rather than
jointly, the fleet optimum identified in Step 3 (using default worker counts)
may differ from the true optimum at the recommended worker counts from Step 4.

A narrow fleet re-sweep is conducted for the reactive and proactive policies
at their Step 4 recommended worker counts:
  Reactive:   fleet 650-825, step 25, workers = 250
  Proactive:  fleet 575-825, step 25, workers = 150

If the profit-optimal fleet at the recommended worker count matches the Step 3
optimum the sequential approach is validated. If it shifts, the corrected
fleet size is reported and used in subsequent steps.

Produces two figures:
  fig_5_val_combined_single   All four curves on one plot (primary)
  fig_5_val_combined          Side-by-side per policy, dashed=default vs solid=recommended

Usage
-----
  python 5_sequential_validation.py                    # run sweep + plots
  python 5_sequential_validation.py --plot             # plots only (data must exist)
  python 5_sequential_validation.py --skip-existing    # resume interrupted sweep
  python 5_sequential_validation.py --policies reactive
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.simulation.engine import SimulationEngine
from src.analysis.metrics import Metrics
from src.analysis.financials import FinancialResults, FinancialModel
from src.relocation.create_policy import load_policy
from src.optimization import worker_pool_size as ws

from final_param import OPTIMAL


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

N_DAYS   = 30
GOODWILL = 1.0

POLICIES = ["reactive", "proactive"]   # nightly excluded (see docstring)

# Recommended worker counts from Step 4 (final_param.OPTIMAL["workers"])
RECOMMENDED_WORKERS = {
    entry["policy_name"]: entry["workers"]
    for entry in OPTIMAL.values()
    if entry["policy_name"] in POLICIES
}

# Goodwill-optimal parameters from Step 2 (final_param.OPTIMAL["params"])
OPTIMAL_PARAMS = {
    entry["policy_name"]: entry["params"]
    for entry in OPTIMAL.values()
    if entry["policy_name"] in POLICIES
}

# Colors consistent with final_param.OPTIMAL
_COLORS = {
    entry["policy_name"]: entry["color"]
    for entry in OPTIMAL.values()
    if entry["policy_name"] in POLICIES
}
_MARKERS = {"reactive": "o", "proactive": "s"}

# Narrow fleet ranges centred on the Step 3 goodwill-optimal fleet
FLEET_RANGES = {
    "reactive":  range(650, 850, 25),   # Step 3 opt = 750; range 650–825
    "proactive": range(575, 850, 25),   # Step 3 opt = 700; wider 575–825
}

# Step 3 goodwill-optimal fleet sizes, identified with DEFAULT worker counts.
# These are the reference points this step validates against.
# Must be read from the Step 3 JSON output, NOT from final_param.OPTIMAL["fleet"]
# (which already contains the corrected post-validation fleet size).
STEP3_OPT_FLEET = {
    "reactive":  750,   # Step 3 profit-optimal fleet at default workers
    "proactive": 700,   # Step 3 profit-optimal fleet at default workers
}

DATA_DIR  = Path("./data/metrics/5_sequential")
PLOTS_DIR = DATA_DIR / "plots"
STEP3_DIR = Path("./data/metrics/3_goodwill_fleet")   # for comparison curves


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


def _load_step3(policy: str):
    """Load Step 3 goodwill fleet sweep for comparison."""
    path = STEP3_DIR / f"fleet_sweep_{policy}.json"
    if not path.exists():
        print(f"  WARNING: Step 3 data not found at {path}")
        return []
    with open(path) as f:
        return json.load(f)


def _exists(name: str) -> bool:
    return (DATA_DIR / f"{name}.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_one(policy_name: str, fleet_size: int, n_workers: int, **policy_kwargs) -> dict:
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

    return {
        "fleet_size":               fleet_size,
        "policy":                   policy_name,
        "n_workers":                n_workers,
        "goodwill_multiplier":      GOODWILL,
        **policy_kwargs,
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "relocations":              m.total_relocations,
        "relocation_min":           rel["total_min"],
        "relocation_km":            rel["total_km"],
        "lost_revenue_eur":         round(fin.lost_revenue, 2),
        "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
        "gross_profit_eur":         round(fin.gross_profit, 2),
        "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
#  VALIDATION SWEEP
# ════════════════════════════════════════════════════════════════════════════

def sweep_policy(policy_name: str, skip_existing: bool) -> list[dict]:
    """Append-mode fleet sweep at recommended worker count."""
    name        = f"validation_{policy_name}"
    n_workers   = RECOMMENDED_WORKERS[policy_name]
    params      = OPTIMAL_PARAMS[policy_name]
    fleet_range = FLEET_RANGES[policy_name]

    results = _load(name) if _exists(name) else []
    done    = {r["fleet_size"] for r in results}
    grid    = [f for f in fleet_range if f not in done]

    if not grid:
        print(f"  {policy_name}: all {len(results)} fleet sizes done — skipping")
        return results

    print(f"\n  {policy_name.upper()}  workers={n_workers}  "
          f"fleet: {min(fleet_range)}–{max(fleet_range)-1}  "
          f"({len(done)} done, {len(grid)} remaining)")

    with ws._patch_workers(n_workers):
        for fleet_size in grid:
            r    = run_one(policy_name, fleet_size, n_workers, **params)
            sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
            print(f"    fleet={fleet_size:>4}  unmet={r['unmet_rate']:>5.1%}  "
                  f"reloc={r['relocations']:>5}  "
                  f"profit/day={sign}€{r['gross_profit_per_day_eur']:>9,.0f}")
            results.append(r)
            _save(results, name)

    results = sorted(results, key=lambda r: r["fleet_size"])
    _save(results, name)

    opt          = find_optimal(results)
    step3_fleet  = STEP3_OPT_FLEET[policy_name]
    match_str    = ("✓ VALIDATED" if opt["fleet_size"] == step3_fleet
                    else f"⚠ SHIFTED: {step3_fleet} → {opt['fleet_size']}")
    print(f"\n  {policy_name} validation result: {match_str}")
    print(f"  Optimal fleet at workers={n_workers}: {opt['fleet_size']}  "
          f"profit=€{opt['gross_profit_per_day_eur']:,.0f}/day  "
          f"unmet={opt['unmet_rate']:.1%}")
    return results


def run_all_sweeps(policies: list[str], skip_existing: bool) -> dict:
    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 5 — Sequential optimisation validation")
    print(f"  Goodwill multiplier: {GOODWILL}")
    print(f"  Policies: {policies}")
    print(f"{'═'*60}")

    all_results = {}
    for policy in policies:
        all_results[policy] = sweep_policy(policy, skip_existing)
    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_validation: dict):
    print(f"\n{'═'*70}")
    print(f"  STEP 5 — SEQUENTIAL OPTIMISATION VALIDATION SUMMARY")
    print(f"{'═'*70}")

    for policy in POLICIES:
        if policy not in all_validation:
            continue

        val_data        = all_validation[policy]
        val_opt         = find_optimal(val_data)
        step3_fleet     = STEP3_OPT_FLEET[policy]
        rec_w           = RECOMMENDED_WORKERS[policy]

        val_at_step3 = next(
            (r["gross_profit_per_day_eur"] for r in val_data
             if r["fleet_size"] == step3_fleet), None
        )

        print(f"\n  {policy.upper()} (recommended workers={rec_w})")
        print(f"    Step 3 optimal fleet (default workers): {step3_fleet} veh")
        print(f"    Validation optimal:   {val_opt['fleet_size']} veh  "
              f"profit=€{val_opt['gross_profit_per_day_eur']:,.0f}/day")
        if val_at_step3 is not None:
            diff = val_opt["gross_profit_per_day_eur"] - val_at_step3
            print(f"    Profit at Step 3 fleet with rec. workers: "
                  f"€{val_at_step3:,.0f}/day")
            print(f"    Gain from shifting fleet: {diff:+,.0f}/day")

        if val_opt["fleet_size"] == step3_fleet:
            print(f"    Result: ✓ VALIDATED — sequential approach confirmed")
        else:
            print(f"    Result: ⚠ FLEET SHIFTED — use {val_opt['fleet_size']} veh "
                  f"for subsequent steps")

    print(f"\n{'═'*70}\n")


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def find_optimal(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


def _trim_step3(step3_data: list[dict], val_data: list[dict]) -> list[dict]:
    """Restrict Step 3 data to the validation fleet range for clean overlays."""
    fleet_min = min(r["fleet_size"] for r in val_data)
    fleet_max = max(r["fleet_size"] for r in val_data)
    return [r for r in step3_data if fleet_min <= r["fleet_size"] <= fleet_max]


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_combined_single(all_validation: dict, all_step3: dict):
    """
    fig_5_val_combined_single — all four curves on one plot.
    Dashed = Step 3 default workers; solid = validation recommended workers.
    ★ marks the validation optimum per policy; ● marks the Step 3 optimum fleet.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for policy in POLICIES:
        if policy not in all_validation:
            continue

        col      = _COLORS[policy]
        mrk      = _MARKERS[policy]
        rec_w    = RECOMMENDED_WORKERS[policy]
        val_data = sorted(all_validation[policy], key=lambda r: r["fleet_size"])
        step3    = sorted(all_step3.get(policy, []), key=lambda r: r["fleet_size"])
        trimmed  = _trim_step3(step3, val_data)

        val_fleets  = [r["fleet_size"]               for r in val_data]
        val_profits = [r["gross_profit_per_day_eur"]  for r in val_data]
        sb_fleets   = [r["fleet_size"]               for r in trimmed]
        sb_profits  = [r["gross_profit_per_day_eur"]  for r in trimmed]

        # Step 3 curve — dashed
        if sb_fleets:
            def_w = trimmed[0].get("n_workers", "default")
            ax.plot(sb_fleets, sb_profits,
                    color=col, linewidth=1.6, linestyle="--",
                    marker=mrk, markersize=5, alpha=0.5,
                    label=f"{policy.capitalize()} — default workers ({def_w})")

        # Validation curve — solid
        ax.plot(val_fleets, val_profits,
                color=col, linewidth=2.5,
                marker=mrk, markersize=6,
                label=f"{policy.capitalize()} — validation ({rec_w} workers)")

        # Step 3 optimum circle on validation curve
        step3_fleet  = STEP3_OPT_FLEET[policy]
        val_at_step3 = next(
            (r["gross_profit_per_day_eur"] for r in val_data
             if r["fleet_size"] == step3_fleet), None
        )
        if val_at_step3 is not None:
            ax.scatter([step3_fleet], [val_at_step3],
                       color=col, s=100, zorder=5,
                       edgecolors="gray", linewidths=1.5, alpha=0.7)

        # Validation optimum star
        val_opt = find_optimal(val_data)
        ax.scatter([val_opt["fleet_size"]], [val_opt["gross_profit_per_day_eur"]],
                   color=col, s=220, zorder=6,
                   edgecolors="black", linewidths=1.8, marker="*")

        # Annotation
        shifted = val_opt["fleet_size"] != step3_fleet
        if shifted and val_at_step3 is not None:
            diff = val_opt["gross_profit_per_day_eur"] - val_at_step3
            ax.annotate(
                f"{policy.capitalize()}: {step3_fleet}→{val_opt['fleet_size']} veh\n"
                f"(Δ = {diff:+,.0f} €/day)",
                xy=(val_opt["fleet_size"], val_opt["gross_profit_per_day_eur"]),
                xytext=(val_opt["fleet_size"] - 40,
                        val_opt["gross_profit_per_day_eur"] + 120),
                fontsize=8.5, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=1),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, alpha=0.9),
            )
        else:
            ax.annotate(
                f"{policy.capitalize()}: fleet unchanged\n({val_opt['fleet_size']} veh) ✓",
                xy=(val_opt["fleet_size"], val_opt["gross_profit_per_day_eur"]),
                xytext=(val_opt["fleet_size"] + 15,
                        val_opt["gross_profit_per_day_eur"] - 200),
                fontsize=8.5, color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=1),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=col, alpha=0.85),
            )

    # y-limits: auto from data with a fixed floor to show detail
    all_p = [
        r["gross_profit_per_day_eur"]
        for data in list(all_validation.values()) + list(all_step3.values())
        for r in data
        if r["fleet_size"] in range(600, 850)
    ]
    if all_p:
        y_pad = (max(all_p) - min(all_p)) * 0.12
        ax.set_ylim(min(all_p) - y_pad, max(all_p) + y_pad)

    ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        "Sequential optimisation validation — fleet re-sweep at recommended worker counts\n"
        "★ = validation optimum  |  ● = Step 3 optimum fleet (default workers)",
        fontsize=11
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_5_val_combined_single.png")
    plt.show()
    plt.close(fig)


def plot_combined_per_policy(all_validation: dict, all_step3: dict):
    """
    fig_5_val_combined — side-by-side subplots, one per policy.
    Each shows dashed Step 3 curve vs solid validation curve.
    """
    policies = [p for p in POLICIES if p in all_validation]
    fig, axes = plt.subplots(1, len(policies), figsize=(8 * len(policies), 6))
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        col      = _COLORS[policy]
        rec_w    = RECOMMENDED_WORKERS[policy]
        val_data = sorted(all_validation[policy], key=lambda r: r["fleet_size"])
        step3    = sorted(all_step3.get(policy, []), key=lambda r: r["fleet_size"])
        trimmed  = _trim_step3(step3, val_data)

        val_fleets  = [r["fleet_size"]               for r in val_data]
        val_profits = [r["gross_profit_per_day_eur"]  for r in val_data]
        sb_fleets   = [r["fleet_size"]               for r in trimmed]
        sb_profits  = [r["gross_profit_per_day_eur"]  for r in trimmed]

        def_w = trimmed[0].get("n_workers", "default") if trimmed else "default"

        if sb_fleets:
            ax.plot(sb_fleets, sb_profits,
                    color=col, linewidth=1.8, linestyle="--",
                    marker="s", markersize=5, alpha=0.6,
                    label=f"Default workers ({def_w})")

        ax.plot(val_fleets, val_profits,
                color=col, linewidth=2.5,
                marker="o", markersize=6,
                label=f"Validation: {rec_w} workers (recommended)")

        # Step 3 optimum on validation curve
        step3_fleet  = STEP3_OPT_FLEET[policy]
        val_at_step3 = next(
            (r["gross_profit_per_day_eur"] for r in val_data
             if r["fleet_size"] == step3_fleet), None
        )
        if val_at_step3 is not None:
            ax.scatter([step3_fleet], [val_at_step3],
                       color=col, s=120, zorder=5,
                       edgecolors="gray", linewidths=1.5,
                       label=f"Step 3 optimum fleet ({step3_fleet} veh)")

        # Validation optimum star
        val_opt = find_optimal(val_data)
        ax.scatter([val_opt["fleet_size"]], [val_opt["gross_profit_per_day_eur"]],
                   color=col, s=180, zorder=6,
                   edgecolors="black", linewidths=2, marker="*",
                   label=f"Validation optimum ({val_opt['fleet_size']} veh)")

        # Annotation
        if val_opt["fleet_size"] != step3_fleet:
            diff = (val_opt["gross_profit_per_day_eur"] - val_at_step3
                    if val_at_step3 is not None else 0)
            ax.annotate(
                f"Fleet shifts:\n{step3_fleet} → {val_opt['fleet_size']} veh\n"
                f"Δprofit = {diff:+,.0f}/day",
                xy=(val_opt["fleet_size"], val_opt["gross_profit_per_day_eur"]),
                xytext=(val_opt["fleet_size"] - 60,
                        val_opt["gross_profit_per_day_eur"] - 300),
                fontsize=8.5, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9),
            )
        else:
            ax.annotate(
                f"✓ Fleet unchanged\n({val_opt['fleet_size']} veh)",
                xy=(val_opt["fleet_size"], val_opt["gross_profit_per_day_eur"]),
                xytext=(val_opt["fleet_size"] + 20,
                        val_opt["gross_profit_per_day_eur"] - 200),
                fontsize=9, color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen",
                          edgecolor="gray", alpha=0.85),
            )

        # y-limits: auto from local data
        all_profits = val_profits + sb_profits
        if all_profits:
            y_pad = (max(all_profits) - min(all_profits)) * 0.15
            ax.set_ylim(min(all_profits) - y_pad, max(all_profits) + y_pad * 2)

        ax.set_xlabel("Fleet size (vehicles)", fontsize=12)
        ax.set_ylabel("Gross profit per day (€)", fontsize=12)
        ax.set_title(
            f"{policy.capitalize()} policy — sequential optimisation validation\n"
            f"Default workers: {def_w}  →  Recommended workers: {rec_w}",
            fontsize=11
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Sequential optimisation validation: fleet re-sweep at recommended worker counts\n"
        "★ = validation optimum",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_5_val_combined.png")
    plt.show()
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOT ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_plots(all_validation: dict, all_step3: dict):
    print(f"\n{'═'*60}")
    print(f"  Generating Step 5 plots  →  {PLOTS_DIR}")
    print(f"{'═'*60}\n")

    print("  fig_5_val_combined_single ...")
    plot_combined_single(all_validation, all_step3)

    print("  fig_5_val_combined ...")
    plot_combined_per_policy(all_validation, all_step3)

    print(f"\n  All plots saved to {PLOTS_DIR}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: sequential optimisation validation"
    )
    parser.add_argument("--policies", nargs="+",
                        default=POLICIES, choices=POLICIES,
                        help="Policies to validate (default: reactive proactive)")
    parser.add_argument("--plot", action="store_true",
                        help="Plot only — skip sweep (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume an interrupted sweep")
    args = parser.parse_args()

    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 5 — Sequential optimisation validation")
    print(f"  Goodwill multiplier: {GOODWILL}")
    print(f"  Policies: {args.policies}")
    print(f"{'═'*60}")

    # load Step 3 comparison data (optional — plots degrade gracefully if missing)
    all_step3 = {}
    for policy in args.policies:
        data = _load_step3(policy)
        if data:
            all_step3[policy] = data

    # run or load validation sweeps
    all_validation = {}
    if args.plot:
        for policy in args.policies:
            name = f"validation_{policy}"
            if _exists(name):
                all_validation[policy] = _load(name)
            else:
                print(f"  WARNING: {name}.json not found — run without --plot first")
    else:
        all_validation = run_all_sweeps(args.policies, args.skip_existing)

    if all_validation:
        print_summary(all_validation)
        run_plots(all_validation, all_step3)


if __name__ == "__main__":
    main()