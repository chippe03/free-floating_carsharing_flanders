"""
4_worker_sweep.py  -  Worker count optimisation (Step 4).

Sweeps worker count per relocation policy at each policy's Step 3
goodwill-optimal fleet size, using Step 2 goodwill-optimal parameters
and goodwill_multiplier = 1.0.

Worker counts are swept over policy-specific ranges that reflect each
policy's relocation structure:
  Reactive:   25-4000 workers
  Nightly:    25-800  workers
  Proactive:  25-4000 workers

The recommended (final) worker counts are determined manually by
inspecting the diminishing-returns plot (fig_4_W4) and are recorded in
RECOMMENDED below. Running the sweep automatically saves a JSON of the
profit-argmax worker count per policy; the manually chosen RECOMMENDED
counts are also saved separately as optimal_workers.json for downstream
steps.

Produces seven figures:
  fig_4_W1_profit_vs_workers      Profit vs worker count — one subplot per policy
  fig_4_W2_marginal_profit        Marginal profit per additional worker — one subplot per policy
  fig_4_W3_worker_utilisation     Forgotten relocations + peak active — one subplot per policy
  fig_4_W4_diminishing_returns    Annotated profit curve with green/red dots per policy
  fig_4_W_combined_profit         Profit vs workers — thesis figure, recommended count marked
  fig_4_W_forgotten               Forgotten relocations + peak active — thesis figure
  fig_4_W_unmet                   Unmet demand rate vs workers — all policies on one plot

Usage
-----
  python 4_worker_sweep.py                        # run sweep + all plots
  python 4_worker_sweep.py --plot                 # plot only (data must exist)
  python 4_worker_sweep.py --skip-existing        # resume interrupted sweep
  python 4_worker_sweep.py --policies reactive    # sweep one policy only
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

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

# Goodwill-optimal fleet sizes from Step 3 (final_param.OPTIMAL["fleet"])
FLEET_SIZES = {entry["policy_name"]: entry["fleet"] for entry in OPTIMAL.values()}

# Goodwill-optimal parameters from Step 2 (final_param.OPTIMAL["params"])
OPTIMAL_PARAMS = {entry["policy_name"]: entry["params"] for entry in OPTIMAL.values()}

# Worker sweep ranges per policy
WORKER_RANGES = {
    "reactive":  (list(range(25, 825, 25))
                  + list(range(800,  1500,  50))
                  + list(range(1500, 2500, 100))
                  + list(range(2500, 4500, 500))),
    "nightly":   list(range(25, 825, 25)),
    "proactive": (list(range(25, 825, 25))
                  + list(range(800,  1500,  50))
                  + list(range(1500, 2500, 100))
                  + list(range(2500, 4500, 500))),
}

# Manually identified recommended worker counts from fig_4_W4.
# Last clearly-positive green dot before the permanent profit plateau.
#   Reactive:  250  — last green dot before permanent plateau
#   Proactive: 150  — last green dot before permanent plateau
#   Nightly:   25   — profit maximised at minimum; declines monotonically
RECOMMENDED = {"reactive": 250, "nightly": 25, "proactive": 150}

# Goodwill profit of baseline at each policy's goodwill-optimal fleet (from Step 3)
BASELINE_GW = {
    "reactive":  195294,
    "nightly":   194828,
    "proactive": 196059,
}

# Unmet rate of baseline at each policy's goodwill-optimal fleet (Step 3 reference)
BASELINE_UNMET = {"reactive": 2.3, "nightly": 3.9, "proactive": 2.2}

DATA_DIR  = Path("./data/metrics/4_workers")
PLOTS_DIR = DATA_DIR / "plots"

# Colors consistent with final_param.py OPTIMAL
_COLORS = {entry["policy_name"]: entry["color"] for entry in OPTIMAL.values()}
# Baseline color for reference lines
_BASELINE_COLOR = OPTIMAL["baseline"]["color"]   # "#00FF99"

POLICIES = ["reactive", "nightly", "proactive"]


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
#  SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_one(policy_name: str, fleet_size: int, n_workers: int, **policy_kwargs) -> dict:
    """Run one simulation at goodwill=1.0; return result dict."""
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
        "policy":                   policy_name,
        "fleet_size":               fleet_size,
        "n_workers":                n_workers,
        "goodwill_multiplier":      GOODWILL,
        **policy_kwargs,
        # service quality
        "unmet_rate":               round(m.unmet_rate, 4),
        "total_served":             m.total_served,
        "total_unmet":              m.total_unmet,
        "relocations":              m.total_relocations,
        "relocation_min":           rel["total_min"],
        "relocation_km":            rel["total_km"],
        # worker utilisation
        "peak_workers":             wor["peak_workers_active"],
        "relocations_queued":       wor["relocations_queued"],
        "relocations_forgotten":    wor["relocations_forgotten"],
        # financial
        "total_revenue_eur":        round(fin.total_revenue, 2),
        "lost_revenue_eur":         round(fin.lost_revenue, 2),
        "vehicle_cost_eur":         round(fin.total_vehicle_cost, 2),
        "relocation_cost_eur":      round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":   round(fin.total_repositioning_cost, 2),
        "total_cost_eur":           round(fin.total_cost, 2),
        "gross_profit_eur":         round(fin.gross_profit, 2),
        "gross_profit_per_day_eur": round(fin.gross_profit_per_day, 2),
    }


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP
# ════════════════════════════════════════════════════════════════════════════

def sweep_policy(policy_name: str, skip_existing: bool) -> list[dict]:
    """Append-mode sweep — safe to interrupt and resume."""
    name         = f"workers_{policy_name}"
    fleet_size   = FLEET_SIZES[policy_name]
    params       = OPTIMAL_PARAMS[policy_name]
    worker_range = WORKER_RANGES[policy_name]

    results = _load(name) if _exists(name) else []
    done    = {r["n_workers"] for r in results}
    grid    = [w for w in worker_range if w not in done]

    if not grid:
        print(f"  {policy_name}: all {len(results)} worker counts done — skipping")
        return results

    print(f"\n  {policy_name.upper()}  fleet={fleet_size}  "
          f"workers: {min(grid)}–{max(grid)}  "
          f"({len(done)} done, {len(grid)} remaining)")
    print(f"  Params: {params}")

    for n_workers in grid:
        with ws._patch_workers(n_workers):
            r = run_one(policy_name, fleet_size, n_workers, **params)

        sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
        print(f"    workers={n_workers:>4}  "
              f"unmet={r['unmet_rate']:>5.1%}  "
              f"reloc={r['relocations']:>5}  "
              f"forgotten={r['relocations_forgotten']:>3}  "
              f"profit/day={sign}€{r['gross_profit_per_day_eur']:>9,.0f}")

        results.append(r)
        _save(results, name)

    results = sorted(results, key=lambda r: r["n_workers"])
    _save(results, name)

    opt = find_optimal(results)
    print(f"\n  ✓ {policy_name} profit-optimal: {opt['n_workers']} workers  "
          f"€{opt['gross_profit_per_day_eur']:,.0f}/day  "
          f"unmet={opt['unmet_rate']:.1%}")
    return results


def run_all_sweeps(policies: list[str], skip_existing: bool) -> dict[str, list[dict]]:
    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 4 — Worker count optimisation (goodwill={GOODWILL})")
    print(f"  Policies: {policies}")
    print(f"{'═'*60}")
    t0 = time.time()

    all_results = {}
    for policy in policies:
        all_results[policy] = sweep_policy(policy, skip_existing)

    elapsed = int(time.time() - t0)
    print(f"\n  Sweep complete in {elapsed//60}m {elapsed%60}s")
    return all_results


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def find_optimal(results: list[dict]) -> dict:
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


def find_knee(results: list[dict]) -> dict:
    """First worker count where adding more workers changes profit < €10/day."""
    sorted_r = sorted(results, key=lambda r: r["n_workers"])
    for i in range(1, len(sorted_r)):
        delta = (sorted_r[i]["gross_profit_per_day_eur"]
                 - sorted_r[i-1]["gross_profit_per_day_eur"])
        if abs(delta) < 10:
            return sorted_r[i-1]
    return sorted_r[-1]


def get_entry(data: list[dict], n_workers: int) -> dict:
    return next(r for r in data if r["n_workers"] == n_workers)


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: dict):
    print(f"\n{'═'*90}")
    print(f"  STEP 4 WORKER OPTIMISATION SUMMARY (goodwill={GOODWILL})")
    print(f"{'═'*90}")

    # automatic profit-optimal
    print(f"\n  Profit-argmax (automatic):")
    print(f"  {'Policy':<12} {'Opt workers':>12} {'Profit/day':>12} "
          f"{'Unmet':>7} {'Reloc':>7} {'Forgotten':>10}")
    print(f"  {'-'*65}")
    for policy, results in all_results.items():
        opt = find_optimal(results)
        print(f"  {policy:<12} {opt['n_workers']:>12} "
              f"€{opt['gross_profit_per_day_eur']:>10,.0f}  "
              f"{opt['unmet_rate']:>6.1%}  "
              f"{opt['relocations']:>6}  "
              f"{opt['relocations_forgotten']:>9}")

    # manually recommended
    print(f"\n  Recommended (manual, from fig_4_W4 diminishing-returns plot):")
    print(f"  {'Policy':<12} {'Rec.workers':>12} {'Profit/day':>12} {'vs Baseline':>12} "
          f"{'Unmet':>7} {'Forgotten':>10} {'Peak active':>12} "
          f"{'Reloc €/day':>12} {'Repos €/day':>12}")
    print(f"  {'-'*105}")

    for policy in POLICIES:
        if policy not in all_results:
            continue
        data  = all_results[policy]
        rec_w = RECOMMENDED[policy]
        r     = get_entry(data, rec_w)
        base  = BASELINE_GW[policy]
        diff  = r["gross_profit_per_day_eur"] - base

        print(f"  {policy:<12} {rec_w:>12} "
              f"€{r['gross_profit_per_day_eur']:>10,.0f}  "
              f"{diff:>+11,.0f}  "
              f"{r['unmet_rate']:>6.1%}  "
              f"{r['relocations_forgotten']:>9}  "
              f"{r['peak_workers']:>11}  "
              f"€{r['relocation_cost_eur']/(N_DAYS+1):>10,.0f}  "
              f"€{r['repositioning_cost_eur']/(N_DAYS+1):>10,.0f}")

    print(f"\n  Baseline reference (goodwill=1.0):")
    for policy in POLICIES:
        if policy in all_results:
            print(f"  {policy:<12} fleet={FLEET_SIZES[policy]}  "
                  f"baseline=€{BASELINE_GW[policy]:,}/day")
    print(f"\n{'═'*90}\n")

    # save both sets of worker counts
    auto_optimal = {p: find_optimal(r)["n_workers"] for p, r in all_results.items()}
    _save(auto_optimal,  "optimal_workers_auto")
    _save(RECOMMENDED,   "optimal_workers")
    print(f"  Recommended worker counts → {DATA_DIR / 'optimal_workers.json'}")
    print(f"  Profit-argmax counts      → {DATA_DIR / 'optimal_workers_auto.json'}")


# ════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PLOTS  (fig_4_W1 – fig_4_W4)
# ════════════════════════════════════════════════════════════════════════════

def plot_profit_vs_workers(all_results: dict):
    """fig_4_W1 — profit vs worker count, one subplot per policy."""
    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(6 * len(all_results), 6),
                              sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (policy, results) in zip(axes, all_results.items()):
        sorted_r = sorted(results, key=lambda r: r["n_workers"])
        workers  = [r["n_workers"]               for r in sorted_r]
        profits  = [r["gross_profit_per_day_eur"] for r in sorted_r]
        opt      = find_optimal(results)
        knee     = find_knee(results)

        ax.plot(workers, profits, color=_COLORS[policy],
                linewidth=2.5, marker="o", markersize=5)
        ax.axvline(opt["n_workers"],  color=_COLORS[policy], linestyle="--",
                   linewidth=1.5, alpha=0.7,
                   label=f"Profit optimum: {opt['n_workers']} workers")
        ax.axvline(knee["n_workers"], color="gray", linestyle=":",
                   linewidth=1.5, alpha=0.7,
                   label=f"Diminishing returns: {knee['n_workers']} workers")
        ax.annotate(
            f"opt: {opt['n_workers']} workers\n€{opt['gross_profit_per_day_eur']:,.0f}/day",
            xy=(opt["n_workers"], opt["gross_profit_per_day_eur"]),
            xytext=(opt["n_workers"] + max(workers) * 0.05,
                    opt["gross_profit_per_day_eur"]),
            fontsize=8.5, color=_COLORS[policy],
            arrowprops=dict(arrowstyle="->", color=_COLORS[policy]),
        )
        ax.set_xlabel("Worker count", fontsize=11)
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(f"{policy.capitalize()} policy\n"
                     f"(fleet={FLEET_SIZES[policy]}, goodwill=1.0)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Worker count vs gross profit per day (goodwill=1.0)\n"
                 "at goodwill-optimal fleet size and parameters", fontsize=12)
    plt.tight_layout()
    _savefig(fig, "fig_4_W1_profit_vs_workers.png")
    plt.show(); plt.close(fig)


def plot_marginal_profit(all_results: dict):
    """fig_4_W2 — marginal profit per additional worker, one subplot per policy."""
    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(6 * len(all_results), 5),
                              sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (policy, results) in zip(axes, all_results.items()):
        sorted_r = sorted(results, key=lambda r: r["n_workers"])
        workers  = [r["n_workers"]               for r in sorted_r]
        profits  = [r["gross_profit_per_day_eur"] for r in sorted_r]

        step     = workers[1] - workers[0] if len(workers) > 1 else 1
        marginal = [(profits[i] - profits[i-1]) / step for i in range(1, len(profits))]
        mid_w    = [(workers[i] + workers[i-1]) / 2    for i in range(1, len(workers))]

        colors = ["#2ca02c" if m > 0 else "#d62728" for m in marginal]
        ax.bar(mid_w, marginal, width=step * 0.8, color=colors, alpha=0.85)
        ax.axhline(0,  color="black", linewidth=1.5)
        ax.axhline(10, color="gray",  linestyle=":", linewidth=1,
                   label="€10/day threshold")

        ax.set_xlabel("Worker count (midpoint)", fontsize=11)
        ax.set_ylabel("Marginal profit per additional worker (€/day)", fontsize=11)
        ax.set_title(f"{policy.capitalize()} — marginal profit per worker", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Marginal gross profit per additional worker (goodwill=1.0)\n"
                 "Green = positive; red = negative; dotted = diminishing returns threshold",
                 fontsize=12)
    plt.tight_layout()
    _savefig(fig, "fig_4_W2_marginal_profit.png")
    plt.show(); plt.close(fig)


def plot_worker_utilisation(all_results: dict):
    """fig_4_W3 — forgotten relocations + peak active workers vs worker count."""
    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(6 * len(all_results), 5),
                              sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (policy, results) in zip(axes, all_results.items()):
        sorted_r  = sorted(results, key=lambda r: r["n_workers"])
        workers   = [r["n_workers"]             for r in sorted_r]
        forgotten = [r["relocations_forgotten"] for r in sorted_r]
        peak      = [r["peak_workers"]          for r in sorted_r]

        ax2 = ax.twinx()
        l1, = ax.plot(workers, forgotten, color=_COLORS[policy],
                      linewidth=2, marker="o", markersize=5,
                      label="Relocations forgotten (capacity exceeded)")
        l2, = ax2.plot(workers, peak, color="gray",
                       linewidth=1.5, marker="s", markersize=4,
                       linestyle="--", label="Peak simultaneous active workers")

        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("Worker count", fontsize=11)
        ax.set_ylabel("Relocations forgotten", fontsize=11, color=_COLORS[policy])
        ax2.set_ylabel("Peak active workers", fontsize=11, color="gray")
        ax.tick_params(axis="y", labelcolor=_COLORS[policy])
        ax2.tick_params(axis="y", labelcolor="gray")
        ax.set_title(f"{policy.capitalize()} — worker utilisation", fontsize=11)
        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Worker utilisation across worker counts (goodwill=1.0)\n"
                 "Zero forgotten relocations = workers no longer a bottleneck",
                 fontsize=12)
    plt.tight_layout()
    _savefig(fig, "fig_4_W3_worker_utilisation.png")
    plt.show(); plt.close(fig)


def plot_diminishing_returns(all_results: dict):
    """
    fig_4_W4 — profit curve with dots coloured by marginal return.
    Green dot = marginal profit ≥ €10/worker/day (worth adding).
    Red dot   = marginal profit < €10/worker/day (diminishing returns).
    Primary diagnostic for choosing RECOMMENDED worker counts manually.
    """
    THRESHOLD = 10   # €/day per additional worker

    fig, axes = plt.subplots(1, len(all_results),
                              figsize=(7 * len(all_results), 6),
                              sharey=False)
    if len(all_results) == 1:
        axes = [axes]

    for ax, (policy, results) in zip(axes, all_results.items()):
        sorted_r = sorted(results, key=lambda r: r["n_workers"])
        workers  = [r["n_workers"]               for r in sorted_r]
        profits  = [r["gross_profit_per_day_eur"] for r in sorted_r]

        steps     = [workers[i] - workers[i-1]           for i in range(1, len(workers))]
        marginals = [(profits[i] - profits[i-1]) / steps[i-1]
                     for i in range(1, len(profits))]

        below = [workers[i+1] for i, m in enumerate(marginals) if abs(m) < THRESHOLD]
        above = [workers[i+1] for i, m in enumerate(marginals) if abs(m) >= THRESHOLD]

        ax.plot(workers, profits, color=_COLORS[policy], linewidth=2.5,
                marker="o", markersize=4, zorder=3)

        if below:
            ax.axvspan(min(below), max(workers), alpha=0.08, color=_COLORS[policy],
                       label=f"Marginal < €{THRESHOLD}/worker/day")
            y_range = max(profits) - min(profits)
            for w in below:
                ax.axvline(w, color=_COLORS[policy],
                           linewidth=0.6, alpha=0.35, linestyle=":")
            first_below = min(below)
            profit_at_first = profits[workers.index(first_below)]
            ax.axvline(first_below, color=_COLORS[policy],
                       linewidth=2, alpha=0.9, linestyle="--",
                       label=f"First < €{THRESHOLD}/worker: {first_below}")
            ax.annotate(
                f"{first_below} workers\n(marginal < €{THRESHOLD}/day)",
                xy=(first_below, profit_at_first),
                xytext=(first_below + max(workers) * 0.04,
                        profit_at_first - y_range * 0.15),
                fontsize=8, color=_COLORS[policy],
                arrowprops=dict(arrowstyle="->", color=_COLORS[policy], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=_COLORS[policy], alpha=0.85),
            )

        peak_actives = [r["peak_workers"] for r in sorted_r]
        ax.axvline(max(peak_actives), color="gray", linewidth=1.5,
                   linestyle="-.", alpha=0.8,
                   label=f"Max peak simultaneous active: {max(peak_actives)}")

        zero_forg = [r for r in sorted_r if r["relocations_forgotten"] == 0]
        if zero_forg:
            first_zero_w = min(zero_forg, key=lambda r: r["n_workers"])["n_workers"]
            ax.axvline(first_zero_w, color="black", linewidth=1.5,
                       linestyle="--", alpha=0.6,
                       label=f"Zero forgotten: {first_zero_w} workers")

        # colour dots by marginal
        dot_colors = [_COLORS[policy]]   # first point: no marginal
        for m in marginals:
            dot_colors.append("#2ca02c" if abs(m) >= THRESHOLD else "#d62728")
        ax.scatter(workers, profits, c=dot_colors, s=30, zorder=4,
                   edgecolors="white", linewidths=0.5)

        ax.set_xlabel("Worker count", fontsize=11)
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(
            f"{policy.capitalize()} policy\n"
            f"(fleet={FLEET_SIZES[policy]}, goodwill=1.0)\n"
            f"Green dots = marginal ≥ €{THRESHOLD}/worker  |  "
            f"Red dots = marginal < €{THRESHOLD}/worker",
            fontsize=10
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Worker count vs profit — diminishing returns analysis\n"
        f"Red dots: marginal profit per additional worker < €{THRESHOLD}/day "
        f"(shaded = low-return zone)",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_4_W4_diminishing_returns.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  THESIS FIGURES  (fig_4_W_combined_profit / _forgotten / _unmet)
# ════════════════════════════════════════════════════════════════════════════

def plot_combined_profit(all_results: dict):
    """
    fig_4_W_combined_profit — three subplots, one per policy.
    Marks the manually recommended worker count and the baseline reference.
    Primary thesis figure for Section 4.6.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, policy in zip(axes, POLICIES):
        if policy not in all_results:
            ax.set_visible(False)
            continue

        data     = all_results[policy]
        sorted_r = sorted(data, key=lambda r: r["n_workers"])
        workers  = [r["n_workers"]               for r in sorted_r]
        profits  = [r["gross_profit_per_day_eur"] for r in sorted_r]
        rec_w    = RECOMMENDED[policy]
        baseline = BASELINE_GW[policy]

        ax.plot(workers, profits, color=_COLORS[policy],
                linewidth=2, marker="o", markersize=3, zorder=3)

        rec_entry  = get_entry(data, rec_w)
        rec_profit = rec_entry["gross_profit_per_day_eur"]

        ax.axvline(rec_w, color=_COLORS[policy], linewidth=2,
                   linestyle="--", alpha=0.9, zorder=4,
                   label=f"Recommended: {rec_w} workers")
        ax.scatter([rec_w], [rec_profit], color=_COLORS[policy], s=120, zorder=5,
                   edgecolors="black", linewidths=1.2)
        ax.annotate(
            f"{rec_w} workers\n€{rec_profit:,.0f}/day",
            xy=(rec_w, rec_profit),
            xytext=(rec_w + max(workers) * 0.05, rec_profit),
            fontsize=8.5, color=_COLORS[policy],
            arrowprops=dict(arrowstyle="->", color=_COLORS[policy], lw=1),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=_COLORS[policy], alpha=0.85),
            zorder=6,
        )

        ax.axhline(baseline, color=_BASELINE_COLOR, linewidth=1.2,
                   linestyle=":", alpha=0.8,
                   label=f"Baseline €{baseline:,}/day")

        y_min   = min(min(profits), baseline)
        y_max   = max(max(profits), baseline)
        y_range = y_max - y_min
        y_pad   = max(y_range * 0.15, 500)
        ax.set_ylim(y_min - y_pad, y_max + y_pad * 2)

        ax.set_xlabel("Worker count", fontsize=11)
        ax.set_ylabel("Gross profit per day (€)", fontsize=11)
        ax.set_title(f"{policy.capitalize()} policy (fleet={FLEET_SIZES[policy]})",
                     fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend(fontsize=9,
                  loc="lower right" if policy != "nightly" else "upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Worker count vs gross profit per day (goodwill_multiplier=1.0)",
                 fontsize=12)
    plt.tight_layout()
    _savefig(fig, "fig_4_W_combined_profit.png")
    plt.show(); plt.close(fig)


def plot_forgotten(all_results: dict):
    """
    fig_4_W_forgotten — forgotten relocations + peak active workers.
    Reactive and proactive only (nightly excluded: not discussed in thesis).
    """
    policies_to_plot = [p for p in ["reactive", "proactive"] if p in all_results]
    fig, axes = plt.subplots(1, len(policies_to_plot), figsize=(13, 5))
    if len(policies_to_plot) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies_to_plot):
        data      = all_results[policy]
        sorted_r  = sorted(data, key=lambda r: r["n_workers"])
        workers   = [r["n_workers"]             for r in sorted_r]
        forgotten = [r["relocations_forgotten"] for r in sorted_r]
        peak      = [r["peak_workers"]          for r in sorted_r]
        rec_w     = RECOMMENDED[policy]

        ax2 = ax.twinx()
        l1, = ax.plot(workers, forgotten, color=_COLORS[policy],
                      linewidth=2, marker="o", markersize=4,
                      label="Relocations forgotten")
        l2, = ax2.plot(workers, peak, color="gray",
                       linewidth=1.5, marker="s", markersize=3,
                       linestyle="--", label="Peak simultaneous active workers")

        ax.axvline(rec_w, color=_COLORS[policy], linewidth=2,
                   linestyle="--", alpha=0.8, label=f"Recommended: {rec_w}")

        ax.set_xlabel("Worker count", fontsize=11)
        ax.set_ylabel("Relocations forgotten", fontsize=11, color=_COLORS[policy])
        ax2.set_ylabel("Peak simultaneous active workers", fontsize=10, color="gray")
        ax.tick_params(axis="y", labelcolor=_COLORS[policy])
        ax2.tick_params(axis="y", labelcolor="gray")
        ax.set_title(f"{policy.capitalize()}", fontsize=11)
        ax.set_ylim(bottom=0)
        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Forgotten relocations and peak simultaneous workers vs worker pool size\n"
        "Dashed vertical = recommended worker count",
        fontsize=12
    )
    plt.tight_layout()
    _savefig(fig, "fig_4_W_forgotten.png")
    plt.show(); plt.close(fig)


def plot_unmet(all_results: dict, x_max: int = 600):
    """
    fig_4_W_unmet — unmet demand rate vs worker count, all policies on one plot.
    x_max caps the x-axis to focus on the region where unmet is still declining.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for policy in POLICIES:
        if policy not in all_results:
            continue
        data    = all_results[policy]
        sorted_r = sorted(data, key=lambda r: r["n_workers"])
        workers  = [r["n_workers"]        for r in sorted_r]
        unmet    = [r["unmet_rate"] * 100 for r in sorted_r]
        rec_w    = RECOMMENDED[policy]

        ws_plot = [w for w in workers if w <= x_max]
        um_plot = [u for w, u in zip(workers, unmet) if w <= x_max]

        ax.plot(ws_plot, um_plot, color=_COLORS[policy], linewidth=2,
                marker="o", markersize=4, label=policy.capitalize(), zorder=3)

        if rec_w <= x_max:
            rec_unmet = get_entry(data, rec_w)["unmet_rate"] * 100
            ax.scatter([rec_w], [rec_unmet], color=_COLORS[policy], s=140,
                       zorder=5, edgecolors="black", linewidths=1.5)
            ax.annotate(
                f"{rec_w} workers\n{rec_unmet:.1f}%",
                xy=(rec_w, rec_unmet),
                xytext=(rec_w + x_max * 0.04, rec_unmet + 0.12),
                fontsize=8.5, color=_COLORS[policy],
                arrowprops=dict(arrowstyle="->", color=_COLORS[policy], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=_COLORS[policy], alpha=0.85),
            )

    # dotted horizontal reference lines at baseline unmet rates
    for policy, u in BASELINE_UNMET.items():
        if policy in all_results:
            ax.axhline(u, color=_COLORS[policy], linewidth=0.9,
                       linestyle=":", alpha=0.6,
                       label=f"{policy.capitalize()} baseline ({u}%)")

    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=1.5)
    ax.set_xlabel("Worker count", fontsize=12)
    ax.set_ylabel("Unmet demand rate (%)", fontsize=12)
    ax.set_title(
        "Unmet demand rate vs worker pool size (goodwill_multiplier=1.0)\n"
        "● = recommended worker count",
        fontsize=11
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_4_W_unmet.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  PLOT ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_plots(all_results: dict):
    print(f"\n{'═'*60}")
    print(f"  Generating Step 4 plots  →  {PLOTS_DIR}")
    print(f"{'═'*60}\n")

    print("  fig_4_W1_profit_vs_workers ...")
    plot_profit_vs_workers(all_results)
    print("  fig_4_W2_marginal_profit ...")
    plot_marginal_profit(all_results)
    print("  fig_4_W3_worker_utilisation ...")
    plot_worker_utilisation(all_results)
    print("  fig_4_W4_diminishing_returns ...")
    plot_diminishing_returns(all_results)
    print("  fig_4_W_combined_profit ...")
    plot_combined_profit(all_results)
    print("  fig_4_W_forgotten ...")
    plot_forgotten(all_results)
    print("  fig_4_W_unmet ...")
    plot_unmet(all_results)

    print(f"\n  All plots saved to {PLOTS_DIR}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 4: worker count optimisation")
    parser.add_argument("--plot", action="store_true",
                        help="Plot only — skip sweep (data must already exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume an interrupted sweep")
    parser.add_argument("--policies", nargs="+",
                        default=POLICIES,
                        choices=POLICIES,
                        help="Which policies to sweep (default: all three)")
    args = parser.parse_args()

    _ensure_dirs()

    if args.plot:
        print("Loading existing worker sweep data ...")
        all_results = {}
        for policy in args.policies:
            name = f"workers_{policy}"
            if _exists(name):
                all_results[policy] = sorted(_load(name), key=lambda r: r["n_workers"])
                print(f"  {policy}: {len(all_results[policy])} data points, "
                      f"workers {all_results[policy][0]['n_workers']}–"
                      f"{all_results[policy][-1]['n_workers']}")
            else:
                print(f"  WARNING: {name}.json not found — run without --plot first")
    else:
        all_results = run_all_sweeps(args.policies, args.skip_existing)

    if all_results:
        print_summary(all_results)
        run_plots(all_results)


if __name__ == "__main__":
    main()
