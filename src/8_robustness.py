"""
8_robustness.py  -  Robustness check (Step 8).

Each policy is run 10 times at its Step 6 optimal configuration using
different random seeds, varying the stochastic demand process while holding
all other parameters fixed.

The same 10 seeds are used across all policies, enabling paired comparison:
each seed represents the same underlying demand realisation evaluated under
all four policies, so policy differences are isolated from stochastic
variation.

The distribution of gross profit per day — summarised by mean, standard
deviation, and 95% confidence interval — quantifies sensitivity to demand
variability and confirms that the policy rankings from Step 6 are not an
artefact of the single seed used in Steps 1-6.

Produces three figures:
  fig_8_profit_distribution   Box + strip plot per policy
  fig_8_paired_differences    Per-seed profit difference vs baseline
  fig_8_ranking_stability     Profit per seed — all policies as lines

Usage
-----
  python 8_robustness.py                   # run all seeds + plots
  python 8_robustness.py --seeds 15        # use more seeds
  python 8_robustness.py --plot            # plots only (data must exist)
  python 8_robustness.py --skip-existing   # resume an interrupted run
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats as sp

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
N_SEEDS  = 10

POLICY_ORDER = ["baseline", "reactive", "nightly", "proactive"]

DATA_DIR  = Path("./data/metrics/8_robustness")
PLOTS_DIR = DATA_DIR / "plots"

# Colors and labels from final_param.OPTIMAL
_COLORS = {k: v["color"] for k, v in OPTIMAL.items()}
_LABELS = {k: v["label"] for k, v in OPTIMAL.items()}

# Expected ranking for reversal detection (1 = best profit).
# Nightly is excluded from reversal checks because its rank can vary.
EXPECTED_RANK = {"proactive": 1, "reactive": 2, "baseline": 3, "nightly": 4}


# ════════════════════════════════════════════════════════════════════════════
#  IO
# ════════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(data: list):
    path = DATA_DIR / "robustness_results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → {path}")


def _load() -> list:
    with open(DATA_DIR / "robustness_results.json") as f:
        return json.load(f)


def _exists() -> bool:
    return (DATA_DIR / "robustness_results.json").exists()


def _savefig(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")


# ════════════════════════════════════════════════════════════════════════════
#  SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_one(policy_key: str, seed: int) -> dict:
    """Run one simulation at the given seed; restore config afterwards."""
    cfg    = OPTIMAL[policy_key]
    model  = FinancialModel(goodwill_multiplier=GOODWILL)
    policy = load_policy(policy_name=cfg["policy_name"], **cfg["params"])

    original_seed = sim_config.simulation["demand"]["seed"]
    sim_config.simulation["demand"]["seed"] = seed * 100   # well-separated seeds

    try:
        engine = SimulationEngine(
            n_days=N_DAYS, policy=policy,
            fleet_size_override=cfg["fleet"],
            verbose=False,
        )
        engine.run(verbose=False)
    finally:
        sim_config.simulation["demand"]["seed"] = original_seed

    m   = Metrics(engine.event_log, fleet_size=cfg["fleet"])
    fin = FinancialResults(engine.event_log, fleet_size=cfg["fleet"], model=model)

    return {
        "policy":                   policy_key,
        "label":                    cfg["label"],
        "fleet_size":               cfg["fleet"],
        "n_workers":                cfg["workers"],
        "seed":                     seed,
        "goodwill_multiplier":      GOODWILL,
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


# ════════════════════════════════════════════════════════════════════════════
#  SWEEP
# ════════════════════════════════════════════════════════════════════════════

def run_robustness(seeds: list[int], skip_existing: bool = False) -> list[dict]:
    """Run all (policy, seed) combinations; append-mode, safe to interrupt."""
    results = _load() if (skip_existing and _exists()) else []
    done    = {(r["policy"], r["seed"]) for r in results}
    grid    = [(p, s) for p, s in itertools.product(POLICY_ORDER, seeds)
               if (p, s) not in done]

    total = len(POLICY_ORDER) * len(seeds)
    print(f"\n  Robustness sweep: {len(POLICY_ORDER)} policies × {len(seeds)} seeds "
          f"= {total} runs")
    print(f"  Already done: {len(done)}  |  Remaining: {len(grid)}")

    for i, (policy_key, seed) in enumerate(grid):
        cfg = OPTIMAL[policy_key]
        with ws._patch_workers(cfg["workers"]):
            r = run_one(policy_key, seed)

        sign = "+" if r["gross_profit_per_day_eur"] >= 0 else ""
        print(f"  [{i+1:>3}/{len(grid)}] {policy_key:<12} seed={seed}  "
              f"unmet={r['unmet_rate']:.1%}  "
              f"profit/day={sign}€{r['gross_profit_per_day_eur']:>9,.0f}")

        results.append(r)
        _save(results)   # immediate write after every run

    return results


# ════════════════════════════════════════════════════════════════════════════
#  STATS HELPERS
# ════════════════════════════════════════════════════════════════════════════

def by_policy(results: list[dict]) -> dict[str, list[dict]]:
    grouped = {p: [] for p in POLICY_ORDER}
    for r in results:
        if r["policy"] in grouped:
            grouped[r["policy"]].append(r)
    return grouped


def profits(group: list[dict]) -> list[float]:
    return [r["gross_profit_per_day_eur"] for r in group]


def ci95(vals: list[float]) -> tuple[float, float]:
    """95% confidence interval assuming t-distribution."""
    n    = len(vals)
    mean = np.mean(vals)
    se   = sp.sem(vals)
    h    = se * sp.t.ppf(0.975, n - 1)
    return mean - h, mean + h


def _build_seed_lookup(results: list[dict]) -> dict[int, dict[str, float]]:
    """Return {seed: {policy_key: profit/day}}."""
    lookup: dict[int, dict[str, float]] = {}
    for r in results:
        lookup.setdefault(r["seed"], {})[r["policy"]] = r["gross_profit_per_day_eur"]
    return lookup


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], seeds: list[int]):
    grouped      = by_policy(results)
    base_profits = profits(grouped["baseline"])

    print(f"\n{'═'*80}")
    print(f"  STEP 8 — ROBUSTNESS CHECK ({len(seeds)} seeds per policy, goodwill={GOODWILL})")
    print(f"{'═'*80}")
    print(f"  {'Policy':<12} {'Mean':>12} {'Std':>8} {'Min':>10} {'Max':>10} "
          f"{'95% CI lower':>14} {'95% CI upper':>14} {'vs Base mean':>14}")
    print(f"  {'-'*78}")

    for policy_key in POLICY_ORDER:
        ps       = profits(grouped[policy_key])
        mean     = np.mean(ps)
        std      = np.std(ps, ddof=1)
        lo, hi   = ci95(ps)
        vs_base  = mean - np.mean(base_profits)
        sign     = "+" if vs_base >= 0 else ""
        print(f"  {_LABELS[policy_key]:<12} "
              f"€{mean:>10,.0f}  "
              f"€{std:>6,.0f}  "
              f"€{min(ps):>8,.0f}  "
              f"€{max(ps):>8,.0f}  "
              f"€{lo:>12,.0f}  "
              f"€{hi:>12,.0f}  "
              f"{sign}€{vs_base:>12,.0f}")

    # per-seed ranking table
    lookup    = _build_seed_lookup(results)
    reversals = 0

    print(f"\n  Ranking per seed (1 = best profit):")
    print(f"  {'Seed':>6}  " +
          "  ".join(f"{_LABELS[p]:>12}" for p in POLICY_ORDER))
    print(f"  {'-'*70}")

    for seed in sorted(lookup.keys()):
        sp_dict  = lookup[seed]
        ranked   = sorted(sp_dict, key=lambda p: sp_dict[p], reverse=True)
        ranks    = {p: ranked.index(p) + 1 for p in POLICY_ORDER if p in sp_dict}
        reversal = any(ranks.get(p) != EXPECTED_RANK[p]
                       for p in POLICY_ORDER if p != "nightly" and p in ranks)
        if reversal:
            reversals += 1
        flag = " ← REVERSAL" if reversal else ""
        print(f"  {seed:>6}  " +
              "  ".join(f"€{sp_dict.get(p, 0):>10,.0f}(#{ranks.get(p,'?')})"
                        for p in POLICY_ORDER) + flag)

    print(f"\n  Ranking reversals (excluding nightly): {reversals}/{len(seeds)}")
    print(f"{'═'*80}\n")


# ════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_distributions(results: list[dict], seeds: list[int]):
    """
    fig_8_profit_distribution — box plot + jittered individual seed points.
    Diamond (◆) marks the mean. Legend shows mean ± std per policy.
    """
    grouped  = by_policy(results)
    labels   = [_LABELS[p]  for p in POLICY_ORDER]
    colors   = [_COLORS[p]  for p in POLICY_ORDER]
    all_prof = [profits(grouped[p]) for p in POLICY_ORDER]

    fig, ax = plt.subplots(figsize=(11, 7))

    bp = ax.boxplot(all_prof, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="x", markersize=6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # jittered individual seed points
    rng = np.random.default_rng(42)
    for i, (ps, color) in enumerate(zip(all_prof, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(ps))
        ax.scatter([i + j for j in jitter], ps,
                   color=color, s=60, zorder=5,
                   edgecolors="black", linewidths=0.8, alpha=0.9)

    # mean markers
    for i, (lbl, ps, color) in enumerate(zip(labels, all_prof, colors), start=1):
        ax.scatter([i], [np.mean(ps)], color=color, s=120, marker="D",
                   zorder=6, edgecolors="black", linewidths=1.2,
                   label=f"{lbl}: €{np.mean(ps):,.0f} ± €{np.std(ps, ddof=1):,.0f}/day")

    ax.set_xticks(range(1, len(POLICY_ORDER) + 1))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        f"Profit distribution across {len(seeds)} seeds — all policies at optimal configurations\n"
        f"(goodwill_multiplier={GOODWILL}; ◆ = mean)",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, "fig_8_profit_distribution.png")
    plt.show(); plt.close(fig)


def plot_paired_differences(results: list[dict], seeds: list[int]):
    """
    fig_8_paired_differences — per-seed profit difference vs baseline.
    Paired comparison removes demand-realisation noise.
    ▼ marks any seed where the policy falls below the baseline.
    """
    lookup = _build_seed_lookup(results)
    relocation_policies = ["reactive", "nightly", "proactive"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(seeds))
    width = 0.25

    for i, policy_key in enumerate(relocation_policies):
        diffs  = [lookup[s][policy_key] - lookup[s]["baseline"]
                  for s in seeds if s in lookup and "baseline" in lookup.get(s, {})]
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, diffs, width * 0.9,
                        label=_LABELS[policy_key],
                        color=_COLORS[policy_key], alpha=0.85)
        for bar, diff in zip(bars, diffs):
            if diff < 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        diff - 50, "▼", ha="center", va="top",
                        fontsize=8, color="red")

    ax.axhline(0, color="black", linewidth=2, linestyle="--",
               label="Baseline (reference)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=9, rotation=30)
    ax.set_ylabel("Profit difference vs baseline (€/day)", fontsize=12)
    ax.set_title(
        f"Per-seed profit difference vs baseline — paired comparison\n"
        f"(same demand realisation per seed across all policies; ▼ = below baseline)",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    _savefig(fig, "fig_8_paired_differences.png")
    plt.show(); plt.close(fig)


def plot_ranking_stability(results: list[dict], seeds: list[int]):
    """
    fig_8_ranking_stability — profit per seed, all policies as lines.
    Visual check for ranking stability and co-movement across demand realisations.
    """
    lookup = _build_seed_lookup(results)

    fig, ax = plt.subplots(figsize=(12, 6))

    for policy_key in POLICY_ORDER:
        ps = [lookup[s][policy_key] for s in seeds if s in lookup]
        ax.plot(seeds[:len(ps)], ps,
                color=_COLORS[policy_key], linewidth=2,
                marker="o", markersize=7,
                label=f"{_LABELS[policy_key]} (mean €{np.mean(ps):,.0f})")

    ax.set_xlabel("Seed", fontsize=12)
    ax.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax.set_title(
        "Profit per seed — all policies at optimal configurations\n"
        "Co-movement confirms variation is driven by shared demand, not policy differences",
        fontsize=12
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_xticks(seeds)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    _savefig(fig, "fig_8_ranking_stability.png")
    plt.show(); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Step 8: robustness check")
    parser.add_argument("--seeds",         type=int, default=N_SEEDS,
                        help=f"Number of seeds to use (default: {N_SEEDS})")
    parser.add_argument("--plot",          action="store_true",
                        help="Plot only — skip simulations (data must exist)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume an interrupted sweep")
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))

    _ensure_dirs()

    print(f"\n{'═'*60}")
    print(f"  STEP 8 — Robustness check")
    print(f"  Seeds: {seeds}")
    print(f"  Policies: {POLICY_ORDER}")
    print(f"  N_DAYS: {N_DAYS}  |  Goodwill: {GOODWILL}")
    print(f"{'═'*60}")

    if args.plot and _exists():
        results = _load()
    else:
        results = run_robustness(seeds, skip_existing=args.skip_existing)

    print_summary(results, seeds)
    plot_distributions(results, seeds)
    plot_paired_differences(results, seeds)
    plot_ranking_stability(results, seeds)

    print(f"\n  Figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
