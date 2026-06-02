"""
Policy parameter sweep: profit maximising.

For a fixed fleet size, sweeps the tunable parameters of each relocation
policy and finds the combination that maximises gross profit per day.


Sweep grids
-----------
Reactive:   alpha
Proactive:  planning_period_minutes × lookahead_period_minutes × time_step_minutes
Nightly:    trigger_hour


Main entry points
-----------------
run_reactive_sweep(...)    -> list[dict]
run_proactive_sweep(...)   -> list[dict]
run_nightly_sweep(...)     -> list[dict]
find_optimal(results)      -> dict         params with highest profit/day

plot_profit_by_param(...)                  1-D sensitivity per parameter
plot_proactive_heatmap(...)                2-D heatmap for proactive grid
plot_nightly_hourly(...)                   bar chart over trigger hours
plot_radar(...)                            multi-metric fingerprint per policy
plot_all(...)                              render everything at once
"""

import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.simulation.engine import SimulationEngine
from src.analysis.financials import FinancialResults, FinancialModel
from src.analysis.metrics import Metrics
from src.relocation.create_policy import load_policy
from src import config


def _build_result(
    event_log:   list[dict],
    policy_name: str,
    fleet_size:  int,
    model:       FinancialModel,
    **params,
) -> dict:
    """
    Compute operational + financial KPIs from a completed simulation run
    and return a flat result dict that includes all param values.
    """
    m   = Metrics(engine_log := event_log,
                  label=f"{policy_name}", fleet_size=fleet_size)
    fin = FinancialResults(event_log, fleet_size=fleet_size, model=model)
    rel = m.relocation_stats()

    return {
        # identity
        "policy":     policy_name,
        "fleet_size": fleet_size,
        **params,                        # all swept parameter values stored flat

        # operational
        "unmet_rate":             round(m.unmet_rate, 4),
        "unmet_total":            m.total_unmet,
        "total_requests":         m.total_requests,
        "total_served":           m.total_served,
        "relocations":            m.total_relocations,
        "relocation_min":         rel["total_min"],
        "relocation_km":          rel["total_km"],

        # financial: revenue
        "total_revenue_eur":          round(fin.total_revenue, 2),
        "lost_revenue_eur":           round(fin.lost_revenue, 2),
        "avg_revenue_per_trip_eur":   round(fin.avg_revenue_per_trip, 2),

        # financial: costs
        "vehicle_cost_eur":           round(fin.total_vehicle_cost, 2),
        "fuel_cost_eur":              round(fin.total_fuel_cost, 2),
        "relocation_cost_eur":        round(fin.total_relocation_cost, 2),
        "repositioning_cost_eur":     round(fin.total_repositioning_cost, 2),
        "total_cost_eur":             round(fin.total_cost, 2),

        # financial: profit (PRIMARY objective)
        "gross_profit_eur":           round(fin.gross_profit, 2),
        "gross_profit_per_day_eur":   round(fin.gross_profit_per_day, 2),
        "cost_per_trip_eur":          round(fin.cost_per_trip, 2),
    }


def _run_single(
    policy_name:  str,
    fleet_size:   int,
    n_days:       int,
    model:        FinancialModel,
    policy_kwargs: dict,
    param_label:  str,
) -> dict:
    """Build policy, run engine, return result dict."""
    policy = load_policy(policy_name=policy_name, **policy_kwargs)
    engine = SimulationEngine(
        n_days              = n_days,
        policy              = policy,
        fleet_size_override = fleet_size,
        verbose             = False,
    )
    engine.run(verbose=False)
    result = _build_result(
        event_log   = engine.event_log,
        policy_name = policy_name,
        fleet_size  = fleet_size,
        model       = model,
        **policy_kwargs,
    )
    profit_sign = "+" if result["gross_profit_per_day_eur"] >= 0 else ""
    print(
        f"  {param_label:<55}  "
        f"unmet={result['unmet_rate']:>5.1%}  "
        f"profit/day={profit_sign}{result['gross_profit_per_day_eur']:>8.0f}€"
    )
    return result


# ------------------------------------------------------------------ #
#  Reactive sweep (alpha)                                            #
# ------------------------------------------------------------------ #

def run_reactive_sweep(
    fleet_size:             int,
    n_days:                 int   = None,
    alpha_values:           list  = None,
    check_interval_values:  list  = None,
) -> list[dict]:
    """
    Sweep `alpha` for the reactive policy.
    alpha controls min-vehicle thresholds: threshold[city] = max(1, round(alpha * demand_weight)).
    Higher alpha → more aggressive rebalancing → fewer unmet trips but higher relocation cost.
    """
    model = FinancialModel()

    # enforce alpha ceiling: sum(thresholds) must not exceed fleet_size
    total_weight = sum(
        c["demand_weight"] for c in config.cities.values()
        if c.get("demand_weight") is not None
    )
    max_alpha = int(fleet_size / total_weight) if total_weight > 0 else float("inf")

    raw_alphas = alpha_values or [100, 200, 300, 400, 500, 600]
    alphas     = [a for a in raw_alphas if a <= max_alpha]
    if not alphas:
        raise ValueError(
            f"All alpha values {raw_alphas} exceed max_alpha={max_alpha} "
            f"for fleet_size={fleet_size} (total_weight={total_weight:.2f})"
        )
    if len(alphas) < len(raw_alphas):
        clipped = [a for a in raw_alphas if a > max_alpha]
        print(f"  ⚠ Alpha ceiling: clipped {clipped} (max_alpha={max_alpha})")

    check_interval_values = check_interval_values or [30, 60]

    print(f"\nReactive parameter sweep  -  fleet={fleet_size}")
    print(f"Sweeping alpha ∈ {alphas}\n")

    grid = [
        (a, c)
        for a, c in itertools.product(alphas, check_interval_values)
    ]

    results = []
    for alpha, interval in grid:
        kwargs = dict(
            alpha                  = alpha,
            check_interval_minutes = interval,
        )

        result = _run_single(
            policy_name   = "reactive",
            fleet_size    = fleet_size,
            n_days        = n_days,
            model         = model,
            policy_kwargs = kwargs,
            param_label   = f"alpha={alpha:<4}  check_interval={interval}",
        )
        results.append(result)

    _print_optimal(results, ["alpha"])
    return results


# ------------------------------------------------------------------ #
#  Proactive sweep (planning × lookahead × time_step)                #
# ------------------------------------------------------------------ #

def run_proactive_sweep(
    fleet_size:                  int,
    n_days:                      int  = None,
    planning_period_values:      list = None,
    lookahead_period_values:     list = None,
    check_interval_values:       list = None,
    history_log_path:            str  = None,
) -> list[dict]:
    """
    Grid search over planning_period_minutes × lookahead_period_minutes × time_step_minutes.

    planning_period_minutes:  window over which the MILP plans relocations (T)
    lookahead_period_minutes: extra horizon used to compute ideal end-state (L)
    time_step_minutes:        discrete step size within the MILP (Δt)

    Constraint enforced: time_step_minutes must divide both planning and lookahead periods.
    """
    model     = FinancialModel()
    planning  = planning_period_values  or [60, 120, 180, 240]
    lookahead = lookahead_period_values or [60, 120, 240]
    time_step = check_interval_values   or [30, 60]

    grid = [
        (p, l, t)
        for p, l, t in itertools.product(planning, lookahead, time_step)
        if p % t == 0 and l % t == 0   # Δt must divide both horizons
    ]

    print(f"\nProactive parameter sweep  -  fleet={fleet_size}")
    print(f"Grid size: {len(grid)} combinations "
          f"(planning×lookahead×time_step after filtering)\n")

    results = []
    for planning_p, lookahead_p, ts in grid:
        kwargs = dict(
            planning_period_minutes  = planning_p,
            lookahead_period_minutes = lookahead_p,
            time_step_minutes        = ts,
            check_interval_minutes   = ts,
        )
        if history_log_path:       kwargs["history_log_path"]       = history_log_path

        result = _run_single(
            policy_name   = "proactive",
            fleet_size    = fleet_size,
            n_days        = n_days,
            model         = model,
            policy_kwargs = kwargs,
            param_label   = f"planning={planning_p:>3} lookahead={lookahead_p:>3} Δt={ts:>2}",
        )
        results.append(result)

    _print_optimal(results, ["planning_period_minutes",
                              "lookahead_period_minutes",
                              "time_step_minutes"])
    return results


# ------------------------------------------------------------------ #
#  Nightly sweep (trigger_hour)                                      #
# ------------------------------------------------------------------ #

def run_nightly_sweep(
    fleet_size:             int,
    n_days:                 int   = None,
    trigger_hours:          list  = None,
    check_interval_values:  list  = None,
) -> list[dict]:
    """
    Sweep `trigger_hour` for the nightly policy.
    trigger_hour is when the LP-based rebalancing fires each night (0-23).
    Earlier hours give workers a longer window (6 - trigger_hour hours),
    but may act on stale demand information.
    """
    model = FinancialModel()
    hours                 = trigger_hours         or list(range(0, 5))
    check_interval_values = check_interval_values or [30, 60]

    print(f"\nNightly parameter sweep  -  fleet={fleet_size}")
    print(f"Sweeping trigger_hour ∈ {hours}\n")

    grid = [
        (t, c)
        for t, c in itertools.product(hours, check_interval_values)
    ]

    results = []
    for hour, interval in grid:
        kwargs = dict(
            trigger_hour           = hour,
            check_interval_minutes = interval,
        )

        result = _run_single(
            policy_name   = "nightly",
            fleet_size    = fleet_size,
            n_days        = n_days,
            model         = model,
            policy_kwargs = kwargs,
            param_label   = f"trigger_hour={hour:02d}:00",
        )
        results.append(result)

    _print_optimal(results, ["trigger_hour"])
    return results


# ------------------------------------------------------------------ #
#  Optimal finder + console summary                                  #
# ------------------------------------------------------------------ #

def find_optimal(results: list[dict]) -> dict:
    """Return the result with the highest gross profit per day."""
    return max(results, key=lambda r: r["gross_profit_per_day_eur"])


def _print_optimal(results: list[dict], param_keys: list[str]):
    opt = find_optimal(results)
    params = ", ".join(f"{k}={opt[k]}" for k in param_keys)
    print(f"\n  ✓ Optimal: {params}")
    print(f"    profit/day = €{opt['gross_profit_per_day_eur']:,.0f}  "
          f"unmet = {opt['unmet_rate']:.1%}  "
          f"relocations = {opt['relocations']}\n")


# ------------------------------------------------------------------ #
#  Save                                                              #
# ------------------------------------------------------------------ #

def save_sweep(results: list[dict], filename: str):
    output_dir = Path("./data/metrics/parameters")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Sweep saved to {path}")


def load_sweep(filename: str) -> list[dict]:
    path = Path("./data/metrics/parameters") / f"{filename}.json"
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Shared style helpers                                              #
# ------------------------------------------------------------------ #

_COLORS = {
    "reactive":  "#1f77b4",
    "proactive": "#2ca02c",
    "nightly":   "#ff7f0e",
}

def _color(policy): return _COLORS.get(policy, "#888888")

def _eur_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")


# ------------------------------------------------------------------ #
#  Plots                                                             #
# ------------------------------------------------------------------ #

def plot_profit_by_param(
    results:    list[dict],
    param_key:  str,
    title:      str       = None,
    save_path:  str       = None,
):
    """
    Line plot of gross profit per day vs a single swept parameter.
    Works for alpha (reactive) and trigger_hour (nightly).
    Also overlays unmet rate on a secondary axis for context.
    """
    results  = sorted(results, key=lambda r: r[param_key])
    x_vals   = [r[param_key]                  for r in results]
    profit   = [r["gross_profit_per_day_eur"] for r in results]
    unmet    = [r["unmet_rate"] * 100         for r in results]
    rel_cost = [r["relocation_cost_eur"]      for r in results]

    policy = results[0]["policy"]
    opt    = find_optimal(results)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(x_vals, profit,   color=_color(policy), linewidth=2,
                   marker="o", markersize=7, label="Profit/day (€)")
    l2, = ax2.plot(x_vals, unmet,    color="#d62728",       linewidth=1.5,
                   marker="s", markersize=5, linestyle="--", label="Unmet rate (%)")
    l3, = ax1.plot(x_vals, rel_cost, color="#ff7f0e",       linewidth=1.5,
                   marker="^", markersize=5, linestyle=":",  label="Relocation cost (€)")

    # mark optimum
    ax1.axvline(opt[param_key], color=_color(policy), linestyle=":",
                linewidth=1.5, alpha=0.7)
    ax1.annotate(
        f"Optimal\n{param_key}={opt[param_key]}\n€{opt['gross_profit_per_day_eur']:,.0f}/day",
        xy     = (opt[param_key], opt["gross_profit_per_day_eur"]),
        xytext = (opt[param_key], max(profit) * 0.75),
        fontsize   = 8,
        color      = _color(policy),
        arrowprops = dict(arrowstyle="->", color=_color(policy)),
        ha = "center",
    )

    ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_xlabel(param_key, fontsize=12)
    ax1.set_ylabel("€ per day", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    _eur_fmt(ax1)

    lines  = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or f"{policy.capitalize()} policy - sensitivity to {param_key}",
                  fontsize=13)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_proactive_heatmap(
    results:         list[dict],
    metric:          str  = "gross_profit_per_day_eur",
    check_interval_minutes: int = None,
    save_path:       str  = None,
):
    """
    2-D heatmap of a metric over planning_period × lookahead_period.
    If multiple time_step values exist, either filter to one or average across them.

    metric options (anything in the result dict):
        "gross_profit_per_day_eur"   ← default / primary
        "unmet_rate"
        "relocation_cost_eur"
        "total_revenue_eur"
    """
    if check_interval_minutes is not None:
        results = [r for r in results if r["check_interval_minutes"] == check_interval_minutes]

    # aggregate by (planning, lookahead); average if multiple time_steps
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in results:
        key = (r["planning_period_minutes"], r["lookahead_period_minutes"])
        buckets[key].append(r[metric])

    agg = {k: np.mean(v) for k, v in buckets.items()}

    planning_vals  = sorted(set(k[0] for k in agg))
    lookahead_vals = sorted(set(k[1] for k in agg))

    matrix = np.array([
        [agg.get((p, l), np.nan) for l in lookahead_vals]
        for p in planning_vals
    ])

    # find best cell
    best_idx  = np.unravel_index(np.nanargmax(matrix) if "profit" in metric or "revenue" in metric
                                 else np.nanargmin(matrix), matrix.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   origin="lower",
                   vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar_label = metric.replace("_", " ").replace("eur", "(€)").title()
    cbar.set_label(cbar_label, fontsize=10)

    ax.set_xticks(range(len(lookahead_vals)))
    ax.set_xticklabels([f"{l}min" for l in lookahead_vals])
    ax.set_yticks(range(len(planning_vals)))
    ax.set_yticklabels([f"{p}min" for p in planning_vals])
    ax.set_xlabel("Lookahead period (min)", fontsize=12)
    ax.set_ylabel("Planning period (min)", fontsize=12)

    # annotate cells
    for i in range(len(planning_vals)):
        for j in range(len(lookahead_vals)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text = f"€{val:,.0f}" if "eur" in metric else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, color="black")

    # mark optimal
    ax.add_patch(plt.Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=2, edgecolor="navy", facecolor="none",
    ))

    ts_note = (f" (Δt={check_interval_minutes}min)" if check_interval_minutes is not None
               else " (averaged over check intervals)")
    ax.set_title(f"Proactive policy - {cbar_label}{ts_note}", fontsize=13)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_nightly_hourly(
    results:   list[dict],
    save_path: str = None,
):
    """
    Bar chart of gross profit per day vs trigger hour.
    Coloured green/red for profit/loss; optimal hour highlighted.
    Also shows worker window (hours before 06:00) for reference.
    """
    results = sorted(results, key=lambda r: r["trigger_hour"])
    hours   = [r["trigger_hour"]              for r in results]
    profit  = [r["gross_profit_per_day_eur"]  for r in results]
    unmet   = [r["unmet_rate"] * 100          for r in results]
    opt     = find_optimal(results)

    colors = ["#2ca02c" if p >= 0 else "#d62728" for p in profit]
    colors[hours.index(opt["trigger_hour"])] = "#1f77b4"   # highlight optimal

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bars = ax1.bar(hours, profit, color=colors, alpha=0.8, width=0.6, label="Profit/day (€)")
    ax2.plot(hours, unmet, color="#d62728", linewidth=2, marker="o",
             markersize=6, label="Unmet rate (%)", zorder=5)

    # worker window annotation
    for h in hours:
        window = max(0, 6 - h)
        ax1.text(h, min(profit) * 1.05, f"{window}h",
                 ha="center", va="bottom", fontsize=7, color="gray")

    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Trigger hour (hour of night)", fontsize=12)
    ax1.set_ylabel("Gross profit per day (€)", fontsize=12)
    ax2.set_ylabel("Unmet demand rate (%)", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    _eur_fmt(ax1)

    ax1.set_xticks(hours)
    ax1.set_xticklabels([f"{h:02d}:00" for h in hours])
    ax1.set_title(
        f"Nightly policy - profit & unmet rate by trigger hour\n"
        f"(grey labels = worker window hours, blue bar = optimal: {opt['trigger_hour']:02d}:00)",
        fontsize=12
    )

    lines  = [bars, ax2.lines[0]]
    labels = ["Profit/day (€)", "Unmet rate (%)"]
    ax1.legend(lines, labels, fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_radar(
    optimal_results: dict,
    save_path: str = None,
):
    """
    Radar / spider chart comparing the optimal parameter set of each policy
    across normalised KPIs: profit, unmet rate (inverted), relocation cost
    (inverted), revenue, and served trips.

    optimal_results: dict mapping label → single result dict (from find_optimal).
    """
    metrics_def = [
        # (display label, result key, higher_is_better)
        ("Profit/day",       "gross_profit_per_day_eur",  True),
        ("Served trips",     "total_served",              True),
        ("Revenue",          "total_revenue_eur",         True),
        ("Low unmet",        "unmet_rate",                False),   # inverted
        ("Low reloc. cost",  "relocation_cost_eur",       False),   # inverted
    ]

    labels   = list(optimal_results.keys())
    n_axes   = len(metrics_def)
    angles   = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles  += angles[:1]   # close polygon

    # normalise each metric to [0, 1] across policies
    all_values = {key: [r[key] for r in optimal_results.values()]
                  for _, key, _ in metrics_def}

    def normalise(vals, higher_is_better):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [0.5] * len(vals)
        normed = [(v - lo) / (hi - lo) for v in vals]
        return normed if higher_is_better else [1 - n for n in normed]

    policy_scores = {}
    for i, label in enumerate(labels):
        scores = []
        for _, key, hib in metrics_def:
            all_vals = all_values[key]
            normed   = normalise(all_vals, hib)
            scores.append(normed[i])
        policy_scores[label] = scores + scores[:1]   # close polygon

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw=dict(polar=True))

    for label, scores in policy_scores.items():
        color = _COLORS.get(label.lower(), "#888888")
        ax.plot(angles, scores, color=color, linewidth=2, label=label)
        ax.fill(angles, scores, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m[0] for m in metrics_def], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
    ax.set_title("Optimal parameter set - multi-metric comparison\n(normalised per metric)",
                 fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def plot_cost_breakdown_optimal(
    optimal_results: dict,   # {"Reactive": result_dict, ...}
    save_path: str = None,
):
    """
    Stacked horizontal bar chart showing cost composition at each policy's
    optimal parameter set, alongside revenue and lost revenue.
    """
    cost_components = [
        ("Vehicle ownership",  "vehicle_cost_eur",       "#4c72b0"),
        ("Fuel (trips)",       "fuel_cost_eur",           "#55a868"),
        ("Relocation",         "relocation_cost_eur",     "#c44e52"),
        ("Repositioning",      "repositioning_cost_eur",  "#dd8452"),
    ]

    labels = list(optimal_results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pos    = np.arange(len(labels))
    bar_h    = 0.35
    lefts    = np.zeros(len(labels))

    for comp_label, key, color in cost_components:
        vals = np.array([optimal_results[l].get(key, 0) for l in labels])
        ax.barh(y_pos - bar_h / 2, vals, left=lefts,
                height=bar_h, label=comp_label, color=color, alpha=0.85)
        lefts += vals

    # revenue bar (upper)
    revenue = np.array([optimal_results[l]["total_revenue_eur"] for l in labels])
    lost    = np.array([optimal_results[l]["lost_revenue_eur"]  for l in labels])
    ax.barh(y_pos + bar_h / 2, revenue, height=bar_h,
            label="Earned revenue", color="#2ca02c", alpha=0.75)
    ax.barh(y_pos + bar_h / 2, lost, left=revenue, height=bar_h,
            label="Lost revenue (unmet)", color="#98df8a", alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("EUR (total simulation period)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.set_title("Cost & revenue breakdown at optimal parameters per policy", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


# ------------------------------------------------------------------ #
#  Run all plots at once                                             #
# ------------------------------------------------------------------ #

def plot_all(
    reactive_file:  str = None,
    proactive_file: str = None,
    nightly_file:   str = None,
    output_dir:     str = None,
):
    """
    Load saved sweeps and render all plots.
    Pass None for any policy not run.
    """
    def path(name):
        return str(Path(output_dir) / name) if output_dir else None

    optimal_results = {}

    if reactive_file:
        res = load_sweep(reactive_file)
        plot_profit_by_param(res, "alpha",
                             save_path=path("reactive_alpha_sensitivity.png"))
        optimal_results["Reactive"] = find_optimal(res)

    if proactive_file:
        res = load_sweep(proactive_file)
        # one heatmap per time_step value
        ci_vals = sorted(set(r["check_interval_minutes"] for r in res))
        for ci in ci_vals:
            plot_proactive_heatmap(res, check_interval_minutes=ci,
                                save_path=path(f"proactive_heatmap_ci{ci}.png"))
        # sensitivity to planning period (averaged over others)
        plot_profit_by_param(res, "planning_period_minutes",
                             save_path=path("proactive_planning_sensitivity.png"))
        plot_profit_by_param(res, "lookahead_period_minutes",
                             save_path=path("proactive_lookahead_sensitivity.png"))
        optimal_results["Proactive"] = find_optimal(res)

    if nightly_file:
        res = load_sweep(nightly_file)
        plot_nightly_hourly(res, save_path=path("nightly_trigger_hour.png"))
        optimal_results["Nightly"] = find_optimal(res)

    if len(optimal_results) > 1:
        plot_radar(optimal_results,
                   save_path=path("radar_optimal_params.png"))
        plot_cost_breakdown_optimal(optimal_results,
                                    save_path=path("cost_breakdown_optimal.png"))


if __name__ == "__main__":
    FLEET_SIZE = 500
    N_DAYS     = 30

    # --- Reactive ---
    reactive_results = run_reactive_sweep(
        fleet_size   = FLEET_SIZE,
        n_days       = N_DAYS,
        alpha_values = [100, 200, 300, 400, 500],
    )
    save_sweep(reactive_results, "param_sweep_reactive")

    # --- Proactive ---
    proactive_results = run_proactive_sweep(
        fleet_size              = FLEET_SIZE,
        n_days                  = N_DAYS,
        planning_period_values  = [120, 180, 240, 300],
        lookahead_period_values = [120, 180, 240, 300],
        check_interval_values   = [60, 120, 180],
    )
    save_sweep(proactive_results, "param_sweep_proactive")

    # --- Nightly ---
    nightly_results = run_nightly_sweep(
        fleet_size    = FLEET_SIZE,
        n_days        = N_DAYS,
        trigger_hours = list(range(0, 5)),
    )
    save_sweep(nightly_results, "param_sweep_nightly")

    # --- Plots ---
    plot_all(
        reactive_file  = "param_sweep_reactive",
        proactive_file = "param_sweep_proactive",
        nightly_file   = "param_sweep_nightly",
        output_dir     = "./data/metrics/parameters/plots",
    )