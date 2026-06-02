import json
from pathlib import Path
from src import config

from src.relocation.base_policy import BaseRelocationPolicy
from src.relocation.reactive import ReactivePolicy
from src.relocation.proactive import ProactivePolicy
from src.relocation.nightly_rebalance import NightlyRebalancePolicy


def load_policy(policy_name:             str = None,    check_interval_minutes:    int = None,
                penalty_unmet:         float = None,     cost_per_relocation:    float = None,
                verbose:                bool = False,

                # REACTIVE
                thresholds:             dict = None,     alpha:                    int = None,
                
                # PROACTIVE
                planning_period_minutes: int = None,     lookahead_period_minutes: int = None,
                time_step_minutes:       int = None,     solver_time_limit_sec:    int = None,
                history_log_path:        str = None,

                # NIGHTLY
                trigger_hour:           int   = None,

) -> BaseRelocationPolicy | None:
    """
    Reads simulation.yaml and returns the correct policy object.
    Returns None for baseline (no relocation).
    """
    sim_cfg = config.simulation
    if policy_name is None:
        policy_name = sim_cfg["relocation"]["policy"]

    if policy_name == "baseline":
        return None

    elif policy_name == "reactive":
        policy = ReactivePolicy(
            check_interval_minutes = check_interval_minutes,
            thresholds             = thresholds,
            alpha                  = alpha,
            verbose                = verbose,
        )
        return policy

    elif policy_name == "proactive":
        policy   = ProactivePolicy(
            check_interval_minutes   = check_interval_minutes,
            planning_period_minutes  = planning_period_minutes,
            lookahead_period_minutes = lookahead_period_minutes,
            time_step_minutes        = time_step_minutes,
            penalty_unmet            = penalty_unmet,
            cost_per_relocation      = cost_per_relocation,
            solver_time_limit_sec    = solver_time_limit_sec,
            verbose                  = verbose,
        )
        # load history if a path is provided
        if history_log_path:
            path = Path(history_log_path)
            if path.exists():
                with open(path) as f:
                    policy.load_history(json.load(f))
            else:
                print(f"  No history file found at {history_log_path}.")
        return policy

    elif policy_name == "nightly":
        policy = NightlyRebalancePolicy(
            check_interval_minutes = check_interval_minutes,
            trigger_hour           = trigger_hour,
            penalty_unmet          = penalty_unmet,
            cost_per_relocation    = cost_per_relocation,
            verbose                = verbose,
        )
        return policy

    else:
        raise ValueError(f"Unknown policy '{policy_name}'. "
                         f"Choose: baseline / reactive / proactive / nightly")