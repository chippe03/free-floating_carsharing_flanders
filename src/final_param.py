OPTIMAL = {
    "baseline": {
        "fleet": 700, "workers": 1,
        "params": {}, "policy_name": "baseline",
        "label": "Baseline", "color": "#00FF99",
    },
    "reactive": {
        "fleet": 700, "workers": 250,
        "params": {"alpha": 150, "check_interval_minutes": 30},
        "policy_name": "reactive",
        "label": "Reactive", "color": "#00CCFF",
    },
    "nightly": {
        "fleet": 775, "workers": 25,
        "params": {"trigger_hour": 1, "check_interval_minutes": 60},
        "policy_name": "nightly",
        "label": "Nightly", "color": "#9933FF",
    },
    "proactive": {
        "fleet": 700, "workers": 150,
        "params": {"planning_period_minutes": 120, "lookahead_period_minutes": 60,
                   "time_step_minutes": 60, "check_interval_minutes": 60},
        "policy_name": "proactive",
        "label": "Proactive", "color": "#FF6EC7",
    },
}