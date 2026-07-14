def decide_alert_action(violations):
    if not violations:
        return "INFO"

    severities = [v[0] for v in violations]

    if "CRITICAL" in severities:
        return "CRITICAL"
    if "WARNING" in severities:
        return "WARNING"

    return "INFO"

