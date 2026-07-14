# rules.py

SAFE = "SAFE"
HIGH_RISK = "HIGH_RISK"

NO_HELMET = "NO_HELMET"
NO_HARNESS = "NO_HARNESS"

WARNING = "WARNING"
CRITICAL = "CRITICAL"

def evaluate_ppe_rules(person, zone, at_height):
    violations = []

    if zone == HIGH_RISK and not person["helmet"]:
        violations.append((WARNING, NO_HELMET))

    if at_height and not person["harness"]:
        violations.append((CRITICAL, NO_HARNESS))

    return violations


