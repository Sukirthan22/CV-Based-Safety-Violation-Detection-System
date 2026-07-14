# rules.py

NO_HELMET = "NO_HELMET"
NO_HARNESS = "NO_HARNESS"

WARNING = "WARNING"
CRITICAL = "CRITICAL"

def evaluate_ppe_rules(person, at_height):
    violations = []

    if not person["helmet"]:
        violations.append((WARNING, NO_HELMET))

    if at_height and not person["harness"]:
        violations.append((CRITICAL, NO_HARNESS))

    return violations
