# rules.py

NO_HELMET = "NO_HELMET"
NO_HARNESS = "NO_HARNESS"
NEAR_EDGE = "NEAR_EDGE"

WARNING = "WARNING"
CRITICAL = "CRITICAL"


def evaluate_ppe_rules(person, at_height, edge_zone=None):
    violations = []

    if not person["helmet"]:
        violations.append((WARNING, NO_HELMET))

    # Positional advisory: independent of PPE, so a harnessed worker at the
    # edge is still told where he is standing.
    if edge_zone:
        violations.append((WARNING, f"{NEAR_EDGE}:{edge_zone}"))

    if at_height and not person["harness"]:
        violations.append((CRITICAL, NO_HARNESS))

    return violations
