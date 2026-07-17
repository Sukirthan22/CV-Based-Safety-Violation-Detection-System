import time

class ViolationTracker:
    def __init__(self, tolerance_seconds=1.5, confirm_seconds=1.5, forget_seconds=10.0, cooldown_seconds=60.0):
        self.tolerance_seconds = tolerance_seconds
        self.confirm_seconds = confirm_seconds
        self.forget_seconds = forget_seconds
        self.cooldown_seconds = cooldown_seconds
        
        # State per event_id. event_id is like "NO_HELMET:worker_7"
        self.states = {}

    def update(self, current_violations):
        """
        current_violations is a list of (severity, violation, reason, event_id, person_id)
        Returns (smoothed_by_person, events_to_log)
        where smoothed_by_person is a dict mapping person_id -> list of (sev, violation, reason, event_id)
        """
        now = time.time()
        
        current_event_ids = set()
        for sev, violation, reason, event_id, person_id in current_violations:
            current_event_ids.add(event_id)
            if event_id not in self.states:
                self.states[event_id] = {
                    "first_seen": now,
                    "last_seen": now,
                    "last_logged": None,
                    "severity": sev,
                    "reason": reason,
                    "violation": violation,
                    "person_id": person_id
                }
            else:
                state = self.states[event_id]
                # If it was inactive for longer than tolerance, reset its start time
                if (now - state["last_seen"]) > self.tolerance_seconds:
                    state["first_seen"] = now
                state["last_seen"] = now
                state["severity"] = sev
                state["reason"] = reason

        # Cleanup forgotten events
        stale_ids = [
            eid for eid, state in self.states.items()
            if (now - state["last_seen"]) > self.forget_seconds
        ]
        for eid in stale_ids:
            del self.states[eid]
            
        # Determine active smoothed violations (within tolerance)
        smoothed_by_person = {}
        events_to_log = []
        
        for eid, state in self.states.items():
            # It is active if it was seen within the tolerance window
            if (now - state["last_seen"]) <= self.tolerance_seconds:
                pid = state["person_id"]
                if pid not in smoothed_by_person:
                    smoothed_by_person[pid] = []
                smoothed_by_person[pid].append((state["severity"], state["violation"], state["reason"], eid))
                
                # Check if it should be logged
                if (now - state["first_seen"]) >= self.confirm_seconds:
                    if state["last_logged"] is None or (now - state["last_logged"]) >= self.cooldown_seconds:
                        events_to_log.append((state["severity"], state["violation"], state["reason"], eid))
                        
        return smoothed_by_person, events_to_log
        
    def mark_logged(self, events):
        now = time.time()
        for _, _, _, eid in events:
            if eid in self.states:
                self.states[eid]["last_logged"] = now
