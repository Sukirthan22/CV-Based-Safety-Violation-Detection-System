import time

class ViolationTracker:
    def __init__(self, tolerance_seconds=1.5, confirm_seconds=1.5, forget_seconds=10.0, cooldown_seconds=60.0):
        self.tolerance_seconds = tolerance_seconds
        self.confirm_seconds = confirm_seconds
        self.forget_seconds = forget_seconds
        self.cooldown_seconds = cooldown_seconds
        
        # State per event_id. event_id is like "NO_HELMET:worker_7"
        self.states = {}

    def update(self, current_violations, current_person_ids=None):
        """
        current_violations is a list of (severity, violation, reason, event_id, person_id)
        current_person_ids is an optional set of person_ids visible in the current frame.
        Returns (smoothed_by_person, events_to_speak, events_started, events_ended)
        where smoothed_by_person is a dict mapping person_id -> list of (sev, violation, reason, event_id)
        """
        now = time.time()
        if current_person_ids is None:
            current_person_ids = set()
        
        current_event_ids = set()
        for sev, violation, reason, event_id, person_id in current_violations:
            current_event_ids.add(event_id)
            if event_id not in self.states:
                self.states[event_id] = {
                    "first_seen": now,
                    "last_seen": now,
                    "last_spoken": None,
                    "is_logged_as_started": False,
                    "severity": sev,
                    "reason": reason,
                    "violation": violation,
                    "person_id": person_id
                }
            else:
                state = self.states[event_id]
                # If it was inactive for longer than tolerance, reset its start time
                tol = 3.0 if state["violation"] == "NO_HARNESS" else self.tolerance_seconds
                if (now - state["last_seen"]) > tol:
                    state["first_seen"] = now
                state["last_seen"] = now
                state["severity"] = sev
                state["reason"] = reason

        # Expire violations for people who are visible but no longer violating
        # We wait for `tolerance_seconds` to pass first to ensure it's not just a 1-frame AI flicker!
        for eid, state in self.states.items():
            if eid not in current_event_ids and state["person_id"] in current_person_ids:
                tol = 3.0 if state["violation"] == "NO_HARNESS" else self.tolerance_seconds
                if (now - state["last_seen"]) > tol:
                    # The person is in the frame, but this violation has been missing for 
                    # longer than the flicker tolerance. They fixed it! Instantly expire it.
                    state["last_seen"] = 0

        stale_ids = [
            eid for eid, state in self.states.items()
            if (now - state["last_seen"]) > self.forget_seconds
        ]
        events_ended = []
        for eid in stale_ids:
            state = self.states[eid]
            if state.get("is_logged_as_started"):
                events_ended.append((state["severity"], state["violation"], state["reason"], eid, state["person_id"]))
            del self.states[eid]
            
        smoothed_by_person = {}
        events_to_speak = []
        events_started = []
        
        for eid, state in self.states.items():
            tol = 3.0 if state["violation"] == "NO_HARNESS" else self.tolerance_seconds
            if (now - state["last_seen"]) <= tol:
                pid = state["person_id"]
                if pid not in smoothed_by_person:
                    smoothed_by_person[pid] = []
                smoothed_by_person[pid].append((state["severity"], state["violation"], state["reason"], eid))

                required_time = 3.0 if state["violation"] == "NO_HARNESS" else self.confirm_seconds
                if (now - state["first_seen"]) >= required_time:
                    if not state.get("is_logged_as_started"):
                        events_started.append((state["severity"], state["violation"], state["reason"], eid, state["person_id"]))
                        
                    if state.get("last_spoken") is None or (now - state.get("last_spoken")) >= self.cooldown_seconds:
                        events_to_speak.append((state["severity"], state["violation"], state["reason"], eid, state["person_id"]))
                        
        return smoothed_by_person, events_to_speak, events_started, events_ended
        
    def mark_started(self, events):
        for _, _, _, eid, _ in events:
            if eid in self.states:
                self.states[eid]["is_logged_as_started"] = True

    def mark_spoken(self, events):
        now = time.time()
        for _, _, _, eid, _ in events:
            if eid in self.states:
                self.states[eid]["last_spoken"] = now
