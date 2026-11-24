"""Demo backend: lets the UI run with no model, corpus, keys or network.
All content here is SAMPLE data and is labelled as such in the UI."""
import secrets
import time
from datetime import datetime, timezone

_KEYWORDS = ("expense", "travel", "approve", "reimburs", "concur")

SAMPLE_EVIDENCE = [{
    "evidence_id": "e1",
    "source": "Sample Travel and Expense Policy",
    "page": 4,
    "text": ("SAMPLE FIXTURE - not real policy text. Employees may not approve their own "
             "expense reports; approval must come from a supervisor or delegate."),
    "score": 0.91,
}]


SAMPLE_FOLLOWUPS = [
    "Who can approve travel expense reports for a department?",
    "How long do I have to submit an expense report?",
    "What expenses are not reimbursable?",
]


def run_turn(question, thread_id):
    time.sleep(0.4)
    q = (question or "").strip()
    base = {"plan": {}, "evidence": [], "followups": []}
    tele = {"latency_ms": 400.0, "tool_calls": 0, "model_calls": 1, "generation_tokens": 0,
            "evidence_count": 0, "estimated_cost_usd": None, "failure_category": "none_observed"}
    if len(q.split()) < 3:
        return {**base, "status": "clarification_needed",
                "answer": "Which policy or situation do you mean? Please add a few more details.",
                "telemetry": tele}
    if any(k in q.casefold() for k in _KEYWORDS):
        tele.update(tool_calls=2, model_calls=3, generation_tokens=180, evidence_count=1)
        return {**base, "status": "accepted", "evidence": SAMPLE_EVIDENCE,
                "followups": SAMPLE_FOLLOWUPS,
                "answer": ("[Demo answer] An employee cannot approve their own expense report; "
                           "a supervisor or delegate must approve it "
                           "[Sample Travel and Expense Policy, PDF p. 4]."),
                "telemetry": tele}
    tele.update(failure_category="manual_failure_review_required")
    return {**base, "status": "safe_fallback",
            "answer": "I could not verify an answer from the retrieved policy excerpts.",
            "telemetry": tele}


class DemoSupport:
    """Stands in for the University Support API. Nothing leaves this process."""
    tickets = []

    @staticmethod
    def new_ticket_number():
        alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        suffix = "".join(secrets.choice(alphabet) for _ in range(12))
        return f"SR-{datetime.now(timezone.utc):%Y%m%d}-{suffix}"

    @classmethod
    def submit_ticket(cls, *, ticket_number, subject, description,
                      current_question, current_answer, consent):
        if consent is not True:
            raise PermissionError("Explicit user consent is required.")
        if not any(t["ticket_number"] == ticket_number for t in cls.tickets):  # idempotent
            cls.tickets.append({"ticket_number": ticket_number, "subject": subject})
        return {"ticket_number": ticket_number, "status": "submitted", "http_status": 200}
