"""Exercises the extracted LangGraph agent with a fake LLM + fake index (no network)."""
import json, os, sys, tempfile, types
from pathlib import Path

os.environ["M4_CHECKPOINT_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "cp.sqlite")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_agent as pa
from langchain_core.documents import Document


class FakeLLM:
    def __init__(self, cite_ok=True):
        self.cite_ok = cite_ok
        self.followup_text = json.dumps({"questions": [
            "Who can approve travel expense reports?",
            "Can an employee approve their own expense report?",   # same as the asked question -> dropped
            "What is the deadline to submit an expense report?",
            "Not a question",                                        # no '?' -> dropped
            "Which expenses are reimbursable under the policy?",
            "What happens if an expense report is late?"]})
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **kw):
        sysmsg = kw["messages"][0]["content"]
        user = kw["messages"][1]["content"]
        if sysmsg.startswith("You are the planner"):
            q = json.loads(user)["user_query"]
            if len(q.split()) < 3:
                out = {"action": "ask_clarification", "search_query": "",
                       "clarification_question": "Which policy do you mean?"}
            else:
                out = {"action": "search_policy", "search_query": q, "clarification_question": ""}
            text = json.dumps(out)
        elif sysmsg.startswith("You suggest follow-up"):
            text = self.followup_text
        elif sysmsg.startswith("Review the draft"):
            text = json.dumps({"verdict": "accept", "issues": [], "feedback": ""})
        elif sysmsg.startswith("Revise"):
            text = "Approval is required [Travel Policy, PDF p. 3]."
        else:
            text = ("Approval is required [Travel Policy, PDF p. 3]." if self.cite_ok
                    else "Approval is required, no citation.")
        usage = types.SimpleNamespace(completion_tokens=20)
        msg = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def setup(cite_ok=True):
    doc = Document(page_content="Employees may not approve their own expenses.",
                   metadata={"source_page": 3, "chunk_id": 0,
                             "split_pdf_member": "01_Travel_Policy_page_003.txt"})
    pa.policy_docs = [doc]
    pa.client = FakeLLM(cite_ok)
    pa.hybrid_rerank_search = lambda q, k=5, candidate_k=20: [(doc, 0.9)]
    pa._READY = True


def test_accepted_with_evidence():
    setup()
    r = pa.run_turn("Can I approve my own expense report?", "policy-ui-t1")
    assert r["status"] == "accepted"
    assert r["evidence"] and r["evidence"][0]["source"] == "Travel Policy"
    assert r["telemetry"]["model_calls"] == 3 and r["telemetry"]["estimated_cost_usd"] is None


def test_clarification_has_no_evidence():
    setup()
    r = pa.run_turn("policy?", "policy-ui-t2")
    assert r["status"] == "clarification_needed" and r["evidence"] == []


def test_unmatched_citation_never_verified():
    setup(cite_ok=False)
    r = pa.run_turn("Can I approve my own expense report?", "policy-ui-t3")
    assert r["status"] in {"safe_fallback", "revision_unverified"}
    assert r["evidence"] == []   # evidence withheld unless accepted


def test_followups_three_grounded_and_filtered():
    setup()
    r = pa.run_turn("Can an employee approve their own expense report?", "policy-ui-t4")
    fu = r["followups"]
    assert len(fu) == 3 and all(q.endswith("?") for q in fu)
    assert "Can an employee approve their own expense report?" not in fu


def test_followups_empty_when_not_accepted_or_bad_json():
    setup()
    assert pa.run_turn("policy?", "policy-ui-t5")["followups"] == []       # clarification
    pa.client.followup_text = "sorry, no json here"
    ev = [{"source": "Travel Policy", "text": "x"}]
    assert pa.suggest_followups("q?", "a", ev) == []
    pa.client.followup_text = '<think>hmm {x}</think>{"questions": ["What is the late filing penalty?"]}'
    assert pa.suggest_followups("q?", "a", ev) == ["What is the late filing penalty?"]


def test_followups_can_be_turned_off(monkeypatch):
    setup()
    monkeypatch.setattr(pa, "FOLLOWUPS_ENABLED", False)
    assert pa.run_turn("Can I approve my own expense report?", "policy-ui-t6")["followups"] == []
