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
