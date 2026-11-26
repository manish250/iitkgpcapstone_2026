"""Runs the agent team's VERBATIM notebook code (policy_agent.py) through ui_adapter.py.

Only the outside world is faked: the LLM server, the embedding server and the
cross-encoder download. The real corpus is ingested, chunked, indexed and searched
by the notebook's own code, in a temporary working folder.
"""
import json, os, re, shutil, sys, tempfile, types
from pathlib import Path

import certifi
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- the world ----
WORK = Path(tempfile.mkdtemp())
(WORK / "data").mkdir()
shutil.copy(certifi.where(), WORK / "data" / "vm02-ca.crt")          # notebook cell: data/vm02-ca.crt
shutil.copy(certifi.where(), WORK / "vm05-ca.crt")
repo_data = WORK / "iitkgpcapstone_2026" / "data"                       # notebook clones this repo; pre-placed
repo_data.mkdir(parents=True)
shutil.copy(ROOT / "data" / "University_Policy_Document_Corpus.zip", repo_data)
os.environ.update({
    "OPENAI_API_KEY": "test-llm-key", "EMBEDDINGS_KEY": "test-emb-key",
    "LLM_CA_CERT": str(WORK / "vm05-ca.crt"),
    "M4_CHECKPOINT_DB_PATH": str(WORK / "cp.sqlite"),
    "OTEL_SDK_DISABLED": "true",
})


class FakeLLM:
    """Plays planner / draft / critic / revise / follow-up by reading the system prompt."""
    def __init__(self, *a, cite_ok=True, critic="accept", **kw):
        self.cite_ok, self.critic, self.calls = cite_ok, critic, []
        self.draft_text = None                                   # override the draft's wording
        self.followup_text = json.dumps({"questions": [
            "Who can approve travel expense reports?",
            "Can an employee approve their own expense report?",   # same as the asked question -> dropped
            "What is the deadline to submit an expense report?",
            "Not a question",                                        # no '?' -> dropped
            "Which expenses are reimbursable under the policy?",
            "What happens if an expense report is late?"]})
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    @staticmethod
    def _first_source(text):
        m = re.search(r"\[Source: ([^,\]]+), PDF p\. ([^\]]+)\]", text)
        return f"[{m.group(1)}, PDF p. {m.group(2)}]" if m else "[Travel Policy, PDF p. 3]"

    def create(self, **kw):
        msgs = kw["messages"]
        sysmsg, user = msgs[0]["content"], msgs[-1]["content"]
        self.calls.append(sysmsg[:40])
        if sysmsg.startswith("You are the planner"):
            body = user.strip().removeprefix("<user_query>").removesuffix("</user_query>")
            q = json.loads(body)["user_query"]
            out = ({"action": "ask_clarification", "search_query": "",
                    "clarification_question": "Which policy do you mean?"} if len(q.split()) < 3 else
                   {"action": "search_policy", "search_query": q, "clarification_question": ""})
            text = json.dumps(out)
        elif sysmsg.startswith("You suggest follow-up"):
            text = self.followup_text
        elif sysmsg.startswith("Review the draft"):
            text = json.dumps({"verdict": self.critic,
                               "issues": [] if self.critic == "accept" else ["tighten"],
                               "feedback": "" if self.critic == "accept" else "Tighten the answer."})
        elif sysmsg.startswith("Revise"):
            text = (self.draft_text.format(cite=self._first_source(user)) if self.draft_text
                    else f"Approval is required {self._first_source(user)}.")
        elif len(msgs) == 1:                                  # notebook's connection test
            text = "hello"
        elif self.draft_text:                                 # draft with fixed wording
            text = self.draft_text.format(cite=self._first_source(user))
        else:                                                 # draft
            text = (f"Approval is required {self._first_source(user)}." if self.cite_ok
                    else "Approval is required, no citation.")
        usage = types.SimpleNamespace(completion_tokens=20)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=text))], usage=usage)


def _fake_embeddings_post(url, headers=None, json=None, verify=None, **kw):
    assert url.endswith("/embeddings") and verify, "notebook must verify TLS"
    def vec(t):
        v = [0.0] * 64
        for w in re.findall(r"[a-z0-9]+", t.casefold()):
            v[hash(w) % 64] += 1.0
        return v
    data = [{"index": i, "embedding": vec(t)} for i, t in enumerate(json["input"])]
    return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": data})


class FakeCrossEncoder:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs):
        return [float(len(set(q.casefold().split()) & set(d.casefold().split()))) for q, d in pairs]


sys.modules["sentence_transformers"] = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
sys.modules["huggingface_hub"] = types.SimpleNamespace(scan_cache_dir=lambda: types.SimpleNamespace(repos=[]))

import openai, requests
_old_cwd = os.getcwd()
os.chdir(WORK)
_saved = (openai.OpenAI, requests.post)
openai.OpenAI, requests.post = FakeLLM, _fake_embeddings_post
try:
    import ui_adapter as ua
    pa = ua.init_runtime()                       # runs all copied notebook cells
finally:
    openai.OpenAI = _saved[0]                    # embeddings stay faked for query-time calls
REAL = {"policy_docs": pa.policy_docs, "hybrid_rerank_search": pa.hybrid_rerank_search}


def setup(cite_ok=True, critic="accept", real_index=False):
    pa.client = FakeLLM(cite_ok=cite_ok, critic=critic)
    if real_index:
        pa.policy_docs, pa.hybrid_rerank_search = REAL["policy_docs"], REAL["hybrid_rerank_search"]
        return
    from langchain_core.documents import Document
    doc = Document(page_content="Employees may not approve their own expenses.",
                   metadata={"source_page": 3, "chunk_id": 0,
                             "split_pdf_member": "01_Travel_Policy_page_003.txt"})
    pa.policy_docs = [doc]
    pa.hybrid_rerank_search = lambda q, k=5, candidate_k=20: [(doc, 0.9)]


# ------------------------------------------------------------------ tests ----
def test_policy_agent_is_verbatim_notebook_code():
    nb = json.loads((ROOT / "University_Policy_Copilot_TrackB.ipynb").read_text(encoding="utf-8"))
    src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"])}
    text = (ROOT / "policy_agent.py").read_text(encoding="utf-8")
    parts = re.split(r"\n# ======== notebook cell (\d+) \((whole cell|definition only), verbatim\) ========\n", text)
    blocks = [(int(parts[k]), parts[k + 1], parts[k + 2].rstrip("\n")) for k in range(1, len(parts), 3)]
    assert len(blocks) == 25
    for idx, kind, body in blocks:
        if kind == "whole cell":
            assert body == src[idx].rstrip("\n"), f"cell {idx} differs from the notebook"
        else:
            assert body and body in src[idx], f"definition from cell {idx} differs from the notebook"
    # nothing else in the file besides the header comment and the copied blocks
    assert all(l.startswith("#") or not l.strip() for l in parts[0].splitlines())


def test_notebook_loaded_real_corpus():
    s = ua.corpus_stats()
    assert s["chunks"] > 3000 and s["reranker"] is True
    assert pa.M4_POLICY_GRAPH is not None


def test_real_index_accepted_turn_has_top5_sources_and_followups():
    setup(real_index=True)
    r = ua.run_turn("What is Concur used for?", "ui-real-1")
    assert r["status"] == "accepted"
    assert 1 <= len(r["evidence"]) <= 5
    assert {"evidence_id", "source", "page", "text", "score"} <= set(r["evidence"][0])
    assert len(r["followups"]) == 3
    assert r["telemetry"]["model_calls"] == 3 and r["telemetry"]["latency_ms"] > 0


def test_clarification_has_no_evidence():
    setup()
    r = ua.run_turn("policy?", "ui-t2")
    assert r["status"] == "clarification_needed" and r["evidence"] == [] and r["followups"] == []


def test_revised_answer_shows_sources_and_followups():
    setup(critic="revise")
    r = ua.run_turn("Can an employee approve their own expense report?", "ui-t6")
    assert r["status"] == "revised" and r["answer"].startswith("Approval is required")
    assert len(r["evidence"]) == 1 and len(r["followups"]) == 3
    assert r["telemetry"]["model_calls"] == 4


def test_uncited_revision_is_never_shown():
    setup(cite_ok=False)
    orig = pa.client.create
    def no_cite(**kw):
        if kw["messages"][0]["content"].startswith("Revise"):
            kw["messages"][0]["content"] = "draft"          # revision comes back uncited too
        return orig(**kw)
    pa.client.chat.completions.create = no_cite
    r = ua.run_turn("Can I approve my own expense report?", "ui-t3")
    assert r["status"] in {"safe_fallback", "revision_unverified"}
    assert r["evidence"] == [] and r["followups"] == []
    assert r["answer"] == ua.FALLBACK_TEXT


def test_followups_filtered_and_safe_on_bad_json():
    setup()
    fu = ua.run_turn("Can an employee approve their own expense report?", "ui-t4")["followups"]
    assert len(fu) == 3 and all(q.endswith("?") for q in fu)
    assert "Can an employee approve their own expense report?" not in fu
    pa.client.followup_text = "sorry, no json here"
    assert ua.suggest_followups("q?", "a", [{"source": "Travel Policy", "text": "x"}]) == []


def test_missing_key_gives_clear_error_not_a_hang():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY and EMBEDDINGS_KEY"):
        ua._refuse_getpass("Please enter your shared Embedding API Key: ")
    with pytest.raises(RuntimeError, match="keyboard input"):
        ua._answer_input("Enter your University Policy Question: ")
    assert ua._answer_input("Enter your group CA certificate filename") == os.environ["LLM_CA_CERT"]


def test_ui_live_mode_revised_answer_shows_sources_and_related_questions():
    """Streamlit UI in live mode on top of the verbatim notebook code."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    setup(critic="revise", real_index=True)
    old = os.environ.get("POLICY_AGENT_MODE")
    os.environ["POLICY_AGENT_MODE"] = "live"
    try:
        st.cache_resource.clear()
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception, at.exception
        at.chat_input[0].set_value("What is Concur used for?").run()
        assert not at.exception, at.exception
        page = json.dumps([str(e.proto) for e in at.get("badge")]) + " ".join(m.value for m in at.markdown)
        assert "Revised to match cited policy" in page
        assert ":green-badge[" in page and "unverified" not in page     # citations match the sources
        assert any(e.label.startswith("Sources (") for e in at.expander)
        labels = [b.label for b in at.button]
        assert "Who can approve travel expense reports?" in labels      # related questions rendered
    finally:
        if old is None:
            os.environ.pop("POLICY_AGENT_MODE", None)
        else:
            os.environ["POLICY_AGENT_MODE"] = old
        st.cache_resource.clear()


RAIL_NOT_FOUND = ("I could not find a specific mention of a rail ticket budget cap in the "
                  "retrieved policy excerpts {cite}.")


def test_not_found_answer_is_labelled_and_has_no_related_questions():
    setup(real_index=True)
    pa.client.draft_text = RAIL_NOT_FOUND
    r = ua.run_turn("What about rail ticket budget cap?", "ui-nf-1")
    assert r["status"] == "not_found"
    assert 1 <= len(r["evidence"]) <= 5          # shown as passages searched
    assert r["followups"] == []


def test_not_found_classifier():
    yes = [RAIL_NOT_FOUND.format(cite="[Travel Expense Policy, PDF p. 11]"),
           "I could not find this in the retrieved policy excerpts.",
           "The retrieved excerpts do not mention a rail ticket cap [Travel Expense Policy, PDF p. 11]. Rail is economy.",
           "There is no specific mention of rail fare limits in the provided policy excerpts."]
    no = ["Based on the retrieved excerpts, the following can be submitted via Concur:\n- Moving. "
          "The policy does not specify a comprehensive list.",
          "Concur is used for submitting expense reports [Business Expense Policy, PDF p. 15].",
          "There is no cap on rail fares, but travelers must book economy class [Travel Expense Policy, PDF p. 11].",
          "No, an employee cannot approve their own expense report [Business Expense Policy, PDF p. 4].",
          "Alcohol is not covered by the University [Business Expense Policy, PDF p. 9]."]
    assert all(ua.is_not_found_answer(t) for t in yes)
    assert not any(ua.is_not_found_answer(t) for t in no)


def test_ui_live_mode_not_found_answer():
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    setup(critic="revise", real_index=True)
    pa.client.draft_text = RAIL_NOT_FOUND
    old = os.environ.get("POLICY_AGENT_MODE")
    os.environ["POLICY_AGENT_MODE"] = "live"
    try:
        st.cache_resource.clear()
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        at.chat_input[0].set_value("What about rail ticket budget cap?").run()
        assert not at.exception, at.exception
        page = json.dumps([str(e.proto) for e in at.get("badge")]) + " ".join(m.value for m in at.markdown)
        assert "Not found in the policies" in page and "Revised to match" not in page
        answer_md = " ".join(m.value for m in at.markdown if "rail ticket" in m.value)
        assert ":gray-badge[checked:" in answer_md and ":green-badge[" not in answer_md
        assert any(e.label.startswith("Passages searched (") for e in at.expander)
        assert not any(e.label.startswith("Sources (") for e in at.expander)
        assert "Related questions" not in " ".join(m.value for m in at.markdown)
        assert "✔ cited" not in page
    finally:
        if old is None:
            os.environ.pop("POLICY_AGENT_MODE", None)
        else:
            os.environ["POLICY_AGENT_MODE"] = old
        st.cache_resource.clear()
