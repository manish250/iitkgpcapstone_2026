"""UI tests (mock mode) mirroring the manual script in the technical summary."""
import os, sys
from pathlib import Path
os.environ["POLICY_AGENT_MODE"] = "mock"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
import mock_agent
import streamlit as st


def fresh():
    st.cache_resource.clear()          # limiters are shared per process: start each test clean
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    return at


def ask(at, q):
    at.chat_input[0].set_value(q).run()
    assert not at.exception, at.exception
    return at


def texts(at):
    return " ".join(m.value for m in at.markdown)


def test_empty_state_and_demo_badge():
    at = fresh()
    assert len(at.button) >= 3          # suggested questions + new chat
    assert not at.chat_message


def test_verified_answer_has_green_citation_and_sources():
    at = ask(fresh(), "Can an employee approve their own travel expense report?")
    assert ":green-badge[Sample Travel and Expense Policy, p. 4]" in texts(at)
    assert any("Sources (1)" in e.label for e in at.expander)
    assert any("cost unavailable" in c.value for c in at.caption)


def test_clarification_and_fallback_have_no_sources():
    at = ask(fresh(), "policy?")
    assert not at.expander and "more details" in texts(at)
    at = ask(fresh(), "what is the weather like today")
    assert not at.expander and "could not verify" in texts(at).lower()


def open_form(at):
    ask(at, "Can an employee approve their own travel expense report?")
    [b for b in at.button if "No, I still need help" in b.label][0].click().run()
    assert any("Contact University Support" in s.value for s in at.subheader)
    return at


def test_yes_feedback_sends_nothing():
    at = ask(fresh(), "Can an employee approve their own travel expense report?")
    n = len(mock_agent.DemoSupport.tickets)
    [b for b in at.button if "Yes, this helped" in b.label][0].click().run()
    assert len(mock_agent.DemoSupport.tickets) == n
    assert not any("Contact University Support" in s.value for s in at.subheader)


def test_support_requires_consent_then_submits_demo_receipt():
    at = open_form(fresh())
    assert "Question:" in at.text_area[0].value                      # prefilled, nothing sent yet
    submit = [b for b in at.button if b.label == "Submit request"][0]
    submit.click().run()
    assert any("consent" in e.value.lower() for e in at.error)       # blocked: consent unticked
    at.checkbox[0].check()
    n = len(mock_agent.DemoSupport.tickets)
    [b for b in at.button if b.label == "Submit request"][0].click().run()
    assert any("SR-" in s.value for s in at.success)
    assert any("DEMO ONLY" in w.value for w in at.warning)
    assert len(mock_agent.DemoSupport.tickets) == n + 1


def test_blank_subject_blocked():
    at = open_form(fresh())
    at.text_input[0].set_value("")
    at.checkbox[0].check()
    [b for b in at.button if b.label == "Submit request"][0].click().run()
    assert any("subject" in e.value.lower() for e in at.error)


def test_unresolved_retry_keeps_same_reference_and_blocks_chat(monkeypatch):
    calls = []
    def flaky(**kw):
        calls.append(kw["ticket_number"])
        if len(calls) == 1:
            raise RuntimeError("Support service request could not be confirmed (ConnectionError).")
        return {"ticket_number": kw["ticket_number"], "status": "submitted", "http_status": 200}
    monkeypatch.setattr(mock_agent.DemoSupport, "submit_ticket", staticmethod(flaky))
    at = open_form(fresh())
    at.checkbox[0].check()
    [b for b in at.button if b.label == "Submit request"][0].click().run()
    assert at.chat_input[0].disabled                                 # chat blocked while unresolved
    assert any("could not confirm" in w.value.lower() for w in at.warning)
    [b for b in at.button if b.label == "Retry submission"][0].click().run()
    assert len(calls) == 2 and calls[0] == calls[1]                  # same idempotency reference
    assert any("SR-" in s.value for s in at.success)
    assert not at.chat_input[0].disabled


def test_definitive_rejection_reopens_form_with_new_reference(monkeypatch):
    class Rejected(Exception): pass
    Rejected.__name__ = "M4SupportTicketRejected"
    def reject(**kw): raise Rejected("Support service rejected the request (HTTP 422).")
    monkeypatch.setattr(mock_agent.DemoSupport, "submit_ticket", staticmethod(reject))
    at = open_form(fresh())
    at.checkbox[0].check()
    [b for b in at.button if b.label == "Submit request"][0].click().run()
    assert any("Nothing was submitted" in e.value for e in at.error)
    assert at.session_state.support["phase"] == "form" and not at.session_state.support["ticket_number"]


def test_new_chat_mints_new_thread():
    at = ask(fresh(), "Can an employee approve their own travel expense report?")
    old = at.session_state.thread_id
    [b for b in at.button if "New chat" in b.label][0].click().run()
    assert at.session_state.thread_id != old and not at.chat_message


def test_related_questions_shown_and_clickable():
    at = ask(fresh(), "Can an employee approve their own travel expense report?")
    labels = [b.label for b in at.button]
    for q in mock_agent.SAMPLE_FOLLOWUPS:
        assert q in labels
    [b for b in at.button if b.label == mock_agent.SAMPLE_FOLLOWUPS[0]][0].click().run()
    assert not at.exception
    users = [m for m in at.chat_message if m.name == "user"]
    assert mock_agent.SAMPLE_FOLLOWUPS[0] in users[-1].text[0].value


def test_no_related_questions_for_clarification_or_fallback():
    at = ask(fresh(), "policy?")
    assert not any(q in [b.label for b in at.button] for q in mock_agent.SAMPLE_FOLLOWUPS)
    assert "Related questions" not in texts(at)


def test_chat_rate_limit_blocks_third_question(monkeypatch):
    monkeypatch.setenv("CHAT_RPM", "2")
    at = fresh()
    ask(at, "Can an employee approve their own travel expense report?")
    ask(at, "What expenses are reimbursable for travel?")
    ask(at, "How do I submit a Concur expense report?")
    assert any("Too many requests" in w.value for w in at.warning)
    assert len([m for m in at.chat_message if m.name == "user"]) == 2   # third question not processed
    at2 = fresh()                                                         # limiter reset -> allowed again
    ask(at2, "Can an employee approve their own travel expense report?")
    assert not any("Too many requests" in w.value for w in at2.warning)


def test_support_ticket_rate_limit(monkeypatch):
    monkeypatch.setenv("TICKET_RPM", "1")
    at = fresh()
    for n in range(2):
        ask(at, ["Can an employee approve their own travel expense report?",
                 "What expenses are reimbursable for travel?"][n])
        [b for b in at.button if "No, I still need help" in b.label][0].click().run()
        at.checkbox[0].check()
        [b for b in at.button if b.label == "Submit request"][0].click().run()
        if n == 0:
            assert any("SR-" in s.value for s in at.success)           # first ticket goes through
    assert any("Too many submissions" in e.value for e in at.error)    # second within the minute is blocked
