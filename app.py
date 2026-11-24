"""University Policy Copilot - Streamlit UI.

Run:  streamlit run app.py
Mode: POLICY_AGENT_MODE=mock (default, sample data) | live (real agent from policy_agent.py)
"""
import os
import re
import uuid
from types import SimpleNamespace

import streamlit as st

try:  # optional: load a local .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

APP_TITLE = os.environ.get("APP_TITLE", "University Policy Copilot")
MODE = os.environ.get("POLICY_AGENT_MODE", "mock").strip().lower()
MAX_QUESTION_CHARS = 1000

st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="wide")

STATUS = {  # status -> (badge colour, label)
    "accepted": ("green", "Verified against policy"),
    "clarification_needed": ("blue", "Needs more information"),
    "revision_unverified": ("orange", "Could not verify"),
    "safe_fallback": ("orange", "Could not verify"),
    "error": ("red", "Something went wrong"),
}
CITATION = re.compile(r"\[([^,\]]+),\s*PDF\s+p\.\s*(\d+)\]", re.IGNORECASE)  # notebook format
SUGGESTED = [
    "Can an employee approve their own travel expense report?",
    "What is Concur used for?",
    "Where can I find the policy about student education records?",
]


# ---------------------------------------------------------------- backend ----
@st.cache_resource(show_spinner="Loading policy index (first start embeds the corpus, this can take a few minutes)...")
def load_backend():
    if MODE == "live":
        import policy_agent as pa
        pa.init_runtime()
        configured, msg = pa.M4SupportEscalationAgent.configuration_status()
        support = pa.M4SupportEscalationAgent if configured else None
        if support is None:
            import mock_agent
            support = mock_agent.DemoSupport
        return SimpleNamespace(
            run_turn=pa.run_turn, support=support, live=True,
            support_demo=not configured, support_msg=msg, stats=pa.corpus_stats(),
        )
    import mock_agent
    return SimpleNamespace(
        run_turn=mock_agent.run_turn, support=mock_agent.DemoSupport, live=False,
        support_demo=True, support_msg="Demo mode: nothing is sent anywhere.", stats={},
    )


# ------------------------------------------------------------------ state ----
def new_chat():
    st.session_state.thread_id = f"policy-ui-{uuid.uuid4().hex}"
    st.session_state.messages = []
    st.session_state.support = {"phase": "idle"}
    st.session_state.feedback_done = False


if "thread_id" not in st.session_state:
    new_chat()
ss = st.session_state


# --------------------------------------------------------------- rendering ----
def render_answer(text, evidence):
    """Citations become badges: green when they match returned evidence (title + page), else orange."""
    allowed = {(str(e["source"]).casefold(), str(e["page"])) for e in evidence}

    def sub(m):
        title, page = m.group(1).strip(), m.group(2)
        label = f"{title}, p. {page}".replace("[", "(").replace("]", ")")
        if (title.casefold(), page) in allowed:
            return f":green-badge[{label}]"
        return f":orange-badge[{label} - unverified]"

    return CITATION.sub(sub, text or "")


def cited_pairs(text):
    return {(t.strip().casefold(), p) for t, p in CITATION.findall(text or "")}


def render_telemetry(t):
    cost = "unavailable" if t.get("estimated_cost_usd") is None else f"${t['estimated_cost_usd']:.4f}"
    st.caption(
        f"⏱ {t['latency_ms'] / 1000:.1f}s · model calls {t['model_calls']} · tool calls {t['tool_calls']} · "
        f"tokens {t['generation_tokens']} · evidence {t['evidence_count']} · cost {cost}"
    )


def render_message(msg, backend):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.text(msg["content"])
            return
        colour, label = STATUS.get(msg["status"], STATUS["safe_fallback"])
        st.badge(label, color=colour)
        st.markdown(render_answer(msg["content"], msg.get("evidence", [])))
        if msg.get("telemetry"):
            render_telemetry(msg["telemetry"])
        ev = msg.get("evidence", [])
        if ev:
            cited = cited_pairs(msg["content"])
            with st.expander(f"Sources ({len(ev)})"):
                if not backend.live:
                    st.warning("SAMPLE FIXTURE - not real policy text.")
                for e in ev:
                    mark = " · ✔ cited" if (str(e["source"]).casefold(), str(e["page"])) in cited else ""
                    st.markdown(f"**{e['evidence_id']} · {e['source']} · p. {e['page']}**{mark}  \n"
                                f"relevance score {e['score']:.3f}")
                    st.text(e["text"])  # untrusted text: rendered as plain text, never as markdown/HTML


# --------------------------------------------------------- support workflow ----
def last_turn():
    msgs = ss.messages
    if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
        return msgs[-2]["content"], msgs[-1]["content"]
    return None


def open_support_form():
    q, a = last_turn()
    ss.support = {"phase": "form", "ticket_number": None}
    ss.sup_subject = f"Policy question: {q}"[:160]
    ss.sup_desc = f"Question:\n{q}\n\nAnswer I received:\n{a}\n\nWhat I still need:\n"
    ss.sup_consent = False


def attempt_submit(backend, payload):
    """One submission attempt. A reference is minted once and reused on retry (idempotent)."""
    sup = ss.support
    if not sup.get("ticket_number"):
        sup["ticket_number"] = backend.support.new_ticket_number()
    sup["payload"] = payload
    try:
        with st.spinner("Submitting to University Support..."):
            receipt = backend.support.submit_ticket(
                ticket_number=sup["ticket_number"], consent=True, **payload)
        sup.update(phase="done", receipt=receipt, error=None)
    except (PermissionError, ValueError) as exc:      # definitive: nothing was submitted
        sup.update(phase="form", ticket_number=None, error=f"Nothing was submitted. {exc}")
    except Exception as exc:
        if exc.__class__.__name__ == "M4SupportTicketRejected":   # definitive rejection (4xx)
            sup.update(phase="form", ticket_number=None, error=f"Nothing was submitted. {exc}")
        else:                                            # unresolved: keep reference for safe retry
            sup.update(phase="unresolved",
                       error=f"{exc} The request may or may not have been received. "
                             "Retrying reuses the same reference, so it cannot create a duplicate.")
    st.rerun()


def render_support(backend):
    sup = ss.support
    phase = sup["phase"]
    lt = last_turn()

    if phase == "idle":
        if lt and ss.messages[-1]["status"] != "error" and not ss.feedback_done:
            st.write("**Did this answer your question?**")
            c1, c2, _ = st.columns([1, 1.3, 4])
            if c1.button("👍 Yes, this helped", key=f"yes_{len(ss.messages)}"):
                ss.feedback_done = True
                st.rerun()
            if c2.button("👎 No, I still need help", key=f"no_{len(ss.messages)}"):
                open_support_form()
                st.rerun()
        elif ss.feedback_done and lt:
            st.success("Thanks for the feedback.")
        return

    if phase == "form":
        with st.container(border=True):
            st.subheader("Contact University Support")
            st.caption("Only this question and answer are shared, and only if you tick consent and submit. "
                       "The copilot never files a request on its own.")
            if backend.support_demo:
                st.warning("DEMO ONLY - nothing will be sent to University Support.")
            if sup.get("error"):
                st.error(sup["error"])
            with st.form("support_form"):
                st.text_input("Subject", key="sup_subject", max_chars=160)
                st.text_area("Details (you can edit this)", key="sup_desc", height=220, max_chars=10000)
                st.checkbox("I agree to share this question, the answer and my details with University Support.",
                            key="sup_consent")
                c1, c2 = st.columns([1, 1])
                send = c1.form_submit_button("Submit request", type="primary")
                cancel = c2.form_submit_button("Cancel")
            if cancel:
                ss.support = {"phase": "idle"}
                ss.feedback_done = True
                st.rerun()
            if send:
                subject, desc = ss.sup_subject.strip(), ss.sup_desc.strip()
                if not ss.sup_consent:
                    st.error("Please tick the consent box before submitting.")
                elif not subject:
                    st.error("Please add a short subject.")
                elif not desc:
                    st.error("Please add a description.")
                else:
                    q, a = lt
                    attempt_submit(backend, dict(subject=subject, description=desc,
                                                 current_question=q, current_answer=a))
        return

    if phase == "unresolved":
        with st.container(border=True):
            st.warning("We could not confirm that your request was received.")
            st.error(sup["error"])
            st.text_input("Reference (unchanged on retry)", value=sup["ticket_number"], disabled=True)
            st.caption("The form is locked and new questions are paused until this is resolved.")
            if st.button("Retry submission", type="primary"):
                attempt_submit(backend, sup["payload"])
        return

    if phase == "done":
        r = sup["receipt"]
        with st.container(border=True):
            st.success(f"Support request submitted. Reference: **{r['ticket_number']}**")
            if backend.support_demo:
                st.warning("DEMO ONLY - this reference is simulated; no real ticket exists.")
            if st.button("Close"):
                ss.support = {"phase": "idle"}
                ss.feedback_done = True
                st.rerun()


# -------------------------------------------------------------------- main ----
try:
    backend = load_backend()
except Exception as exc:  # fail with a clear, non-secret message
    st.error("The policy agent could not start. Check the configuration in README.md.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.stop()

with st.sidebar:
    st.title("📘 " + APP_TITLE)
    if backend.live:
        st.badge("Live mode", color="green", icon=":material/bolt:")
        st.caption(f"{backend.stats.get('chunks', 0):,} policy chunks · reranker "
                   f"{'on' if backend.stats.get('reranker') else 'off (hybrid ranking only)'}")
    else:
        st.badge("Demo mode", color="orange", icon=":material/science:")
        st.caption("Sample data only. Answers are rule-based, not from a model.")
    st.markdown("**What it does**\n\n- Answers questions from the policy corpus, with citations\n"
                "- Asks for clarification when the question is vague\n"
                "- Drafts a support request that you review and consent to")
    st.markdown("**What it will not do**\n\n- Give legal, medical or personal advice\n"
                "- Answer without evidence - it declines instead\n- Submit anything without your consent")
    st.caption("Each question is limited to 6 tool calls, 4 model calls and 1,500 generated tokens.")
    st.caption(f"Support: {backend.support_msg}")
    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()
    st.caption(f"Thread: `{ss.thread_id[:18]}…`")

st.title(APP_TITLE)
st.caption("Explains policy text and cites its sources. Always check the cited passage; "
           "this is not legal advice.")

pending = None
if not ss.messages:
    st.write("Try a question:")
    for i, q in enumerate(SUGGESTED):
        if st.button(q, key=f"sugg_{i}"):
            pending = q

for m in ss.messages:
    render_message(m, backend)

render_support(backend)

blocked = ss.support["phase"] == "unresolved"
prompt = st.chat_input(
    "Resolve the support request above first" if blocked else "Ask about a university policy…",
    disabled=blocked, max_chars=MAX_QUESTION_CHARS,
) or pending

if prompt:
    prompt = prompt.strip()
    if ss.support["phase"] in ("form", "done"):   # a new question closes the support panel
        ss.support = {"phase": "idle"}
    ss.feedback_done = False
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.text(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching policies and checking the answer…"):
            try:
                res = backend.run_turn(prompt, ss.thread_id)
                ss.messages.append({
                    "role": "assistant", "content": res["answer"], "status": res["status"],
                    "telemetry": res["telemetry"], "evidence": res["evidence"],
                })
            except Exception:
                import logging
                logging.getLogger("policy_ui").exception("agent turn failed")
                ss.messages.append({
                    "role": "assistant", "status": "error", "evidence": [], "telemetry": None,
                    "content": "Sorry, the assistant is unavailable right now. Please try again.",
                })
    st.rerun()
