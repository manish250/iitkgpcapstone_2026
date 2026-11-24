# University Policy Copilot - Streamlit UI

A Streamlit front end for the agent in the milestone-4 notebook (planner -> retrieval -> draft ->
critic -> finalize, LangGraph + SQLite memory). It follows the behaviour in
`Policy_Copilot_UI_Technical_Summary.pdf`.

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, status badges, citation chips, Sources, telemetry, consent-gated support flow |
| `policy_agent.py` | The notebook's agent code, extracted and cleaned (env config, fail-closed TLS, no prompts/demos) |
| `mock_agent.py` | Rule-based demo backend + demo support service (sample data only) |
| `tests/` | 3 graph tests (fake LLM) + 9 UI tests |

## 0. Google Colab (recommended)
Upload `policy-copilot-colab.zip` to Colab, open `Policy_Copilot_Colab_Launcher.ipynb`, run the cells top to bottom
(upload 2 certificates, type 2 keys, open the printed link). Data is already in `data/`.

## 1. Try it now (demo mode, no keys)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install streamlit python-dotenv
streamlit run app.py
```
Demo rules: fewer than 3 words -> clarification; a question with *expense, travel, approve, reimburs, concur*
-> verified sample answer; anything else -> safe fallback. Sample data is labelled in the UI.

## 2. Run the real agent
```bash
pip install -r requirements.txt
cp .env.example .env        # fill in keys; put the CA .crt files in `certs/` (the policy ZIP is already in `data/`)
# set POLICY_AGENT_MODE=live in .env
streamlit run app.py
```
First start embeds roughly 3.4k chunks through the embedding API (a few minutes). The index is then cached
for the life of the server process (`st.cache_resource`). Conversation memory persists per `thread_id` in
`M4_CHECKPOINT_DB_PATH`.

Support requests use the notebook's `M4SupportEscalationAgent` when `UNIVERSITY_SUPPORT_API_URL` and
`_TOKEN` are set; otherwise they run as clearly marked **DEMO ONLY** (nothing is sent).

## 3. Tests
```bash
pip install -r requirements-dev.txt && python -m pytest -q tests
```

## Related questions
After each **verified** answer the UI offers three related questions as buttons (click one to ask it).
In live mode they come from one small extra model call (max 150 tokens) grounded in the policies just
retrieved; it sits outside the bounded graph, so it is not counted in the per-query call/token limits.
Set `FOLLOWUP_SUGGESTIONS=off` to disable it. Clarifications and fallbacks never show suggestions.

## Retries
- **Embedding calls** (index build and each question): up to 3 retries with exponential backoff (~1s, 2s, 4s,
  jittered) for timeouts, connection errors and HTTP 408/425/429/5xx; `Retry-After` is honoured (max 30s).
  Certificate errors and other 4xx (e.g. 401/403) are **never** retried. Set `EMBEDDING_MAX_RETRIES`.
- **LLM calls:** retried by the OpenAI client (default 2). Set `LLM_MAX_RETRIES`.
- **Inside the agent graph there are no extra retries** - the notebook's call budget stays intact; a failed
  step degrades to a safe fallback. Related-question suggestions are not retried.

## Behaviour carried over from the notebook / PDF
- Sources are shown only for `accepted` answers; clarifications and fallbacks never look evidence-backed.
- Citation chips: green only if title + page match returned evidence, otherwise orange "unverified".
- Retrieved text is shown as plain text, never as markdown/HTML.
- Support: nothing is sent without consent; only the current question/answer are shared; the SR reference is
  created once and reused on retry (idempotent); a definitive rejection reopens the form with a fresh reference;
  chat is blocked while a submission is unresolved.
- Cost shows "unavailable" (no pricing configured in the notebook).

## Known limits (see PDF section 5 before real users)
Streamlit has no built-in login: put it behind SSO / a reverse proxy. Thread ownership is per browser session.
The idempotency reference lives in session state, and the vector index and checkpoint are per-process.
The notebook's gold evaluation set is a synthetic fallback, so answer quality has not been measured.
