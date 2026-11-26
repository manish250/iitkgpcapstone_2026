# University Policy Copilot - Streamlit UI

The UI team's Streamlit front end for the agent team's notebook
(`University_Policy_Copilot_TrackB.ipynb`: planner -> retrieval -> draft -> critic -> revise -> finalize,
LangGraph + SQLite memory). It follows `Policy_Copilot_UI_Technical_Summary.pdf`.

## Who owns what
| File | Owner | Purpose |
|---|---|---|
| `University_Policy_Copilot_TrackB.ipynb` | Agent team | The notebook, unchanged |
| `policy_agent.py` | Agent team | **Verbatim** copy of the notebook's agent cells (do not edit) |
| `ui_adapter.py` | UI team | Runs `policy_agent.py`, answers its prompts from env vars, returns answer + sources for the UI, related questions |
| `app.py` | UI team | Chat, status badges, citation chips, Sources, telemetry, related questions, feedback, consent-gated support, rate limits |
| `limits.py`, `mock_agent.py` | UI team | Rate limiter; rule-based demo backend (sample data only) |
| `tests/` | UI team | Verbatim check, agent code run against fake servers, UI tests |

If the agent misbehaves, report it to the agent team; the UI does not patch agent code.

### What `policy_agent.py` contains
25 notebook cells copied character for character, in notebook order (each block is labelled with its cell
number): setup and telemetry, LLM client, corpus, cleaning, chunking, embeddings + hybrid search, reranker,
M4 state/tools/planner/executor/draft/critic/finalize/revise, graph + checkpointer, `run_m4_persistent_turn`,
support escalation, final graph compile. From two evaluation cells only the helper definitions
`_policy_members_from_doc` and `_policy_title_from_doc` are copied (the rest of those cells runs evaluations).
Not copied: pip installs, evaluations, chunking experiments, widget UIs, mock-support demos, Track B reports.

Starting the app runs that code once, exactly as the notebook would, including its built-in checks: an LLM
connection test, a planner check, and one demo turn ("What is concur?").

### What `ui_adapter.py` adds (UI side only)
- Answers the notebook's `input()` certificate prompt with `LLM_CA_CERT`; refuses `getpass()` with a clear
  error if a key is missing (a server has no keyboard).
- Calls the notebook's `run_m4_persistent_turn()` and reads that turn's evidence (top 5 sources) from the
  graph checkpoint.
- Shows a critic-revised answer as **"Revised to match cited policy"** with its sources. (The notebook's
  finalize publishes revised answers, while its status function still labels them `revision_unverified`.)
- Shows an answer whose first sentence says the policies do not cover the question ("I could not find...")
  as **"Not found in the policies"**: its citations turn grey ("checked: ..."), the panel is titled
  "Passages searched", and no related questions are offered.
- Related questions: one small extra model call after an answer, outside the agent graph
  (`FOLLOWUP_SUGGESTIONS=off` to disable).

## Google Colab
Upload the zip and `Policy_Copilot_Colab_Launcher.ipynb`, run the cells top to bottom (certificates, two keys,
open the link; use the Colab-proxy cell if the Cloudflare link fails).

## Demo mode (no keys)
```bash
pip install streamlit python-dotenv && streamlit run app.py
```

## Live mode locally
```bash
pip install -r requirements.txt
cp .env.example .env     # set POLICY_AGENT_MODE=live and the keys; put data/vm02-ca.crt and the LLM cert in place
streamlit run app.py
```
First start embeds about 3.4k chunks (the notebook code has no embedding cache), then the index stays in
memory for the life of the server process.

## Rate limiting (UI)
Per client: `CHAT_RPM` questions/minute, `TICKET_RPM` support submissions/minute; `MAX_CONCURRENT_TURNS`
caps simultaneous turns.

## Tests
```bash
pip install -r requirements-dev.txt && python -m pytest -q tests
```
