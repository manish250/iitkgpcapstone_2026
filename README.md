# University Policy Copilot — Streamlit UI

A chat interface for the University Policy Copilot. Users ask questions about university policy and get
answers with citations, the source passages, related questions, and a consent-gated support request.
The UI runs on top of the agent team's notebook code (`University_Policy_Copilot_TrackB.ipynb`):
planner → retrieval → draft → critic → revise → finalize, built with LangGraph and SQLite memory.

---

## 1. What is in this repo

| File | Owner | Purpose |
|---|---|---|
| `app.py` | UI team | Entry point: sidebar navigation between the two pages |
| `chat_page.py` | UI team | Chat: status badges, citation chips, sources panel, related questions, feedback, support request, rate limits |
| `trace_dashboard.py`, `trace_data.py` | UI team | Track B trace dashboard built from the agent's per-query trace file |
| `ui_adapter.py` | UI team | Connects the UI to the agent code; answers the notebook's prompts from environment variables |
| `limits.py` | UI team | Per-user rate limiting and a cap on simultaneous questions |
| `mock_agent.py` | UI team | Demo mode with sample data (no model or keys needed) |
| `Policy_Copilot_Colab_Launcher.ipynb` | UI team | Runs everything in Google Colab |
| `requirements.txt` / `requirements-agent.txt` | UI team | UI-only packages (enough for demo mode and hosting) / the agent notebook's packages (live mode) |
| `tests/` | UI team | UI tests, rate-limit tests, and a check that `policy_agent.py` matches the notebook |
| `policy_agent.py` | Agent team | **Verbatim copy** of the notebook's agent cells. Do not edit. |
| `University_Policy_Copilot_TrackB.ipynb` | Agent team | The agent notebook, unchanged |
| `data/` | Shared | Policy corpus ZIP and gold Q&A sets |

> **Agent problems go to the agent team.** The UI never changes agent code. If an answer is wrong,
> report the question and the app log to the agent team.

---

## 2. What you need

- A Google account (for Colab), **or** Python 3.10+ on your own machine
- The two CA certificates: **`vm05-ca.crt`** (LLM server) and **`vm02-ca.crt`** (embedding server)
- The two shared keys: the **group LLM API key** and the **embedding API key**

Never commit certificates or keys. `.gitignore` already excludes `*.crt`, `.env` and `*.sqlite`.

---

## 3. Run in Google Colab (recommended)

### Option A: from the project zip
1. Open [colab.research.google.com](https://colab.research.google.com) and upload
   `Policy_Copilot_Colab_Launcher.ipynb` (**File → Upload notebook**).
2. In the left **Files** panel, upload the project zip (`policy-copilot-colab_v9.zip`) to `/content`.
   Do not unzip it yourself.
3. Run the cells in order:

| Cell | What it does | What you do |
|---|---|---|
| 1 | Unpacks the newest zip, checks the version, installs packages (2–3 min) | Check it prints `Project version: Build ... (v9)` |
| 2 | Asks for certificates and copies them to where the notebook expects them | Select **both** `vm05-ca.crt` and `vm02-ca.crt` |
| 3 | Sets configuration | Type the LLM key, then the embedding key |
| 4 | Starts the app and opens a public link | Open the printed link |

4. The first start takes a few minutes. The app runs the agent notebook code once: an LLM connection
   check, embedding about 3,400 policy chunks, loading the reranker, and one demo question. Wait until
   the chat box appears.

### Option B: from this Git repo
```python
!git clone -b <your-branch> https://github.com/<owner>/<repo>.git
%cd <repo>/<folder-with-app.py>
```
Then open `Policy_Copilot_Colab_Launcher.ipynb` and run cells 1–4. With no zip in `/content`, cell 1
uses the current folder.

### If the link does not open
- **`Tunnel did not start` / `status 429`**: Cloudflare is limiting free tunnels from Colab. Run the
  **"Alternative: Colab's own proxy"** cell instead; it opens the app inside Colab.
- **`DNS_PROBE_FINISHED_NXDOMAIN`**: the new link is not live yet. Wait a minute and reload, or use the
  proxy cell.
- **App stuck on "Loading"** or shows an error: run the **Logs** cell and read the end of the log.

### Stop
Run the **Stop everything** cell, or use **Runtime → Disconnect and delete runtime**.

### Updating to a new version
Use **Runtime → Disconnect and delete runtime**, upload the new zip and launcher, and run again.
Cell 1 stops with an "OLD copy" message if it finds an older project.

---

## 4. Run on your own machine

### Demo mode (no keys, sample data)
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install streamlit python-dotenv
streamlit run app.py
```
Open http://localhost:8501. The sidebar shows **Demo mode**. Answers are rule-based sample data.

### Live mode (real agent)
```bash
pip install -r requirements.txt -r requirements-agent.txt
cp .env.example .env
```
Edit `.env`:
```
POLICY_AGENT_MODE=live
OPENAI_API_KEY=<group LLM key>
EMBEDDINGS_KEY=<embedding key>
LLM_CA_CERT=vm05-ca.crt
```
Put the certificates where the notebook code looks for them:
- `data/vm02-ca.crt` (embedding server)
- `vm05-ca.crt` in the project folder (the path given in `LLM_CA_CERT`)

Then run `streamlit run app.py`. The first start embeds the whole corpus and takes a few minutes.

> Endpoints, model name and budgets are set inside the agent notebook code, not in `.env`.

---

## 5. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `POLICY_AGENT_MODE` | `mock` | `mock` = sample data, `live` = real agent |
| `OPENAI_API_KEY` | – | Group LLM key (read by the notebook code) |
| `EMBEDDINGS_KEY` | – | Embedding API key (read by the notebook code) |
| `LLM_CA_CERT` | `vm05-ca.crt` | Answer the UI gives to the notebook's certificate prompt |
| `M4_CHECKPOINT_DB_PATH` | `./policy_agent_checkpoints.sqlite` | Where conversation memory is stored |
| `FOLLOWUP_SUGGESTIONS` | `on` | `off` hides related questions (saves one model call per answer) |
| `CHAT_RPM` | `20` | Questions per minute per user |
| `TICKET_RPM` | `5` | Support submissions per minute per user |
| `MAX_CONCURRENT_TURNS` | `4` | Questions processed at the same time; others wait up to 30 s |
| `TRUST_PROXY` | `false` | Set `true` only behind a proxy you control |
| `UNIVERSITY_SUPPORT_API_URL` / `_TOKEN` | – | Real ticketing. If unset, support requests are **DEMO ONLY** |

---

## 6. What the UI shows

| Badge | Meaning | Sources | Related questions |
|---|---|---|---|
| 🟢 **Verified against policy** | The critic accepted the answer | Sources (top 5) | Yes |
| 🔵 **Revised to match cited policy** | Accepted after one revision; citations match the sources | Sources (top 5) | Yes |
| ⚪ **Not found in the policies** | The agent says the policies do not answer this | "Passages searched", grey citations | No |
| 🔵 **Needs more information** | The agent asks a clarifying question | – | No |
| 🟠 **Could not verify** | No answer could be supported | – | No |
| 🔴 **Something went wrong** | The turn failed; see the log | – | No |

Citation chips are green when they match a returned source and orange ("unverified") when they do not.
Under each answer: time taken, model calls, tool calls and tokens.

**Feedback and support:** "No, I still need help" opens an editable support request. Nothing is sent
until the user ticks the consent box. Without a configured support API the reference is simulated and
labelled **DEMO ONLY**.

---

## 7. Trace dashboard (Track B)

Open **Trace dashboard** in the app's sidebar. It reads the trace the agent notebook writes for every
question (`policy_copilot_m4_trace.jsonl`, from `run_m4_persistent_turn`), which stores hashes, never
question or answer text.

- **Headline numbers:** questions traced, accepted by critic, declined, median and p95 latency, average
  model calls, share within budget (≤ 6 tool calls, ≤ 4 model calls, ≤ 1,500 tokens), runtime errors
- **Charts:** outcomes, failure categories, latency per question (with p95 line), and model calls, tool
  calls and tokens per question against their budgets. Hover any mark for details.
- **Table** of recent questions and a **CSV download**
- **Filters:** time range, outcome, and whether to include the notebook's start-up check ("What is concur?")
- **Trace source:** the app's own trace file by default; you can also upload a trace file from a notebook
  run, or set `TRACE_FILE`.

The trace only fills in live mode (one record per answered question).

## 8. Permanent link (Streamlit Community Cloud)

A free, permanent `https://<name>.streamlit.app` link for your report. It runs in **demo mode** (no keys or
certificates on the internet) and the Trace dashboard shows a snapshot of traces from a real live run.

1. **Save real traces.** In Colab, after asking several questions in live mode, download
   `/content/policy-copilot-colab/policy_copilot_m4_trace.jsonl` (Files panel → ⋮ → Download).
2. **Add it to the repo** as `data/trace_snapshot.jsonl` (it holds hashes and numbers only):
   ```bash
   git add -f data/trace_snapshot.jsonl     # -f because .gitignore excludes *.jsonl
   git commit -m "Add trace snapshot for the hosted dashboard"
   git push
   ```
3. Go to **share.streamlit.io**, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
4. Choose your **repository**, **branch**, and **main file path** (`app.py`, or e.g. `ui/app.py` if the app
   is in a subfolder). `requirements.txt` must be in the same folder as `app.py` or in the repo root.
5. Optional: pick a custom **App URL**. Click **Deploy**. The first build takes a few minutes.
6. Put the link in your report. Pushing new commits updates the app automatically.

Free apps go to sleep after a period without visitors; the first visitor wakes it (about a minute).

## 9. Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```
- `test_app_mock.py`: UI behaviour in demo mode
- `test_limits.py`: rate limiter and concurrency cap
- `test_trace_dashboard.py`: trace parsing and the dashboard page
- `test_live_graph_fake.py`: runs the real `policy_agent.py` on the real corpus with fake LLM, embedding
  and reranker servers. It also checks that `policy_agent.py` still matches the notebook exactly.

No keys or network are needed for the tests.

---

## 10. When the agent team updates the notebook

1. Replace `University_Policy_Copilot_TrackB.ipynb` with the new version.
2. Copy the changed agent cells into `policy_agent.py`, unchanged, keeping the
   `# ======== notebook cell N ... ========` banners.
3. Run the tests. `test_policy_agent_is_verbatim_notebook_code` fails if a copied block differs.

---

## 11. Known issues to report to the agent team

- `_m4_status_from_state` labels published revised answers `revision_unverified`. The UI shows them as
  "Revised to match cited policy".
- `run_m4_persistent_turn()` does not return the evidence. The UI reads it from the graph checkpoint.
- A "not found" answer can still carry a citation. The UI shows it grey as "checked".
- `_m4_append_local_trace` writes a literal backslash-n (`"\\n"` in the code) instead of a newline, so
  the trace file is one long line, not valid JSONL. The dashboard reads it anyway; the fix is to use `"\n"`.
- Each app start makes about 5 model calls (the notebook's built-in checks and demo turn) and re-embeds
  the whole corpus, because the notebook has no embedding cache.
