"""Trace dashboard (Track B): what the agent did on every question it answered.

Reads the per-query trace the agent notebook writes in run_m4_persistent_turn()
(policy_copilot_m4_trace.jsonl). The trace holds hashes, not question text.
UI team code: it only reads the file, it never changes agent behaviour.
"""
import io
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from trace_data import DEFAULT_BUDGET, DEFAULT_TRACE, parse_trace_text, to_frame

BLUE = "#257de0"   # single sequential hue (validated)
INK_2 = "#666666"  # secondary ink for reference rules and notes
SNAPSHOT = "data/trace_snapshot.jsonl"   # traces copied from a live run, committed for a hosted demo


def budgets():
    """The notebook's limits if the agent is loaded in this process, else the notebook's defaults."""
    agent = getattr(sys.modules.get("ui_adapter"), "_agent", None)
    if agent is None:
        return dict(DEFAULT_BUDGET), "notebook defaults"
    return ({"tool_calls": getattr(agent, "M4_MAX_TOOL_CALLS", 6),
             "model_calls": getattr(agent, "M4_MAX_LLM_CALLS", 4),
             "generation_tokens": getattr(agent, "M4_MAX_TOTAL_GENERATION_TOKENS", 1500)},
            "read from the running agent")


def default_trace_path():
    agent = getattr(sys.modules.get("ui_adapter"), "_agent", None)
    if os.environ.get("TRACE_FILE"):                 # read per run, so a changed setting applies
        return os.environ["TRACE_FILE"]
    live = str(getattr(agent, "M4_TRACE_FILE", DEFAULT_TRACE))
    if not Path(live).is_file() and Path(SNAPSHOT).is_file():
        return SNAPSHOT                              # e.g. a hosted copy with a committed snapshot
    return live


# ----------------------------------------------------------------- charts ----
def ranked_bars(counts, label_col, title):
    """Ranked horizontal bars, one hue, count printed at the end of each bar."""
    data = counts.reset_index()
    data.columns = [label_col, "Questions"]
    data = data.sort_values("Questions", ascending=False)
    order = list(data[label_col])
    total = max(int(data["Questions"].sum()), 1)
    data["Share"] = (data["Questions"] / total).map(lambda v: f"{v:.0%}")
    y = alt.Y(f"{label_col}:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=260, ticks=False, domain=False))
    x = alt.X("Questions:Q", title="Questions", axis=alt.Axis(tickMinStep=1),
              scale=alt.Scale(domain=[0, float(data["Questions"].max()) * 1.15]))
    tip = [alt.Tooltip(f"{label_col}:N"), alt.Tooltip("Questions:Q"), alt.Tooltip("Share:N")]
    bars = alt.Chart(data).mark_bar(color=BLUE, cornerRadiusEnd=4, height=18).encode(y=y, x=x, tooltip=tip)
    text = alt.Chart(data).mark_text(align="left", dx=6).encode(y=y, x=x, text="Questions:Q")
    return alt.layer(bars, text).properties(title=title, height=alt.Step(34))


def latency_chart(df):
    df = df.assign(when=df["time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC"))
    p95 = float(df["latency_s"].quantile(0.95))
    x = alt.X("time:T", title="Time (UTC)", scale=alt.Scale(type="utc"),
              axis=alt.Axis(format="%H:%M", labelAngle=0, tickCount=6, labelOverlap=True))
    y = alt.Y("latency_s:Q", title="Seconds")
    line = alt.Chart(df).mark_line(color=BLUE, strokeWidth=2).encode(x=x, y=y)
    points = alt.Chart(df).mark_circle(color=BLUE, size=64, opacity=1).encode(x=x, y=y, tooltip=[
        alt.Tooltip("when:N", title="Time"),
        alt.Tooltip("outcome:N", title="Outcome"),
        alt.Tooltip("latency_s:Q", title="Latency (s)", format=".1f"),
        alt.Tooltip("model_calls:Q", title="Model calls"),
        alt.Tooltip("tool_calls:Q", title="Tool calls"),
        alt.Tooltip("generation_tokens:Q", title="Tokens"),
    ])
    rule_df = pd.DataFrame({"p95": [p95], "label": [f"p95 {p95:.1f}s"]})
    rule = alt.Chart(rule_df).mark_rule(color=INK_2, strokeDash=[4, 4]).encode(y="p95:Q")
    rule_text = alt.Chart(rule_df).mark_text(color=INK_2, align="left", dy=-6, x=4).encode(
        y="p95:Q", text="label:N")
    return alt.layer(line, points, rule, rule_text).properties(title="Latency per question", height=260)


def budget_chart(df, col, title, limit, binned=False):
    """Distribution of one resource per question, with the budget as a dashed rule."""
    tip_count = alt.Tooltip("count():Q", title="Questions")
    if binned:
        top = max(float(limit), float(df[col].max() or 0))
        step = max(25, int(round(top / 15 / 25.0)) * 25)
        x = alt.X(f"{col}:Q", bin=alt.Bin(extent=[0, top + step], step=step), title=title,
                  axis=alt.Axis(labelAngle=0))
        tips = [alt.Tooltip(f"{col}:Q", bin=alt.Bin(extent=[0, top + step], step=step), title=title), tip_count]
    else:
        values = list(range(0, int(max(limit, df[col].max() or 0)) + 1))
        x = alt.X(f"{col}:O", title=title, scale=alt.Scale(domain=values), axis=alt.Axis(labelAngle=0))
        tips = [alt.Tooltip(f"{col}:Q", title=title), tip_count]
    bars = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x, y=alt.Y("count():Q", title="Questions", axis=alt.Axis(tickMinStep=1)), tooltip=tips)
    layers = [bars]
    if binned:
        rule_df = pd.DataFrame({"limit": [limit], "label": [f"budget {limit}"]})
        layers += [alt.Chart(rule_df).mark_rule(color=INK_2, strokeDash=[4, 4]).encode(x="limit:Q"),
                   alt.Chart(rule_df).mark_text(color=INK_2, align="right", dx=-4, y=8).encode(
                       x="limit:Q", text="label:N")]
    return alt.layer(*layers).properties(height=200, title=f"{title} (budget {limit})")


# ------------------------------------------------------------------- page ----
st.title("Trace dashboard")
st.caption("What the agent did on every question: outcome, latency, model and tool calls, tokens and failures. "
           "Source: the agent notebook's per-query trace (`run_m4_persistent_turn`). "
           "It stores hashes, never question or answer text.")

with st.expander("Trace source", expanded=False):
    path = st.text_input("Trace file on the server", value=default_trace_path())
    upload = st.file_uploader("…or upload a trace file (.jsonl) from a notebook run", type=["jsonl", "txt", "json"])

if upload is not None:
    text, source = upload.getvalue().decode("utf-8", errors="replace"), upload.name
elif Path(path).is_file():
    text, source = Path(path).read_text(encoding="utf-8", errors="replace"), path
else:
    text, source = "", path

records, bad = parse_trace_text(text)
if not records:
    st.info(f"No traces yet in `{source}`. Ask a question on the **Chat** page (live mode); every question "
            "the agent answers adds one record. You can also upload a trace file from a notebook run.")
    st.stop()

df_all = to_frame(records)
limits, limits_src = budgets()

# filters: one row above the charts
f1, f2, f3 = st.columns([1.2, 2.6, 1.4])
window = f1.selectbox("Time range", ["All", "Last hour", "Last 24 hours", "Last 7 days"])
outcomes = sorted(df_all["outcome"].unique())
picked = f2.multiselect("Outcome", outcomes, default=outcomes)
include_startup = f3.checkbox("Include start-up check", value=False,
                              help="The agent notebook asks 'What is concur?' once each time the app starts.")

df = df_all[df_all["outcome"].isin(picked)]
if not include_startup:
    df = df[~df["startup_check"]]
if window != "All" and df["time"].notna().any():
    span = {"Last hour": "1h", "Last 24 hours": "24h", "Last 7 days": "7D"}[window]
    df = df[df["time"] >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(span)]

st.caption(f"Source: `{source}` · {len(df_all)} records"
           + (f" · {bad} unreadable lines skipped" if bad else "")
           + f" · showing {len(df)} · budgets {limits_src}")
if df.empty:
    st.warning("No questions match these filters.")
    st.stop()

n = len(df)
lat = df["latency_s"].dropna()
within = ((df["tool_calls"] <= limits["tool_calls"]) & (df["model_calls"] <= limits["model_calls"])
          & (df["generation_tokens"] <= limits["generation_tokens"]))
share = lambda mask: f"{mask.mean():.0%}" if n else "–"

k = st.columns(4)
k[0].metric("Questions traced", f"{n}")
k[1].metric("Accepted by critic", share(df["status"] == "accepted"))
k[2].metric("Median latency", f"{lat.median():.1f} s" if len(lat) else "–")
k[3].metric("Within budget", share(within),
            help=f"≤ {limits['tool_calls']} tool calls, ≤ {limits['model_calls']} model calls and "
                 f"≤ {limits['generation_tokens']} generated tokens")
k = st.columns(4)
k[0].metric("Declined (could not verify)", share(df["status"] == "safe_fallback"))
k[1].metric("p95 latency", f"{lat.quantile(0.95):.1f} s" if len(lat) else "–")
k[2].metric("Avg model calls", f"{df['model_calls'].mean():.1f}" if df["model_calls"].notna().any() else "–")
k[3].metric("Runtime errors", f"{int((df['status'] == 'runtime_error').sum())}")

c1, c2 = st.columns(2)
with c1:
    st.altair_chart(ranked_bars(df["outcome"].value_counts(), "Outcome", "Outcomes"), width="stretch")
with c2:
    fails = df.loc[df["failure"] != "None", "failure"].value_counts()
    if fails.empty:
        st.markdown("**Failure categories**")
        st.success("No failures observed in these questions.")
    else:
        st.altair_chart(ranked_bars(fails, "Failure", "Failure categories"), width="stretch")

if lat.notna().any():
    st.altair_chart(latency_chart(df.dropna(subset=["time", "latency_s"])), width="stretch")

b1, b2, b3 = st.columns(3)
with b1:
    st.altair_chart(budget_chart(df.dropna(subset=["model_calls"]), "model_calls", "Model calls",
                                 limits["model_calls"]), width="stretch")
with b2:
    st.altair_chart(budget_chart(df.dropna(subset=["tool_calls"]), "tool_calls", "Tool calls",
                                 limits["tool_calls"]), width="stretch")
with b3:
    st.altair_chart(budget_chart(df.dropna(subset=["generation_tokens"]), "generation_tokens",
                                 "Generated tokens", limits["generation_tokens"], binned=True),
                    width="stretch")

st.subheader("Recent questions")
table = df.sort_values("time", ascending=False).head(200)[[
    "time", "outcome", "failure", "latency_s", "model_calls", "tool_calls", "generation_tokens",
    "evidence_count", "thread_id_hash", "query_hash"]].rename(columns={
        "time": "Time (UTC)", "outcome": "Outcome", "failure": "Failure", "latency_s": "Latency (s)",
        "model_calls": "Model calls", "tool_calls": "Tool calls", "generation_tokens": "Tokens",
        "evidence_count": "Sources", "thread_id_hash": "Conversation (hash)", "query_hash": "Question (hash)"})
st.dataframe(table, hide_index=True, width="stretch",
             column_config={"Latency (s)": st.column_config.NumberColumn(format="%.1f")})
buf = io.StringIO()
df.drop(columns=["startup_check"]).to_csv(buf, index=False)
st.download_button("Download filtered traces (CSV)", buf.getvalue(), "policy_copilot_traces.csv", "text/csv")
st.caption("Trace statuses are the agent's own. The Chat page may show a published revised answer as "
           "\"Revised to match cited policy\" and some answers as \"Not found in the policies\"; those "
           "UI labels are not in the trace.")
