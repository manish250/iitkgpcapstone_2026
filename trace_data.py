"""Reading the agent notebook's per-query trace file (UI team code, read-only)."""
import hashlib
import json
import os
import re

import pandas as pd

DEFAULT_TRACE = os.environ.get("TRACE_FILE", "policy_copilot_m4_trace.jsonl")
STARTUP_THREAD_HASH = hashlib.sha256(b"policy-copilot-demo").hexdigest()[:16]  # notebook's start-up demo turn
DEFAULT_BUDGET = {"tool_calls": 6, "model_calls": 4, "generation_tokens": 1500}  # notebook M4 limits

STATUS_LABELS = {
    "accepted": "Accepted by critic",
    "revision_unverified": "Revised (agent status: unverified)",
    "clarification_needed": "Asked for clarification",
    "safe_fallback": "Declined (could not verify)",
    "runtime_error": "Runtime error",
}
FAILURE_LABELS = {
    "none_observed": "None",
    "budget_guard": "Budget limit reached",
    "malformed_model_output": "Malformed model output",
    "tool_or_retrieval_error": "Tool or retrieval error",
    "model_api_error": "Model API error",
    "revision_requires_semantic_review": "Revision needs review",
    "manual_failure_review_required": "Needs manual review",
}


def parse_trace_text(text):
    """Records separated by real newlines OR by the literal backslash-n the notebook writes."""
    records, bad = [], 0
    for chunk in re.split(r"\r?\n|\\n(?=\s*\{)", text or ""):
        chunk = chunk.strip()
        while chunk.endswith("\\n"):                     # literal backslash-n after the last record
            chunk = chunk[:-2].rstrip()
        if not chunk:
            continue
        try:
            rec = json.loads(chunk)
            if isinstance(rec, dict):
                records.append(rec)
            else:
                bad += 1
        except json.JSONDecodeError:
            bad += 1
    return records, bad


def to_frame(records):
    cols = ["started_at_utc", "status", "failure_category_observed", "latency_ms", "model_calls",
            "tool_calls", "generation_tokens", "evidence_count", "thread_id_hash", "query_hash", "error_type"]
    df = pd.DataFrame(records)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    df["time"] = pd.to_datetime(df["started_at_utc"], utc=True, errors="coerce")
    for c in ["latency_ms", "model_calls", "tool_calls", "generation_tokens", "evidence_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["latency_s"] = df["latency_ms"] / 1000.0
    df["status"] = df["status"].fillna("unknown")
    df["outcome"] = df["status"].map(STATUS_LABELS).fillna(df["status"])
    df["failure"] = df["failure_category_observed"].map(FAILURE_LABELS).fillna(
        df["failure_category_observed"].fillna("—"))
    df.loc[df["status"] == "runtime_error", "failure"] = "Runtime error: " + df["error_type"].fillna("unknown")
    df["startup_check"] = df["thread_id_hash"] == STARTUP_THREAD_HASH
    return df.sort_values("time", na_position="first").reset_index(drop=True)
