"""Trace dashboard page: parsing of the notebook's trace file and the rendered page (no network)."""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

STARTUP = hashlib.sha256(b"policy-copilot-demo").hexdigest()[:16]


def rec(minute, status, latency_ms, model=3, tool=3, tokens=200, failure="none_observed", thread="aaaa"):
    return {"started_at_utc": f"2026-11-26T15:{minute:02d}:00+00:00", "thread_id_hash": thread,
            "query_hash": f"q{minute:02d}", "status": status, "failure_category_observed": failure,
            "evidence_count": 5, "tool_calls": tool, "model_calls": model, "generation_tokens": tokens,
            "latency_ms": latency_ms, "estimated_cost_usd": None, "trace_backend": "local_jsonl"}


RECORDS = [
    rec(0, "accepted", 9000, thread=STARTUP),                     # notebook start-up demo turn
    rec(1, "accepted", 12100), rec(2, "accepted", 13500),
    rec(3, "revision_unverified", 16500, model=4, tokens=456),
    rec(4, "clarification_needed", 2100, model=1, tool=1, tokens=40),
    rec(5, "safe_fallback", 10200, model=4, failure="manual_failure_review_required"),
    {"started_at_utc": "2026-11-26T15:06:00+00:00", "thread_id_hash": "aaaa", "query_hash": "q06",
     "status": "runtime_error", "error_type": "APITimeoutError", "latency_ms": 60000,
     "trace_backend": "local_jsonl"},
]


def write_trace(path, literal_backslash_n=True):
    sep = "\\n" if literal_backslash_n else "\n"      # the notebook currently writes a literal \n
    path.write_text(sep.join(json.dumps(r) for r in RECORDS) + sep, encoding="utf-8")


def page(tmp_path, monkeypatch, **kw):
    trace = tmp_path / "trace.jsonl"
    write_trace(trace, **kw)
    monkeypatch.setenv("TRACE_FILE", str(trace))
    at = AppTest.from_file(str(ROOT / "trace_dashboard.py"), default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_parser_handles_literal_and_real_newlines(tmp_path):
    import trace_data as p
    for literal in (True, False):
        f = tmp_path / f"t{literal}.jsonl"
        write_trace(f, literal_backslash_n=literal)
        recs, bad = p.parse_trace_text(f.read_text())
        assert len(recs) == len(RECORDS) and bad == 0
    recs, bad = p.parse_trace_text('{"status": "accepted"}\nnot json\n')
    assert len(recs) == 1 and bad == 1


def test_kpis_exclude_startup_check_by_default(tmp_path, monkeypatch):
    m = metrics(page(tmp_path, monkeypatch))
    assert m["Questions traced"] == "6"                        # 7 records minus the start-up turn
    assert m["Accepted by critic"] == "33%"                    # 2 of 6
    assert m["Declined (could not verify)"] == "17%"
    assert m["Runtime errors"] == "1"
    assert m["Median latency"] == "12.8 s"                     # median of 2.1,10.2,12.1,13.5,16.5,60.0


def test_page_renders_charts_table_and_download(tmp_path, monkeypatch):
    at = page(tmp_path, monkeypatch, literal_backslash_n=False)
    assert len(at.get("vega_lite_chart")) >= 5           # outcomes, failures, latency, 3 budgets
    assert at.dataframe and len(at.dataframe[0].value) == 6
    assert at.get("download_button")
    assert any("budgets " in c.value for c in at.caption)


def test_include_startup_check_toggle(tmp_path, monkeypatch):
    at = page(tmp_path, monkeypatch)
    at.checkbox[0].check().run()
    assert metrics(at)["Questions traced"] == "7"


def test_empty_trace_shows_help(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACE_FILE", str(tmp_path / "missing.jsonl"))
    at = AppTest.from_file(str(ROOT / "trace_dashboard.py"), default_timeout=30).run()
    assert not at.exception
    assert "No traces yet" in at.info[0].value


def test_falls_back_to_committed_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACE_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "ui_adapter", None)   # no live agent in this process
    (tmp_path / "data").mkdir()
    write_trace(tmp_path / "data" / "trace_snapshot.jsonl")
    monkeypatch.chdir(tmp_path)                      # no live trace file here, only the snapshot
    at = AppTest.from_file(str(ROOT / "trace_dashboard.py"), default_timeout=30).run()
    assert not at.exception
    assert any("trace_snapshot.jsonl" in c.value for c in at.caption)
    assert metrics(at)["Questions traced"] == "6"
