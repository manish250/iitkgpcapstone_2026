"""Retry behaviour of the embedding client (no network)."""
import os, sys, tempfile
from pathlib import Path
os.environ["M4_CHECKPOINT_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "cp.sqlite")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest, requests
import policy_agent as pa


class Resp:
    def __init__(self, status=200, headers=None):
        self.status_code, self.headers = status, headers or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")
    def json(self):
        return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}


def run(monkeypatch, outcomes, retries=3):
    """outcomes: list of Resp or Exception, consumed one per POST."""
    calls, sleeps = [], []
    def fake_post(*a, **k):
        calls.append(1)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(pa.requests, "post", fake_post)
    monkeypatch.setattr(pa.time, "sleep", lambda d: sleeps.append(d))
    monkeypatch.setattr(pa, "EMBEDDING_MAX_RETRIES", retries)
    emb = pa.CapstoneEmbeddings("https://x/v1", "m", "key")
    return emb, calls, sleeps


def test_retries_503_then_succeeds(monkeypatch):
    emb, calls, sleeps = run(monkeypatch, [Resp(503), Resp(502), Resp(200)])
    assert emb.embed_query("hi") == [0.1, 0.2]
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9


def test_retries_timeout_and_connection_errors(monkeypatch):
    emb, calls, _ = run(monkeypatch, [requests.exceptions.Timeout(),
                                      requests.exceptions.ConnectionError(), Resp(200)])
    assert emb.embed_query("hi") and len(calls) == 3


def test_honours_retry_after(monkeypatch):
    emb, _, sleeps = run(monkeypatch, [Resp(429, {"Retry-After": "7"}), Resp(200)])
    emb.embed_query("hi")
    assert sleeps == [7.0]


def test_gives_up_after_max_retries(monkeypatch):
    emb, calls, _ = run(monkeypatch, [Resp(503)] * 10, retries=2)
    with pytest.raises(requests.exceptions.HTTPError):
        emb.embed_query("hi")
    assert len(calls) == 3                      # 1 try + 2 retries


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_no_retry_on_permanent_http_errors(monkeypatch, status):
    emb, calls, sleeps = run(monkeypatch, [Resp(status)] * 5)
    with pytest.raises(requests.exceptions.HTTPError):
        emb.embed_query("hi")
    assert len(calls) == 1 and not sleeps


def test_tls_errors_fail_closed_without_retry(monkeypatch):
    emb, calls, sleeps = run(monkeypatch, [requests.exceptions.SSLError("bad cert")] * 5)
    with pytest.raises(requests.exceptions.SSLError):
        emb.embed_query("hi")
    assert len(calls) == 1 and not sleeps


def test_batches_retry_independently(monkeypatch):
    # 40 texts = 3 batches; the 2nd batch hits one 503 and is retried; others are not repeated
    emb, calls, _ = run(monkeypatch, [Resp(200), Resp(503), Resp(200), Resp(200)])
    emb.embed_documents(["t"] * 40)
    assert len(calls) == 4
