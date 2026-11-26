"""Rate limiter and concurrency gate (UI side, no network)."""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limits import SlidingWindowLimiter, TurnGate


class Clock:
    t = 0.0
    def __call__(self): return self.t


def test_limiter_blocks_after_limit_and_recovers():
    c = Clock(); lim = SlidingWindowLimiter(3, 60, clock=c)
    assert [lim.check("a")[0] for _ in range(3)] == [True] * 3
    ok, wait = lim.check("a")
    assert not ok and 0 < wait <= 60
    c.t = 61
    assert lim.check("a")[0]


def test_limiter_keys_are_independent():
    lim = SlidingWindowLimiter(1, 60, clock=Clock())
    assert lim.check("a")[0] and not lim.check("a")[0] and lim.check("b")[0]


def test_turn_gate_caps_concurrency():
    g = TurnGate(2)
    assert g.acquire(0.01) and g.acquire(0.01) and not g.acquire(0.01)
    g.release(); assert g.acquire(0.01)
