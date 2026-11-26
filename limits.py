"""Rate limiting and concurrency control (framework-free, thread-safe)."""
import threading
import time
from collections import deque


class SlidingWindowLimiter:
    """Allow at most `limit` events per `window` seconds for each key."""

    def __init__(self, limit, window=60.0, clock=time.monotonic, max_keys=10_000):
        self.limit, self.window, self.clock, self.max_keys = int(limit), float(window), clock, max_keys
        self._events, self._lock = {}, threading.Lock()

    def check(self, key):
        """Record an attempt. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            if len(self._events) > self.max_keys:          # drop idle keys
                for k in [k for k, q in self._events.items() if not q or now - q[-1] > self.window]:
                    del self._events[k]
            q = self._events.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1.0, self.window - (now - q[0]))
            q.append(now)
            return True, 0.0


class TurnGate:
    """Caps how many model turns run at the same time."""

    def __init__(self, max_concurrent):
        self._sem = threading.BoundedSemaphore(int(max_concurrent))

    def acquire(self, timeout=30.0):
        return self._sem.acquire(timeout=timeout)

    def release(self):
        self._sem.release()
