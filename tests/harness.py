"""A test runner in forty lines, so the suites run with nothing installed.

The point of the pure suites is that they need neither torch nor a ComfyUI, and
pulling in pytest to check that would be a third thing to install before the
first assertion. `python3 tests/test_refine.py` is the whole contract; pytest
collects the same files if you would rather have it, because a `test_*` function
that raises is a failure either way.
"""

import sys

FAILURES = []


def check(claim, ok, detail=""):
    """One assertion. Records rather than raises, so a run reports every failure."""
    if ok:
        return True
    FAILURES.append(claim + (f" — {detail}" if detail else ""))
    return False


def passed(name):
    """Print what a suite found, and exit non-zero if anything failed."""
    if FAILURES:
        print(f"{name}: {len(FAILURES)} failed")
        for failure in FAILURES:
            print(f"  ✗ {failure}")
        sys.exit(1)
    print(f"{name}: ok")


def expect(claim, fn, fragment):
    """`fn` raises, and the message says `fragment`."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — which exception is the claim
        return check(claim, fragment in str(exc), f"raised {exc!r}")
    return check(claim, False, "did not raise")
