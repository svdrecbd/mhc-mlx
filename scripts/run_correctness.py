"""Quick correctness runner.

Use this to warm Metal JIT caches without remembering pytest flags.
"""


def main() -> int:
    try:
        import pytest
    except ImportError:
        print("pytest is required. Install with: pip install -e '.[dev]'")
        return 1
    return pytest.main(["-q", "tests/test_correctness.py"])


if __name__ == "__main__":
    raise SystemExit(main())
