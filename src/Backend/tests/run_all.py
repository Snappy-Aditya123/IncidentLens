"""Run all test modules and print a final summary."""
import os, sys, unittest

# Ensure project root is on path so src.Backend.* imports resolve
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(root)
if root not in sys.path:
    sys.path.insert(0, root)

# ── Unittest suites ──
loader = unittest.TestLoader()
suite = unittest.TestSuite()
suite.addTests(loader.loadTestsFromName("src.Backend.tests.test_temporal_gnn_full"))
suite.addTests(loader.loadTestsFromName("src.Backend.tests.test_temporal_gnn_meticulous"))

runner = unittest.TextTestRunner(verbosity=0)
result = runner.run(suite)

# ── Pytest suites (collected separately) ──
try:
    import pytest
    tests_dir = os.path.join(root, "src", "Backend", "tests")
    pytest_exit = pytest.main([tests_dir, "-x", "-q", "--tb=short", "--no-header"])
except ImportError:
    print("[WARN] pytest not installed — skipping pytest-based tests")
    pytest_exit = 0

print(f"\n{'='*50}")
print(f"  UNITTEST TESTS RUN : {result.testsRun}")
print(f"  FAILURES           : {len(result.failures)}")
print(f"  ERRORS             : {len(result.errors)}")

unittest_ok = result.wasSuccessful()
pytest_ok = pytest_exit == 0
status = "ALL PASSED" if (unittest_ok and pytest_ok) else "FAILED"
print(f"  STATUS             : {status}")
print(f"{'='*50}")

sys.exit(0 if (unittest_ok and pytest_ok) else 1)
