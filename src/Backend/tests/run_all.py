"""Run all test modules and print a final summary."""
import os, sys, unittest

# Ensure project root is on path so src.Backend.* imports resolve
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(root)
if root not in sys.path:
    sys.path.insert(0, root)

loader = unittest.TestLoader()
suite = unittest.TestSuite()
suite.addTests(loader.loadTestsFromName("src.Backend.tests.test_temporal_gnn_full"))
suite.addTests(loader.loadTestsFromName("src.Backend.tests.test_temporal_gnn_meticulous"))

runner = unittest.TextTestRunner(verbosity=0)
result = runner.run(suite)

print(f"\n{'='*50}")
print(f"  TOTAL TESTS RUN : {result.testsRun}")
print(f"  FAILURES        : {len(result.failures)}")
print(f"  ERRORS          : {len(result.errors)}")
status = "ALL PASSED" if result.wasSuccessful() else "FAILED"
print(f"  STATUS          : {status}")
print(f"{'='*50}")

sys.exit(0 if result.wasSuccessful() else 1)
