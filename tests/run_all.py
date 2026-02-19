"""Run all test modules and print a final summary."""
import os, sys, unittest

# Ensure project root is on path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
sys.path.insert(0, root)

loader = unittest.TestLoader()
suite = unittest.TestSuite()
suite.addTests(loader.loadTestsFromName("tests.test_temporal_gnn_full"))
suite.addTests(loader.loadTestsFromName("tests.test_temporal_gnn_meticulous"))

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
