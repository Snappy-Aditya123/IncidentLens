# ──────────────────────────────────────────────
# IncidentLens CLI — delegates to cli module
# Run: python main.py health | ingest | investigate | serve | convert
# ──────────────────────────────────────────────

import os, sys

# Ensure project root is on sys.path so "src.Backend.*" imports resolve
# regardless of how this script is invoked (python src/Backend/main.py, etc.)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.Backend.cli import main

if __name__ == "__main__":
    main()
