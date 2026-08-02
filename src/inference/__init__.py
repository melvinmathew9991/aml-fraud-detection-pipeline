"""
The only code path that turns a raw transaction into a score
(ARCHITECTURE.md §4). Both the API and the batch scorer import this
package; nothing outside it reimplements feature construction.

Run via `uvicorn inference.main:app --app-dir src`, or import in tests via
tests/conftest.py's sys.path insertion of src/ -- both put src/ on
sys.path, which is what lets this package's modules flat-import sibling
training-side modules (`features`, `config`) the same way every other
script in src/ does.
"""
