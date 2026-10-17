# Test suites

The default test command is intentionally offline:

```powershell
pytest
```

Pytest is configured to collect only `tests/unit`. These tests cover pure
policies, graph routing, ingestion boundaries, diagnostics, and request-local
timing without connecting to PostgreSQL, Pinecone, an LLM provider, or a model
registry. Live provider probes and obsolete manual scripts are intentionally
not stored as tests.

Test-suite boundaries:

- `tests/unit`: deterministic and offline; runs by default.
- `tests/component`: one application component with external boundaries
  replaced by fakes or mocks (planned).
- `tests/integration`: explicitly provisioned infrastructure (planned).
- `tests/e2e`: live providers and full application flows (planned).

Do not place credential-dependent or network-dependent tests in `tests/unit`.
The live V3 retrieval executor in `scripts/evaluate_v3_retrieval.py` is an
explicit evaluation command, not a unit test. Unit coverage verifies its case
selection, rank/score report assembly, evidence-label semantics, and summary
denominators with in-memory records.
