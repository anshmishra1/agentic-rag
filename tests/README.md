# Test suites

The default test command is intentionally offline:

```powershell
pytest
```

Pytest is configured to collect only `tests/unit`. These tests cover pure
policies, graph routing, and request-local timing without connecting to
PostgreSQL, Pinecone, an LLM provider, or a model registry.

The Python files directly under `tests/` are legacy/manual checks. Several of
them require live services or load local ML models, and some still describe
older application contracts. They are excluded from default collection until
they are classified and migrated into explicit component, integration, or
end-to-end suites.

Test-suite boundaries:

- `tests/unit`: deterministic and offline; runs by default.
- `tests/component`: one application component with external boundaries
  replaced by fakes or mocks (planned).
- `tests/integration`: explicitly provisioned infrastructure (planned).
- `tests/e2e`: live providers and full application flows (planned).

Do not place credential-dependent or network-dependent tests in `tests/unit`.
