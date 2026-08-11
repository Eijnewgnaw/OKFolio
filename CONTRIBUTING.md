# Contributing

1. Create an isolated branch.
2. Keep business logic under `okfolio/`; scripts should remain thin CLI
   adapters.
3. Add or update tests for every behavior change.
4. Run `PYTHONPATH=. python3 -m pytest -q`.
5. Confirm that no runtime data, private endpoint, credential, or model weight
   is included.
6. Keep ConceptRef, Concept, relation, provenance, and publication contracts
   backward-compatible unless the change is explicitly versioned.

Contributions are accepted under the Apache License 2.0.
