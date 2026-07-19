# Package QA record

- Frozen protocol SHA-256 (canonical JSON): `353ed499fa2f54c08262b3f254cb6c921a8609d4a16e7fe5c9765db7b64988b0`
- Frozen P3 matrix registry: `08899a7c14edb548acda759b2e9d61e232699ffc223d33a88d0b81e40f8d0203`
- Frozen label-swap registry: `614431f8cad56ba59ca8046d4dc351ff40e401e3723e9122ac12bc57804f87a9`
- Main dry run: 640 cells, 240 tasks, 32 workers.
- Optional P2 source block: 120 additional cells.
- `python -m py_compile code/*.py`: passed on 2026-07-19.
- `python -m unittest discover -s code/tests -v`: 10/10 passed, including a complete synthetic 80-context analysis and Agent-2 payoff-table orientation test.
- CLI smoke tests: task runner, validator, analyzer, and campaign `--help` passed.

The runtime is copied from the S2/P3 code used by the submitted experiments. The only scientific change inside `hettom_baseline.py` is confined to the previously unused payoff-in-prompt branch: the table is now rendered from each agent's own row/column perspective. No S1/S2/P3 result is rewritten.

