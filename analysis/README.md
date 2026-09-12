# Analysis

These modules interpret saved activations, directions, and experiment outputs. Each retains its existing configuration constants and output formats.

| Folder | Purpose |
| --- | --- |
| [behavior](behavior/) | Confidence relationships, answer transfer, and prompt positions. |
| [directions](directions/) | Direction comparisons, orthogonalization, and interpretation. |
| [interventions](interventions/) | Intervention diagnostics, plots, and synthesis. |

From the repository root, for example:

```bash
python -m analysis.directions.analyze_directions
python -m analysis.interventions.synthesize_causal_results
```

Read the selected module's docstring for upstream artifacts and the [execution guidelines](../CLAUDE.md) before running. The saved [causal ordering analysis](../docs/analysis/causal_ordering_analysis.md) is indexed with the project documents.
