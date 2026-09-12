# Experiments

Run these modules from the repository root with `python -m`. Each module keeps its existing configuration constants and output formats.

| Folder | Purpose |
| --- | --- |
| [data](data/) | Build next-token data and filter multiple-choice data. |
| [directions](directions/) | Extract activations and identify direct-task directions. |
| [transfer](transfer/) | Evaluate meta-task and cross-dataset transfer. |
| [interventions](interventions/) | Run ablation, steering, cross-direction, and activation-patching experiments. |

The main entry points are:

```bash
python -m experiments.directions.identify_mc_correlate
python -m experiments.transfer.test_meta_transfer
python -m experiments.interventions.run_ablation_causality
python -m experiments.interventions.run_steering_causality
```

Read each module's input requirements and the [execution guidelines](../CLAUDE.md) before running it on the configured remote machine. Outputs continue through `core.config_utils`.
