# Core library

Shared implementations used by the experiment and analysis modules:

| Modules | Role |
| --- | --- |
| `datasets.py`, `questions.py`, `tasks.py` | Dataset loading, question handling, and prompt templates. |
| `model_utils.py`, `extraction.py` | Model loading and activation extraction. |
| `metrics.py`, `probes.py`, `directions.py`, `answer_directions.py`, `confidence_directions.py` | Metrics, probes, and direction estimation. |
| `steering.py`, `steering_experiments.py` | Intervention hooks and shared intervention routines. |
| `paths.py`, `config_utils.py`, `logging_utils.py`, `plotting.py` | Repository paths, output routing, metadata, logs, and plots. |

`paths.py` anchors data and outputs to the repository root. Existing model settings, prompt templates, statistical routines, and output schemas remain in their original modules. Imports use `core.*`; runnable workflows live in [experiments](../experiments/) and [analysis](../analysis/).
