# CohortFlow

CohortFlow is a research implementation for revising clinical cohort requests during execution. It translates natural-language eligibility criteria into explicit task state, accepts natural-language and manual corrections, and reuses unaffected work.

## Quick start

Requires Python 3.12+ and a running Ollama server. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3:4b
python run.py --demo
```

On Windows, activate with `.venv\Scripts\activate`. Start Ollama separately with `ollama serve` if needed. Model weights are downloaded separately.

Demo mode creates 24 invented patients and synthetic OMOP-shaped records locally. Enter “Find patients with type 2 diabetes mellitus,” followed by “Also require metformin exposure.” The matching cohorts contain 12 and 8 patients, respectively. If a concept selection is requested, choose the matching demo concept. Demo mode still uses Ollama for natural-language parsing and correction.

For local MIMIC mode, prepare `data/mimic_hosp.duckdb`, pull `medgemma1.5:4b` for terminology fallback, and run `python run.py`. Real patient data is not distributed. See [data setup](data/README.md) for the required tables and custom dataset onboarding.

## How it works

- **Explicit task state:** `TaskSpec` stores requirements, Boolean logic, and context.
- **Localized correction:** compact TaskDelta (v2) interprets edits against stable references. Code compiles changed predicates and applies validated structural edits.
- **Selective reuse:** completed artifacts are reused only when their dependencies and semantic fingerprints remain valid.
- **Human continuation:** clarification, concept choices, note execution, and correction fallback use explicit pending interactions.
- **Inspection and manual editing:** Streamlit exposes task structure, evidence, execution progress, and a structured fallback editor.

## Defaults and controls

| Setting | Default |
| --- | --- |
| Initial parsing | Graphite: Boolean structure, then batch predicate compilation |
| Correction | Robust Delta with compact TaskDelta (v2), in the UI and cohort parser |
| Alternative correction | Full Regeneration, selected explicitly or offered after an unsafe edit |
| Execution | Native backend. LangGraph is also available |
| Steering | Sequential. Parallel may overlap interpretation with active work |
| Thinking | Off for parsing and correction. On for note adjudication |
| Source repair | Off. Optional bounded recovery and isolated patch proposals |

Both execution backends run graph operators sequentially. Parallel steering is separate from concurrent graph execution. Other parsing and localized strategies remain available as advanced options. See [agent behavior](agents/README.md) and [interface controls](interfaces/README.md).

## Repository guide

| Directory | Contents |
| --- | --- |
| [agents/](agents/README.md) | Cohort parsing, correction, planning, operators, dataset adapters |
| [framework/](framework/README.md) | Task state, execution graphs, steering, reuse, repair |
| [infrastructure/](infrastructure/README.md) | Ollama, DuckDB, MIMIC/OMOP backends, retrieval |
| [interfaces/](interfaces/README.md) | Streamlit and interaction orchestration |
| [data/](data/README.md) | Data preparation documentation and synthetic demo |
| `prompts/`, `presentation/` | Prompt text and shared view models |
| `tests/` | Software tests using mocks and temporary databases |

## Validation

Use the same activated Python environment for dependency installation, the app, and tests:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests -q
```

The test suite does not require real MIMIC data or a running model server. Custom-dataset unit tests require the declared Pint and ucumvert dependencies. If an import is missing, install the requirements using the same Python interpreter that runs pytest.

## Paper and citation

- **Authors:** Neil Chen, Adrian Serapio, Hanxue Gu, Evan Lee, Kang Wang, Yang Yang
- **Paper title:** CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction
- **Paper URL:** TODO
- **Repository URL:** https://github.com/NEILCHENHENRI/cohortflow

If you use CohortFlow in research, please cite the associated paper once its citation is added here. Citation is requested separately from the code license.

## License

The code is distributed under the [MIT License](LICENSE). Third-party packages, model weights, and external datasets retain their own licenses and access conditions. The MIT license does not cover them.
