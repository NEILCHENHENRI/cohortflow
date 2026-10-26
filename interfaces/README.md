# Interfaces

Streamlit provides the cohort workspace. The interaction controller coordinates requests, staged corrections, pending replies, and resumable execution. Rendering uses shared models from `presentation/`.

## Components

| Module | Responsibility |
| --- | --- |
| `streamlit_app.py` | Application setup, settings, workspace rendering |
| `controller.py` | Requests, corrections, clarification, continuation |
| `background.py` | Background work, queued corrections, progress events |
| `session.py` | Messages, selections, and graph-view state |
| `components/` | Graph, entity, evidence, and manual-editor panels |
| `renderers/sections.py` | Shared text, table, JSON, and warning rendering |

## Running

From the repository root:

```bash
python run.py --demo
# With a separately prepared MIMIC database:
python run.py
```

See [installation](../README.md#quick-start) and [data setup](../data/README.md).

## Controls and behavior

- **Correction:** Robust Delta defaults to Compact TaskDelta (v2). Advanced localized strategies and Full Regeneration remain selectable. Unsafe edits preserve task state and request confirmation before regeneration.
- **Steering:** Sequential applies queued corrections at cooperative checkpoints. Parallel may interpret a correction during active work. Graph operators remain sequential.
- **Notes:** confirmation precedes note retrieval and adjudication. Declining notes removes the corresponding criteria explicitly.
- **Manual editing:** `/fallback` opens a structured editor. Validated edits are staged before execution.
- **Inspection:** graph topology, evidence, node details, and model activity explain the execution path.
- **Dataset:** Prepared uses the configured MIMIC or synthetic backend. Custom loads a constructed adapter directory.

Settings and dataset switching are locked during active work. A valid dataset switch starts a fresh session. An invalid path preserves the current session. Thinking settings are isolated per model-call context. New settings affect future calls and do not recompute completed results.

Source repair is off by default. Review and adoption of a saved patch require a restart and a fresh session. See [repair behavior](../infrastructure/README.md#experimental-source-repair).

UI tests use mocked services.
