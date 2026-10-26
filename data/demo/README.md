# CohortFlow synthetic demo

Demo mode uses two locally generated databases:

- `clinical.duckdb`: 24 invented patients and synthetic clinical records.
- `vocabulary.duckdb`: demo-only concept IDs in an OMOP-shaped vocabulary.

No real patient data is used. Generated DuckDB files are excluded from Git.

From the repository root, start the app. Missing demo databases are created automatically:

```bash
python run.py --demo
```

To replace existing demo databases with a fresh generated copy:

```bash
PYTHONPATH=. python scripts/build_demo_databases.py --overwrite
```

Try “Find patients with type 2 diabetes mellitus,” then “Also require metformin exposure.” The matching cohorts contain 12 and 8 patients, respectively. If prompted, select the matching demo concept. A running Ollama server and the qwen3:4b model are required for parsing and correction. Demo mode uses the OMOP backend and supports conditions, medications, age, gender, and visits. It does not include clinical notes, laboratory results, or procedures. For MIMIC-specific note retrieval, see the [infrastructure guide](../../infrastructure/README.md#models-and-note-evidence).

## Database version mismatch

If startup reports that a demo database was created by a newer DuckDB version, stop the app and rebuild only the synthetic demo files with the same Python environment used to launch the app:

```bash
python scripts/build_demo_databases.py --overwrite
python run.py --demo
```

The generator writes DuckDB 1.0-compatible storage, including when run with DuckDB 2.x. The overwrite command replaces only the generated demo databases. It is not a migration command for MIMIC or custom data.
