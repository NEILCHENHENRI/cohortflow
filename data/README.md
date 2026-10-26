# CohortFlow data setup

Runtime databases and derived indexes are local artifacts, excluded from Git. Real patient data is not included in this repository. Obtain MIMIC through its authorized distribution and prepare the required tables locally. A complete MIMIC download/import pipeline is not bundled. Do not commit patient records, clinical text, derived indexes, or data-containing logs.

## Runtime layout

| Mode | Files | Backend |
| --- | --- | --- |
| Local (default) | `data/mimic_hosp.duckdb` | Raw MIMIC-IV hospital tables, plus discharge notes |
| Demo | `data/demo/clinical.duckdb`, `data/demo/vocabulary.duckdb` | Generated synthetic OMOP-shaped data |

Local evidence domains map to these tables:

| Domain | Tables |
| --- | --- |
| `patient` | `patients` |
| `admission` | `admissions` |
| `condition` | `diagnoses_icd`, `d_icd_diagnoses` |
| `procedure` | `procedures_icd`, `d_icd_procedures` |
| `lab` | `labevents`, `d_labitems` |
| `medication` | `prescriptions` |
| `note` | `discharge` |

Local mode has no separate vocabulary database: resolution uses ICD dictionary text and recorded medication names. Backend capabilities depend on available tables and fields. Note chunks and retrieval indexes are cached under `data/note_index/`.

Run `python run.py` for local data or `python run.py --demo` for the [synthetic demo](demo/README.md). Commands are run from the repository root.

## Custom dataset onboarding

One command performs inspection, construction, and mapping validation:

```bash
PYTHONPATH=. python scripts/inspect_dataset.py /path/to/source \
  --output /path/to/adapter --build
# Optional field descriptions and conversion proposals: --llm
# Optional reviewed mapping: --mapping reviewed.json
```

The source may be CSV/CSV.GZ, Parquet, DuckDB, or SQLite. The output contains **`dataset.duckdb` alongside `adaptation_log.json`**, plus `field_catalog.json`, `construction_spec.json`, `schema_profile.json`, `cohort_mapping.json`, and reports. The Python entry point is `agents.cohort.adapters.adapt_dataset(...)`. Inspection without `--build` and the separate `scripts/construct_dataset.py` remain available for reviewed reruns.

Select **Custom** in the app and enter the adapter directory. Prepared retains the existing MIMIC/demo behavior. Custom uses catalog names, descriptions, types, and units in all four parsing modes and localized correction. The same cohort graph, Boolean logic, reuse, evidence, and manual recovery execute the results.

### Patient and event mappings

Conventional `patients.patient_id`/`subject_id`/`person_id` anchors can be nominated automatically. Names and overlapping samples only propose other relationships. Full-data checks report key types, uniqueness, nulls, overlap, and unmatched rows. They cannot prove that two identifiers have the same clinical meaning.

For different names, composite identities, indirect joins, or event relationships, supply a reviewed mapping:

```json
{
  "patient": {"table": "people", "columns": ["site", "person_key"]},
  "tables": {
    "observations": {
      "description": "Laboratory results",
      "patient_columns": ["site", "person_key"],
      "joins": [{"table": "visits", "left_columns": ["encounter"], "right_columns": ["visit_key"]}],
      "admission_column": "encounter",
      "event_time_column": "recorded_at__norm"
    }
  },
  "unit_aliases": {"local_unit_label": "mg"}
}
```

Joins form a chain starting at the queried table. `patient_columns` refer to its last table. Every join parent must be unique with compatible key types. Unmatched patient mappings block readiness. Admission and time columns refer to the queried table and must be reviewed explicitly. Encounter identifiers must share a namespace across mapped tables. Composite patient identities are supported. Encounter IDs are currently a single column. Stable integer cohort handles are ordered from the patient anchor's keys. Source identifiers, including leading zeros, remain available in patient details. Handles are local to a particular built dataset.

`ready` means the mapped domains passed validation. Unmapped tables remain stored and are listed in the log. `needs_review` means the DB exists but its mapping cannot be used yet. Custom supports typed comparisons, conjunctions on the same row, patient-level Boolean composition, and relations with mapped event/admission columns. It does not infer clinical concepts, dictionary expansions, aggregation semantics, or note adjudication from arbitrary columns.

### Values, units, and failure behavior

- Preserve raw columns and native numeric types. CSV stays lexical text. Derived numeric columns use exact integers or decimals. Text decimals normalize to `DECIMAL(38,12)`. Values that cannot fit exactly become NULL and are counted, never rounded silently. Native decimal precision is retained. Mixed SQLite storage classes retain their original type tags in DuckDB UNION columns and require review before prompt exposure.
- The LLM copies numbers as decimal strings and units literally, avoiding JSON float precision loss. Numeric JSON values remain accepted. Runtime conversion uses [Pint](https://pint.readthedocs.io/en/stable/) and [ucumvert](https://pypi.org/project/ucumvert/), including packaged UCUM definitions. No molar-mass tables are downloaded. Unit symbols are case-sensitive. Typography variants of micro are normalized. Local aliases live in reviewed mapping metadata.
- Unit checks are scoped by code/text filters in long-format measurement tables. Missing/unknown units, incompatible dimensions, and substance-dependent conversions leave the criterion unresolved with a logged clinician-review error. They never become an empty cohort or a silently omitted criterion. Prepared MIMIC unit matching is unchanged. Affine conversions use rational arithmetic before decimal output. Runtime exact-decimal comparisons are limited to DuckDB's 38-digit range.
- Invalid/nonfinite normalized values become NULL. Raw values and failure counts remain in the DB/log. LLM proposals define conversion rules, never replacement values for individual rows. Arrays and other nonscalar conversion results are rejected. Timezone offsets continue to be discarded by policy.
- All outputs are staged. Ordinary failures preserve the previous bundle and write `adaptation_failure.json`. Publication is not crash-atomic across multiple files. A runtime session rejects a changed bundle and must be reloaded.

Above 30 available fields, Diamond selects domains once per batch and Graphene once per predicate. Carbon and Graphite retain their existing call structure. The count comes from the active catalog. There is no second ranking/truncation layer: each selected domain is assumed to fit in the model context.

External data remains subject to its own license and access conditions.
