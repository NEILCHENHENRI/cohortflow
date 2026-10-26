from __future__ import annotations

from pathlib import Path
import json

from agents.cohort.adapters.models import FieldSpec


class FieldRegistry:
    def __init__(self, fields: list[FieldSpec]) -> None:
        names = [field.semantic_field for field in fields]
        if any(not name.strip() for name in names) or len(set(names)) != len(names):
            raise ValueError("semantic field names must be non-empty and unique")
        self._fields = {field.semantic_field: field for field in fields}

    @classmethod
    def from_adapter_dir(cls, adapter_dir: str | Path) -> "FieldRegistry":
        root = Path(adapter_dir)
        catalog = json.loads((root / "field_catalog.json").read_text())
        construction = json.loads((root / "construction_spec.json").read_text())
        normalized = {}
        catalog_keys = {(item["table"], item["column"]) for item in catalog}
        if len(catalog_keys) != len(catalog):
            raise ValueError("duplicate catalog table/column entries")
        for item in construction.get("derived_fields", []):
            key = (item["table"], item["source_column"])
            if key in normalized:
                raise ValueError(f"ambiguous derived field for {key}")
            if key not in catalog_keys:
                raise ValueError(f"derived field is absent from catalog: {key}")
            normalized[key] = item
        fields: list[FieldSpec] = []
        for item in catalog:
            derived = normalized.get((item["table"], item["column"]))
            if item.get("is_identifier_or_code"):
                kind = "identifier"
                if derived is not None:
                    raise ValueError(f"identifier/code cannot use a normalized field: {item['semantic_field']}")
            elif item.get("is_numeric") and item.get("is_datetime"):
                raise ValueError(f"field cannot be both numeric and datetime: {item['semantic_field']}")
            elif item.get("is_numeric"):
                kind = "numeric"
            elif item.get("is_datetime"):
                kind = "datetime"
            else:
                kind = "semantic"
            if derived is not None and derived.get("kind", kind) != kind:
                raise ValueError(f"catalog/construction type mismatch: {item['semantic_field']}")
            fields.append(FieldSpec(
                semantic_field=item["semantic_field"],
                description=item["description"],
                table=item["table"],
                source_column=item["column"],
                value_column=(derived["target_column"] if derived else item["column"]),
                value_kind=kind,
                unit_column=((derived.get("unit_target_column") or item.get("unit_column")) if derived else item.get("unit_column")),
                source_unit_column=item.get("unit_column"),
                identifier_or_code=bool(item.get("is_identifier_or_code")),
                included_in_prompt=bool(item.get("included_in_prompt", True)),
                canonical_unit=derived.get("canonical_unit") if derived else None,
                observed_units=tuple(derived.get("observed_units", ())) if derived else (),
            ))
        return cls(fields)

    def get(self, semantic_field: str) -> FieldSpec:
        try:
            return self._fields[semantic_field]
        except KeyError as exc:
            raise KeyError(f"unknown semantic field: {semantic_field}") from exc

    def __contains__(self, semantic_field: str) -> bool:
        return semantic_field in self._fields

    def all(self) -> tuple[FieldSpec, ...]:
        return tuple(self._fields.values())


# Custom datasets use the same cohort graph and correction machinery. Only the
# field contract and deterministic repository are dataset-specific.
class CustomDataset:
    PREFIX = "dataset:"
    OPERATORS = ("=", "!=", ">", ">=", "<", "<=")

    def __init__(self, adapter_dir: str | Path):
        from infrastructure.database.duckdb import DuckDBConfig, DuckDBConnectionFactory
        self.root = Path(adapter_dir).expanduser().resolve()
        self.registry = FieldRegistry.from_adapter_dir(self.root)
        self.mapping = json.loads((self.root / "cohort_mapping.json").read_text())
        self.database = DuckDBConnectionFactory(DuckDBConfig(self.root / "dataset.duckdb", read_only=True))
        self.unit_aliases = self.mapping.get("unit_aliases", {})
        self._units = None
        self.validation = self.validate_mapping()
        # The runtime caches/graph artifacts refer to this immutable bundle.
        self._bundle_state = self._current_bundle_state()

    def _current_bundle_state(self):
        return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in (
            self.root / "dataset.duckdb", self.root / "field_catalog.json",
            self.root / "construction_spec.json", self.root / "cohort_mapping.json",
        ))

    @staticmethod
    def quote(value):
        return '"' + value.replace('"', '""') + '"'

    def _query(self, sql, params=()):
        if hasattr(self, "_bundle_state") and self._bundle_state != self._current_bundle_state():
            raise ValueError("adapter bundle changed. Reload the Custom session before executing")
        from decimal import Decimal
        for value in params:
            if isinstance(value, Decimal):
                digits = len(value.as_tuple().digits)
                exponent = value.as_tuple().exponent
                if max(digits + max(exponent, 0), -exponent) > 38:
                    raise ValueError("numeric comparison exceeds DuckDB exact decimal precision. Clinician review required")
        with self.database.connect() as con:
            return con.execute(sql, params).fetchall()

    def domains(self):
        return tuple(self.PREFIX + table for table in self.mapping["tables"])

    def render_domains(self):
        return "\n".join(
            f"- {domain}: {self.mapping['tables'][domain.removeprefix(self.PREFIX)].get('description', 'dataset table')}, {len(self.fields((domain,)))} fields"
            for domain in self.domains()
        )

    def fields(self, domains=None):
        allowed = set(domains or self.domains())
        return tuple(f for f in self.registry.all() if f.included_in_prompt and self.PREFIX + f.table in allowed)

    def render_fields(self, domains=None):
        return "AVAILABLE DATASET FIELDS (exact names. Values and units are literal):\n" + "\n".join(
            f"- {self.PREFIX + f.table}: {f.semantic_field} ({f.value_kind}). {f.description}, "
            f"unit={f.canonical_unit or ('per row in ' + f.unit_column if f.unit_column else 'unspecified')}, "
            f"observed units={json.dumps(f.observed_units, ensure_ascii=False)}"
            for f in self.fields(domains)
        )

    def requirement_schema(self, domains=None):
        variants = []
        for domain in domains or self.domains():
            filters = []
            for field in self.fields((domain,)):
                value_type = ["string", "number"] if field.value_kind == "numeric" else "string"
                filters.append({"type": "object", "additionalProperties": False,
                    "required": ["field", "operator", "value"], "properties": {
                        "field": {"const": field.semantic_field},
                        "operator": {"enum": list(self.OPERATORS if field.value_kind in {"numeric", "datetime"} else ("=", "!="))},
                        "value": {"type": value_type}, "unit": {"type": "string"},
                    }})
            if not filters:
                continue
            variants.append({"type": "object", "additionalProperties": False,
                "required": ["requirement_id", "kind", "payload", "polarity", "required"], "properties": {
                    "requirement_id": {"type": "string"}, "kind": {"const": domain},
                    "polarity": {"enum": ["include", "exclude"]}, "required": {"type": "boolean"},
                    "payload": {"type": "object", "additionalProperties": False, "required": ["filters"],
                        "properties": {"filters": {"type": "array", "minItems": 1, "items": {"oneOf": filters}}}},
                }})
        return {"oneOf": variants}

    def requirement_from_payload(self, item):
        from framework.task import Requirement
        self.validate_requirement(item["kind"], item["payload"])
        if item.get("polarity") not in {"include", "exclude"} or type(item.get("required")) is not bool:
            raise ValueError("invalid requirement polarity/required")
        return Requirement(item["requirement_id"], item["kind"], dict(item["payload"]), item["polarity"], item["required"])

    def validate_requirement(self, kind, payload):
        from decimal import Decimal
        if kind not in self.domains():
            raise ValueError(f"unknown custom domain: {kind}")
        if not isinstance(payload, dict) or set(payload) != {"filters"}:
            raise ValueError("custom predicate requires only filters")
        filters = payload["filters"]
        if not isinstance(filters, list) or not filters:
            raise ValueError("custom predicate requires non-empty filters")
        available = {f.semantic_field: f for f in self.fields((kind,))}
        for item in filters:
            if not isinstance(item, dict) or not {"field", "operator", "value"} <= set(item) or set(item) - {"field", "operator", "value", "unit"}:
                raise ValueError("invalid custom filter fields")
            field = available.get(item["field"])
            if field is None:
                raise ValueError(f"field not available in {kind}: {item['field']}")
            if item["operator"] not in self.OPERATORS:
                raise ValueError("unsupported comparison operator")
            if field.value_kind == "numeric":
                if type(item["value"]) not in {str, int, float, Decimal}:
                    raise ValueError("finite numeric value required")
                try:
                    finite = Decimal(str(item["value"])).is_finite()
                except Exception as exc:
                    raise ValueError("finite numeric value required") from exc
                if not finite:
                    raise ValueError("finite numeric value required")
            else:
                if not isinstance(item["value"], str):
                    raise ValueError("text/datetime/identifier values must be strings")
                if field.value_kind != "datetime" and item["operator"] not in {"=", "!="}:
                    raise ValueError("text/identifier fields support equality only")
            if "unit" in item and (field.value_kind != "numeric" or not isinstance(item["unit"], str) or not item["unit"].strip()):
                raise ValueError("unit must be non-empty and attached to a number")

    def _columns(self, table):
        return {r[0]: r[1] for r in self._query(f"DESCRIBE {self.quote(table)}")}

    def _keys(self, columns, available):
        if not isinstance(columns, list) or not columns or len(set(columns)) != len(columns) or any(c not in available for c in columns):
            raise ValueError(f"invalid key columns: {columns}")
        return columns

    def validate_mapping(self):
        """Full-data checks, with no coercion of lexical IDs or speculative joins."""
        patient = self.mapping["patient"]
        patient_columns = self._columns(patient["table"])
        keys = self._keys(patient["columns"], patient_columns)
        q = self.quote
        key_sql = ', '.join(q(k) for k in keys)
        nulls = ' OR '.join(f'{q(k)} IS NULL' for k in keys)
        n, unique, missing = self._query(f"SELECT count(*), count(DISTINCT ({key_sql})), count(*) FILTER (WHERE {nulls}) FROM {q(patient['table'])}")[0]
        if "__id" in keys:
            raise ValueError("patient key __id is reserved for cohort handles")
        if not n or unique != n or missing:
            raise ValueError("patient anchor keys must be non-null, unique, and non-empty")
        report = []
        if not self.mapping.get("tables"):
            raise ValueError("no patient-mapped tables")
        for table, binding in self.mapping["tables"].items():
            cols = self._columns(table)
            # Reviewed joins are a chain. Each step must have a unique parent key.
            current = cols
            for join in binding.get("joins", []):
                parent = self._columns(join["table"])
                left = self._keys(join["left_columns"], current)
                right = self._keys(join["right_columns"], parent)
                if len(left) != len(right) or any(current[a] != parent[b] for a, b in zip(left, right)):
                    raise ValueError("join keys must have identical types and arity")
                parent_keys = ', '.join(q(k) for k in right)
                parent_nulls = ' OR '.join(f'{q(k)} IS NULL' for k in right)
                total, distinct, null_count = self._query(f"SELECT count(*), count(DISTINCT ({parent_keys})), count(*) FILTER (WHERE {parent_nulls}) FROM {q(join['table'])}")[0]
                if total != distinct or null_count:
                    raise ValueError("join parent must be unique. Ambiguous many-to-many mapping")
                current = parent
            mapped = self._keys(binding["patient_columns"], current)
            if len(mapped) != len(keys) or any(current[a] != patient_columns[b] for a, b in zip(mapped, keys)):
                raise ValueError("patient keys must have identical types and arity. Explicit adaptation required")
            for optional in ("admission_column", "event_time_column"):
                if binding.get(optional) and binding[optional] not in cols:
                    raise ValueError(f"unknown {optional} in {table}")
            if binding.get("event_time_column") and not cols[binding['event_time_column']].startswith(("TIMESTAMP", "DATE")):
                raise ValueError("event_time_column must be a normalized date/timestamp")
            for field in self.registry.all():
                if field.table == table:
                    for col in (field.source_column, field.value_column, field.unit_column):
                        if col and col not in cols:
                            raise ValueError(f"catalog references missing column: {table}.{col}")
                    from agents.cohort.adapters.semantics import numeric_storage_type
                    if field.value_kind == "numeric" and not numeric_storage_type(cols[field.value_column]):
                        raise ValueError(f"catalog numeric type does not match database: {field.semantic_field}")
                    if field.value_kind == "datetime" and cols[field.value_column] not in {"DATE", "TIMESTAMP", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S"}:
                        raise ValueError(f"catalog datetime type does not match database: {field.semantic_field}")
            sql = self._from(table, left=True)
            rows, matched = self._query(f"{self._patient_cte()} SELECT count(*), count(p.__id) {sql}")[0]
            report.append({"table": table, "rows": rows, "matched_rows": matched, "unmatched_rows": rows - matched})
            if rows != matched:
                raise ValueError(f"{table}: {rows - matched} rows lack a validated patient mapping")
        return report

    def _patient_cte(self):
        q = self.quote
        patient = self.mapping["patient"]
        keys = ', '.join(q(k) for k in patient["columns"])
        return f"WITH __patients AS (SELECT {keys}, row_number() OVER (ORDER BY {keys}) AS __id FROM {q(patient['table'])})"

    def _from(self, table, left=False):
        q = self.quote
        binding = self.mapping["tables"][table]
        clause = f"FROM {q(table)} t0"
        alias = "t0"
        how = "LEFT JOIN" if left else "JOIN"
        for index, join in enumerate(binding.get("joins", []), 1):
            nxt = f"t{index}"
            on = ' AND '.join(f'{alias}.{q(a)} = {nxt}.{q(b)}' for a, b in zip(join['left_columns'], join['right_columns']))
            clause += f" {how} {q(join['table'])} {nxt} ON {on}"
            alias = nxt
        on = ' AND '.join(f'{alias}.{q(a)} = p.{q(b)}' for a, b in zip(binding['patient_columns'], self.mapping['patient']['columns']))
        return clause + f" {how} __patients p ON {on}"

    def _where(self, kind, payload):
        from agents.cohort.adapters.semantics import UnitConverter, UnitConversionError
        from decimal import Decimal
        self.validate_requirement(kind, payload)
        clauses, params = [], []
        # Unit alternatives must be scoped to this analyte/code, not every
        # measurement stored in a shared long-format observations table.
        scope_clauses, scope_params = [], []
        for selector in payload["filters"]:
            selected = self.registry.get(selector["field"])
            if selected.value_kind in {"semantic", "identifier"}:
                scope_clauses.append(f"CAST({self.quote(selected.value_column)} AS VARCHAR) {selector['operator']} ?")
                scope_params.append(selector["value"])
        unit_scope = " AND " + " AND ".join(scope_clauses) if scope_clauses else ""
        for item in payload["filters"]:
            f = self.registry.get(item["field"])
            column = 't0.' + self.quote(f.value_column)
            op, value = item["operator"], item["value"]
            if f.value_kind == "datetime":
                # Same deliberate timezone-discard convention as adaptation.
                if self._query("SELECT TRY_CAST(? AS TIMESTAMP)", [value])[0][0] is None:
                    raise ValueError("invalid datetime literal")
                clauses.append(f"{column} {op} CAST(? AS TIMESTAMP)"); params.append(value)
            elif f.value_kind == "numeric":
                value = Decimal(str(value))
                if "unit" not in item:
                    if f.unit_column or f.canonical_unit:
                        raise UnitConversionError(f"{f.semantic_field}: request unit missing. Clarification required")
                    clauses.append(f"{column} {op} ?"); params.append(value)
                else:
                    if self._units is None:
                        self._units = UnitConverter(self.unit_aliases)
                    if f.unit_column:
                        if f.canonical_unit and f.source_unit_column:
                            unsupported = self._query(
                                f"SELECT count(*) FROM {self.quote(f.table)} WHERE {self.quote(f.source_column)} IS NOT NULL "
                                f"AND {self.quote(f.unit_column)} IS NULL{unit_scope}", scope_params,
                            )[0][0]
                            if unsupported:
                                raise UnitConversionError(f"{f.semantic_field}: {unsupported} rows have unresolved adapted units. Clinician review required")
                        units = [r[0] for r in self._query(f"SELECT DISTINCT {self.quote(f.unit_column)} FROM {self.quote(f.table)} WHERE {self.quote(f.value_column)} IS NOT NULL{unit_scope}", scope_params)]
                        alternatives = []
                        for unit in units:
                            if not unit:
                                raise UnitConversionError(f"{f.semantic_field}: stored unit missing. Clinician review required")
                            converted = self._units.convert(value, item["unit"], str(unit))
                            alternatives.append(f"(t0.{self.quote(f.unit_column)} = ? AND {column} {op} ?)")
                            params.extend([unit, converted])
                        clauses.append('(' + ' OR '.join(alternatives) + ')' if alternatives else 'FALSE')
                    elif f.canonical_unit:
                        clauses.append(f"{column} {op} ?"); params.append(self._units.convert(value, item["unit"], f.canonical_unit))
                    else:
                        raise UnitConversionError(f"{f.semantic_field}: dataset unit unspecified. Clinician review required")
            else:
                clauses.append(f"CAST({column} AS VARCHAR) {op} ?"); params.append(value)
        return ' AND '.join(clauses), params

    def event_sql(self, kind, payload, concept=None):
        table = kind.removeprefix(self.PREFIX)
        where, params = self._where(kind, payload)
        binding = self.mapping["tables"][table]
        admission = 't0.' + self.quote(binding['admission_column']) if binding.get('admission_column') else 'NULL'
        event = 't0.' + self.quote(binding['event_time_column']) if binding.get('event_time_column') else 'NULL::TIMESTAMP'
        return f"{self._patient_cte()} SELECT p.__id subject_id, {admission} hadm_id, {event} event_time {self._from(table)} WHERE {where}", params

    def evaluate_requirement(self, kind, payload):
        import logging
        try:
            sql, params = self.event_sql(kind, payload)
            return {row[0] for row in self._query(f"SELECT DISTINCT subject_id FROM ({sql})", params)}
        except ValueError as exc:
            logging.getLogger(__name__).warning("Custom criterion unresolved (%s): %s", kind, exc)
            raise

    def all_patients(self):
        return {r[0] for r in self._query(self._patient_cte() + " SELECT __id FROM __patients")}

    def admissions_matching_requirement(self, kind, payload, concept=None):
        if not self.mapping["tables"][kind.removeprefix(self.PREFIX)].get("admission_column"):
            raise ValueError("same-admission relation requires a reviewed admission mapping")
        sql, params = self.event_sql(kind, payload)
        return set(self._query(f"SELECT DISTINCT subject_id, hadm_id FROM ({sql}) WHERE hadm_id IS NOT NULL", params))

    def patients_satisfying_relation(self, *, relation, left_kind, left_payload, right_kind, right_payload,
                                    left_concept=None, right_concept=None, max_gap_hours=None, same_admission=False):
        for kind in (left_kind, right_kind):
            binding = self.mapping["tables"][kind.removeprefix(self.PREFIX)]
            if not binding.get("event_time_column") or (same_admission and not binding.get("admission_column")):
                raise ValueError("relation requires reviewed event/admission mappings")
        if relation not in {"before", "after"}:
            raise ValueError("invalid temporal relation")
        lsql, lp = self.event_sql(left_kind, left_payload)
        rsql, rp = self.event_sql(right_kind, right_payload)
        op = '<' if relation == 'before' else '>'
        clauses = [f"l.event_time {op} r.event_time"]
        params = lp + rp
        if same_admission:
            clauses.append('l.hadm_id = r.hadm_id')
        if max_gap_hours is not None:
            if max_gap_hours < 0:
                raise ValueError("negative time gap")
            clauses.append("abs(date_diff('hour', l.event_time, r.event_time)) <= ?"); params.append(max_gap_hours)
        return {r[0] for r in self._query(f"WITH l AS ({lsql}), r AS ({rsql}) SELECT DISTINCT l.subject_id FROM l JOIN r ON l.subject_id = r.subject_id WHERE {' AND '.join(clauses)}", params)}

    def patient_detail(self, patient_id):
        rows = self._query(self._patient_cte() + ' SELECT * FROM __patients WHERE __id = ?', [patient_id])
        keys = self.mapping['patient']['columns']
        return {"demographics": dict(zip(keys, rows[0][:-1])) if rows else {}}
