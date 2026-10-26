from __future__ import annotations

import re
import csv
import gzip
from pathlib import Path

_ID_HINT = re.compile(r"(^|_)(id|identifier|code|num|number)$", re.I)
_CODE_HINT = re.compile(r"(icd|ndc|gsn|itemid|hadm_id|subject_id|stay_id|specimen_id|event_id|seq_num)", re.I)


def identifier_or_code_hint(column_name: str, samples: list[str]) -> bool:
    name = column_name.strip()
    if _CODE_HINT.search(name) or _ID_HINT.search(name):
        return True
    non_empty = [value.strip() for value in samples if value and value.strip()]
    if not non_empty:
        return False
    # Leading zeroes are a strong indication of codes rather than quantities.
    leading_zero = sum(
        1 for value in non_empty
        if re.fullmatch(r"[+-]?0\d+", value)
    )
    return leading_zero / len(non_empty) >= 0.5


def default_semantic_description(table: str, column: str) -> str:
    readable = column.replace("_", " ").strip()
    table_readable = Path(table).stem.replace("_", " ")
    return f"{readable} recorded in {table_readable}"


def semantic_field_name(table: str, column: str) -> str:
    # Logical SQL names may contain dots (schema.table). Strip only known
    # file extensions. Path.stem would silently collapse those namespaces.
    domain = table
    for suffix in (".csv.gz", ".parquet", ".csv"):
        if domain.lower().endswith(suffix):
            domain = domain[:-len(suffix)]
            break
    return f"{domain}.{column}"


def source_table_name(path: Path) -> str:
    for suffix in (".csv.gz", ".parquet", ".csv"):
        if path.name.lower().endswith(suffix):
            return path.name[:-len(suffix)]
    return path.name


def validate_table_names(names: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name.casefold() in seen:
            raise ValueError(f"duplicate logical table name: {name}. Use distinct source table names")
        seen.add(name.casefold())


def csv_text_stream(path: Path):
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8-sig", newline="")


def validate_csv_header(path: Path) -> list[str]:
    with csv_text_stream(path) as stream:
        header = next(csv.reader(stream), [])
    if not header or any(not name.strip() for name in header):
        raise ValueError(f"CSV requires non-empty column names: {path}")
    if len({name.casefold() for name in header}) != len(header):
        raise ValueError(f"duplicate CSV column names: {path}")
    return header


def numeric_storage_type(value: str) -> bool:
    """Recognize scalar SQL numbers. Never arrays, structs or INTERVAL."""
    return bool(re.fullmatch(
        r"(?:U?(?:TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT)|INT|INT[1248]|FLOAT|DOUBLE|REAL|(?:DECIMAL|NUMERIC)\(\d+,\s*\d+\))",
        value.upper().strip(),
    ))


def normalized_unit(value: str) -> str:
    # SI prefixes are case-sensitive (mg != Mg). Normalize only typography.
    return value.strip().replace("µ", "μ")


class UnitConversionError(ValueError):
    """An unresolved unit needs review. Never means a negative match."""


class UnitConverter:
    """Library-backed, case-sensitive conversions without chemistry contexts.

    Explicit aliases belong in reviewed dataset metadata. UCUM annotations are
    not silently stripped: they may identify clinically different substances.
    """
    def __init__(self, aliases: dict[str, str] | None = None):
        from fractions import Fraction
        from ucumvert import PintUcumRegistry
        self.registry = PintUcumRegistry(non_int_type=Fraction)
        self.aliases = aliases or {}

    def _unit(self, text):
        text = normalized_unit(self.aliases.get(text, text))
        if not text or "{" in text or "}" in text:
            raise UnitConversionError("missing or annotated unit requires clinician review")
        try:
            return self.registry.parse_units(text), 1
        except Exception:
            try:
                quantity = self.registry.from_ucum(text)
                return quantity.units, quantity.magnitude
            except Exception as exc:
                raise UnitConversionError(f"unknown unit {text!r}. Clinician review required") from exc

    def convert(self, value, source: str, target: str):
        from decimal import localcontext
        with localcontext() as context:
            context.prec = 80
            return self._convert(value, source, target)

    def _convert(self, value, source: str, target: str):
        from decimal import Decimal
        number = Decimal(str(value))
        if not number.is_finite():
            raise UnitConversionError("non-finite numeric value")
        # Identical, explicit units do not require interpretation by the library.
        if normalized_unit(source) == normalized_unit(target) and source.strip():
            return number
        try:
            source_unit, source_scale = self._unit(source)
            target_unit, target_scale = self._unit(target)
            from fractions import Fraction
            # Rational library definitions avoid affine cancellation residue
            # (e.g. 32 degF must compare equal to 0 degC).
            result = self.registry.Quantity(Fraction(number) * Fraction(str(source_scale)), source_unit).to(target_unit)
            rational = Fraction(result.magnitude) / Fraction(str(target_scale))
            from decimal import localcontext
            with localcontext() as context:
                context.prec = 38
                converted = Decimal(rational.numerator) / Decimal(rational.denominator)
            if not converted.is_finite():
                raise UnitConversionError("non-finite converted value")
            return converted
        except UnitConversionError:
            raise
        except Exception as exc:
            raise UnitConversionError(
                f"cannot convert {source!r} to {target!r}. Incompatible dimensions or substance-dependent conversion. Clinician review required"
            ) from exc
