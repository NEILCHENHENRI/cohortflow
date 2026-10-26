from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping


LOGIC_KINDS = {"predicate", "and", "or", "not"}
_BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}
_TOKEN_RE = re.compile(r"\s*(\(|\)|[A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True, slots=True)
class LogicExpression:
    """Framework-level Boolean expression with natural-language leaves.

    Domain parsers may keep leaf text in natural language and compile the leaf
    separately. The execution layer therefore gets a stable symbolic tree
    without forcing the language model to generate a recursive JSON object.
    """

    kind: str
    predicate_id: str | None = None
    text: str | None = None
    children: tuple["LogicExpression", ...] = ()
    node_id: str | None = None

    def __post_init__(self) -> None:
        kind = self.kind.casefold().strip()
        object.__setattr__(self, "kind", kind)
        if kind not in LOGIC_KINDS:
            raise ValueError(f"unsupported logic kind: {self.kind!r}")

        if kind == "predicate":
            if self.node_id is not None:
                raise ValueError("predicate nodes use predicate_id, not node_id")
            if self.children:
                raise ValueError("predicate nodes cannot have children")
            if self.predicate_id is None or not self.predicate_id.strip():
                raise ValueError("predicate nodes require predicate_id")
            if self.text is None or not self.text.strip():
                raise ValueError("predicate nodes require natural-language text")
            object.__setattr__(self, "predicate_id", self.predicate_id.strip())
            object.__setattr__(self, "text", self.text.strip())
            return

        if self.predicate_id is not None or self.text is not None:
            raise ValueError("operator nodes cannot carry predicate fields")
        if self.node_id is not None:
            node_id = self.node_id.strip()
            if not node_id:
                raise ValueError("logic node_id cannot be blank")
            object.__setattr__(self, "node_id", node_id)
        if kind == "not":
            if len(self.children) != 1:
                raise ValueError("NOT requires exactly one child")
        elif len(self.children) < 2:
            raise ValueError(f"{kind.upper()} requires at least two children")

    @classmethod
    def predicate(cls, predicate_id: str, text: str) -> "LogicExpression":
        return cls("predicate", predicate_id=predicate_id, text=text)

    @classmethod
    def and_(cls, *children: "LogicExpression", node_id: str | None = None) -> "LogicExpression":
        return cls("and", children=tuple(children), node_id=node_id)

    @classmethod
    def or_(cls, *children: "LogicExpression", node_id: str | None = None) -> "LogicExpression":
        return cls("or", children=tuple(children), node_id=node_id)

    @classmethod
    def not_(cls, child: "LogicExpression", node_id: str | None = None) -> "LogicExpression":
        return cls("not", children=(child,), node_id=node_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "predicate_id": self.predicate_id,
            "text": self.text,
            "node_id": self.node_id,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LogicExpression":
        if not isinstance(payload, dict):
            raise TypeError("logic node must be an object")
        kind = str(payload.get("kind", "")).casefold()
        raw_children = payload.get("children", [])
        if not isinstance(raw_children, list):
            raise TypeError("logic children must be a list")
        children = tuple(cls.from_dict(item) for item in raw_children)
        predicate_id = payload.get("predicate_id")
        text = payload.get("text")
        node_id = payload.get("node_id")
        if predicate_id is not None and not isinstance(predicate_id, str):
            raise TypeError("predicate_id must be text or null")
        if text is not None and not isinstance(text, str):
            raise TypeError("logic text must be text or null")
        if node_id is not None and not isinstance(node_id, str):
            raise TypeError("logic node_id must be text or null")
        return cls(
            kind=kind,
            predicate_id=predicate_id,
            text=text,
            children=children,
            node_id=node_id,
        )

    def leaves(self) -> tuple["LogicExpression", ...]:
        if self.kind == "predicate":
            return (self,)
        result: list[LogicExpression] = []
        for child in self.children:
            result.extend(child.leaves())
        return tuple(result)

    def predicate_ids(self) -> tuple[str, ...]:
        return tuple(leaf.predicate_id or "" for leaf in self.leaves())

    def contains_not(self) -> bool:
        return self.kind == "not" or any(
            child.contains_not() for child in self.children
        )

    def render(self) -> str:
        if self.kind == "predicate":
            return f"{self.predicate_id}: {self.text}"
        if self.kind == "not":
            return f"NOT ({self.children[0].render()})"
        joiner = f" {self.kind.upper()} "
        return "(" + joiner.join(child.render() for child in self.children) + ")"

    def render_boolean(self) -> str:
        """Render only stable predicate IDs and Boolean operators."""
        if self.kind == "predicate":
            return self.predicate_id or ""
        if self.kind == "not":
            child = self.children[0]
            rendered = child.render_boolean()
            if child.kind == "predicate":
                return f"NOT {rendered}"
            return f"NOT ({rendered})"
        joiner = f" {self.kind.upper()} "
        rendered = joiner.join(child.render_boolean() for child in self.children)
        return f"({rendered})"

    def remap_predicate_ids(self, mapping: dict[str, str]) -> "LogicExpression":
        if self.kind == "predicate":
            predicate_id = self.predicate_id or ""
            return LogicExpression.predicate(
                mapping.get(predicate_id, predicate_id),
                self.text or "",
            )
        return LogicExpression(
            self.kind,
            children=tuple(
                child.remap_predicate_ids(mapping) for child in self.children
            ),
            node_id=self.node_id,
        )


def assign_logic_node_ids(expression: LogicExpression | None) -> LogicExpression | None:
    """Assign stable gN refs to Boolean operator nodes that do not have one.

    Existing refs are preserved. New refs are allocated in deterministic preorder,
    which lets older TaskSpecs participate in ref-based corrections without a migration.
    """
    if expression is None:
        return None
    used = {node.node_id for node in _operator_nodes(expression) if node.node_id}
    counter = 1

    def next_id() -> str:
        nonlocal counter
        while f"g{counter}" in used:
            counter += 1
        value = f"g{counter}"
        used.add(value)
        counter += 1
        return value

    def visit(node: LogicExpression) -> LogicExpression:
        if node.kind == "predicate":
            return node
        node_id = node.node_id or next_id()
        return LogicExpression(
            node.kind,
            children=tuple(visit(child) for child in node.children),
            node_id=node_id,
        )

    return visit(expression)


def _operator_nodes(expression: LogicExpression) -> tuple[LogicExpression, ...]:
    if expression.kind == "predicate":
        return ()
    result = [expression]
    for child in expression.children:
        result.extend(_operator_nodes(child))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PredicateText:
    predicate_id: str
    text: str

    def __post_init__(self) -> None:
        predicate_id = self.predicate_id.strip()
        text = self.text.strip()
        if not predicate_id:
            raise ValueError("predicate_id cannot be empty")
        if predicate_id.upper() in _BOOLEAN_OPERATORS:
            raise ValueError(f"predicate_id is reserved: {predicate_id}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", predicate_id):
            raise ValueError(
                "predicate_id must start with a letter/underscore and contain "
                "only letters, digits, and underscores"
            )
        if not text:
            raise ValueError("predicate text cannot be empty")
        object.__setattr__(self, "predicate_id", predicate_id)
        object.__setattr__(self, "text", text)


class BooleanExpressionSyntaxError(ValueError):
    pass


class _BooleanParser:
    def __init__(self, source: str, predicates: Mapping[str, str]) -> None:
        self.source = source
        self.predicates = predicates
        self.tokens = self._tokenize(source)
        self.index = 0

    @staticmethod
    def _tokenize(source: str) -> tuple[str, ...]:
        if not source.strip():
            raise BooleanExpressionSyntaxError("Boolean expression cannot be empty")
        tokens: list[str] = []
        position = 0
        while position < len(source):
            match = _TOKEN_RE.match(source, position)
            if match is None:
                tail = source[position:].strip()
                if not tail:
                    break
                raise BooleanExpressionSyntaxError(
                    f"invalid token near {source[position:position + 20]!r}"
                )
            tokens.append(match.group(1))
            position = match.end()
        if not tokens:
            raise BooleanExpressionSyntaxError("Boolean expression cannot be empty")
        return tuple(tokens)

    def parse(self) -> LogicExpression:
        expression = self._parse_or()
        if self.index != len(self.tokens):
            raise BooleanExpressionSyntaxError(
                f"unexpected token {self.tokens[self.index]!r}"
            )
        return expression

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise BooleanExpressionSyntaxError("unexpected end of expression")
        self.index += 1
        return token

    def _accept(self, operator: str) -> bool:
        token = self._peek()
        if token is not None and token.upper() == operator:
            self.index += 1
            return True
        return False

    def _parse_or(self) -> LogicExpression:
        children = [self._parse_and()]
        while self._accept("OR"):
            children.append(self._parse_and())
        if len(children) == 1:
            return children[0]
        return LogicExpression.or_(*children)

    def _parse_and(self) -> LogicExpression:
        children = [self._parse_factor()]
        while self._accept("AND"):
            children.append(self._parse_factor())
        if len(children) == 1:
            return children[0]
        return LogicExpression.and_(*children)

    def _parse_factor(self) -> LogicExpression:
        if self._accept("NOT"):
            return LogicExpression.not_(self._parse_factor())
        token = self._take()
        if token == "(":
            expression = self._parse_or()
            if self._take() != ")":
                raise BooleanExpressionSyntaxError("missing closing parenthesis")
            return expression
        if token == ")":
            raise BooleanExpressionSyntaxError("unexpected closing parenthesis")
        if token.upper() in {"AND", "OR"}:
            raise BooleanExpressionSyntaxError(f"unexpected operator {token!r}")
        if token not in self.predicates:
            raise BooleanExpressionSyntaxError(
                f"Boolean expression references unknown predicate {token!r}"
            )
        return LogicExpression.predicate(token, self.predicates[token])


def parse_boolean_expression(
    source: str,
    predicates: Iterable[PredicateText] | Mapping[str, str],
    *,
    require_all_predicates: bool = True,
) -> LogicExpression:
    """Parse a tiny Boolean language into :class:`LogicExpression`.

    Grammar (case-insensitive operators)::

        expr   := term (OR term)*
        term   := factor (AND factor)*
        factor := NOT factor | '(' expr ')' | predicate_id

    Predicate text is kept outside the Boolean string. This deliberately keeps
    the LLM-facing representation flat while the framework owns recursive
    structure and validation deterministically.
    """

    if isinstance(predicates, Mapping):
        mapping = {str(key): str(value) for key, value in predicates.items()}
    else:
        items = tuple(predicates)
        mapping = {item.predicate_id: item.text for item in items}
        if len(mapping) != len(items):
            raise ValueError("predicate IDs must be unique")
    if not mapping:
        raise ValueError("at least one predicate is required")

    expression = _BooleanParser(source, mapping).parse()
    ids = expression.predicate_ids()
    if len(ids) != len(set(ids)):
        raise BooleanExpressionSyntaxError(
            "each predicate_id may appear only once in the Boolean expression"
        )
    if require_all_predicates:
        missing = set(mapping) - set(ids)
        if missing:
            raise BooleanExpressionSyntaxError(
                "Boolean expression omits predicates: " + ", ".join(sorted(missing))
            )
    validate_unique_predicates(expression)
    return expression


def validate_unique_predicates(expression: LogicExpression) -> None:
    seen: set[str] = set()
    for predicate_id in expression.predicate_ids():
        if predicate_id in seen:
            raise ValueError(f"duplicate predicate_id in logic tree: {predicate_id}")
        seen.add(predicate_id)


def prune_predicates(
    expression: LogicExpression,
    removed_ids: Iterable[str],
) -> LogicExpression | None:
    """Remove leaves and simplify now-degenerate Boolean operators."""

    removed = set(removed_ids)
    if expression.kind == "predicate":
        return None if expression.predicate_id in removed else expression

    children = tuple(
        child
        for child in (
            prune_predicates(item, removed) for item in expression.children
        )
        if child is not None
    )
    if not children:
        return None
    if expression.kind == "not":
        return LogicExpression.not_(children[0], node_id=expression.node_id)
    if len(children) == 1:
        return children[0]
    return LogicExpression(expression.kind, children=children, node_id=expression.node_id)


def append_conjuncts(
    expression: LogicExpression | None,
    children: Iterable[LogicExpression],
) -> LogicExpression | None:
    additions = tuple(children)
    if not additions:
        return expression
    if expression is None:
        if len(additions) == 1:
            return additions[0]
        return LogicExpression.and_(*additions)
    if expression.kind == "and":
        return LogicExpression.and_(*expression.children, *additions, node_id=expression.node_id)
    return LogicExpression.and_(expression, *additions)

