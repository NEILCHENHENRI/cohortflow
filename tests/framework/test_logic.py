import pytest

from framework.logic import (
    LogicExpression,
    append_conjuncts,
    prune_predicates,
    validate_unique_predicates,
)


def test_nested_boolean_expression_round_trips_and_renders():
    expression = LogicExpression.and_(
        LogicExpression.predicate("adult", "patient is over 40"),
        LogicExpression.or_(
            LogicExpression.predicate("diabetes", "patient has diabetes"),
            LogicExpression.predicate("hypertension", "patient has hypertension"),
        ),
        LogicExpression.not_(
            LogicExpression.predicate("malignancy", "patient has active malignancy")
        ),
    )

    restored = LogicExpression.from_dict(expression.to_dict())
    assert restored == expression
    assert restored.predicate_ids() == (
        "adult", "diabetes", "hypertension", "malignancy"
    )
    assert restored.contains_not()
    assert "OR" in restored.render()
    assert "NOT" in restored.render()


def test_operator_invariants_are_semantically_validated_after_schema_generation():
    with pytest.raises(ValueError, match="AND requires at least two children"):
        LogicExpression.and_(LogicExpression.predicate("a", "A"))
    with pytest.raises(ValueError, match="NOT requires exactly one child"):
        LogicExpression("not", children=())


def test_prune_and_append_preserve_a_valid_tree():
    expression = LogicExpression.or_(
        LogicExpression.predicate("a", "A"),
        LogicExpression.predicate("b", "B"),
    )
    pruned = prune_predicates(expression, {"a"})
    assert pruned == LogicExpression.predicate("b", "B")

    expanded = append_conjuncts(
        pruned,
        [LogicExpression.predicate("c", "C")],
    )
    assert expanded is not None
    assert expanded.kind == "and"
    assert expanded.predicate_ids() == ("b", "c")


def test_duplicate_predicate_ids_are_rejected():
    expression = LogicExpression.and_(
        LogicExpression.predicate("same", "A"),
        LogicExpression.predicate("same", "B"),
    )
    with pytest.raises(ValueError, match="duplicate predicate_id"):
        validate_unique_predicates(expression)


def test_flat_boolean_parser_builds_nested_ast_with_precedence():
    from framework.logic import PredicateText, parse_boolean_expression

    expression = parse_boolean_expression(
        "adult AND (diabetes OR hypertension) AND NOT malignancy",
        [
            PredicateText("adult", "patient is over 40"),
            PredicateText("diabetes", "patient has diabetes"),
            PredicateText("hypertension", "patient has hypertension"),
            PredicateText("malignancy", "patient has malignancy"),
        ],
    )

    assert expression.kind == "and"
    assert expression.children[1].kind == "or"
    assert expression.children[2].kind == "not"
    assert expression.render_boolean() == "(adult AND (diabetes OR hypertension) AND NOT malignancy)"


def test_flat_boolean_parser_rejects_unknown_omitted_and_duplicate_predicates():
    from framework.logic import (
        BooleanExpressionSyntaxError,
        PredicateText,
        parse_boolean_expression,
    )

    predicates = [PredicateText("a", "A"), PredicateText("b", "B")]
    with pytest.raises(BooleanExpressionSyntaxError, match="unknown predicate"):
        parse_boolean_expression("a AND c", predicates)
    with pytest.raises(BooleanExpressionSyntaxError, match="omits predicates"):
        parse_boolean_expression("a", predicates)
    with pytest.raises(BooleanExpressionSyntaxError, match="only once"):
        parse_boolean_expression("a OR a OR b", predicates)
