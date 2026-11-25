"""Conservative lexical number-provenance screen for arithmetic expressions.

This only checks whether the numbers used in a proposed expression can be found
in the question (plus a small set of conventional conversion constants).
It is NOT a semantic correctness checker, nor a language understanding model.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
import re

from cpu_first.core import InvalidExpression, exact_arithmetic

# A number may be followed by sentence punctuation ("pay with $50."), but not by
# a word character or another decimal part ("v1.2.3", "5th"). v0.5.1 fix: the
# earlier (?![\w.]) lookahead silently dropped every sentence-final number.
NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w|\.\d)")


@dataclass(frozen=True)
class GroundingResult:
    passed: bool
    ungrounded: tuple[str, ...]
    rule: str = "numeric_literal_membership"


def _number_set(text: str) -> set[Fraction]:
    return {Fraction(m.group(0).replace(",", "")) for m in NUMBER.finditer(text)}


def allowed_numbers(question: str) -> set[Fraction]:
    """Literal values and narrowly defined conventional conversion constants."""
    source = _number_set(question)
    if re.search(r"\b(?:percent|percentage|per\s*cent)\b|%", question, flags=re.I):
        source.update((Fraction(1), Fraction(100)))
    if re.search(r"\b(?:twice|double|half|halved|two\s+times)\b", question, flags=re.I):
        source.add(Fraction(2))
    return source


def check_numeric_grounding(question: str, expression: str) -> GroundingResult:
    """Inspect ONLY numeric literals, not the correctness of operations/order.

    Conventional conversion constants used for percentages and words like
    'twice' are accepted; passing this test is weak evidence, not proof.
    """
    exact_arithmetic(expression)  # enforce AST whitelist and resource bounds first
    source = allowed_numbers(question)
    tree = ast.parse(expression, mode="eval")
    expression_numbers = {Fraction(str(node.value)) for node in ast.walk(tree)
                          if isinstance(node, ast.Constant) and type(node.value) in (int, float)}
    missing = sorted(expression_numbers - source)
    return GroundingResult(
        passed=not missing,
        ungrounded=tuple(str(item.numerator) if item.denominator == 1 else str(item)
                         for item in missing),
    )
