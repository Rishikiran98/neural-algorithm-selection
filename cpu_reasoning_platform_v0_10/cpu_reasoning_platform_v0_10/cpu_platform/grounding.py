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


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "dozen": 12, "dozens": 12, "triple": 3, "tripled": 3,
    "thrice": 3, "quadruple": 4, "third": 3, "thirds": 3, "quarter": 4, "quarters": 4,
}
# v0.6: conversion constants, each allowed only when BOTH unit words occur in the question.
UNIT_CONSTANTS = (
    (r"\bhours?\b", r"\bminutes?\b", 60), (r"\bminutes?\b", r"\bseconds?\b", 60),
    (r"\bdays?\b", r"\bhours?\b", 24), (r"\bweeks?\b", r"\bdays?\b", 7),
    (r"\byears?\b", r"\bmonths?\b", 12), (r"\byears?\b", r"\bweeks?\b", 52),
    (r"\byears?\b", r"\bdays?\b", 365), (r"\bdollars?\b|\$", r"\bcents?\b", 100),
    (r"\b(?:feet|foot)\b", r"\binch(?:es)?\b", 12), (r"\byards?\b", r"\b(?:feet|foot)\b", 3),
    (r"\bpounds?\b", r"\bounces?\b", 16), (r"\bgallons?\b", r"\bquarts?\b", 4),
    (r"\bmeters?\b", r"\bcentimeters?\b", 100), (r"\bkilometers?\b", r"\bmeters?\b", 1000),
    (r"\bkilograms?\b", r"\bgrams?\b", 1000), (r"\bliters?\b", r"\bmilliliters?\b", 1000),
)


def allowed_numbers(question: str) -> set[Fraction]:
    """Literal values and narrowly defined conventional conversion constants.

    v0.6 adds number words ("three", "a dozen") and unit-conversion constants that
    are triggered only when both units are mentioned (e.g. hours + minutes -> 60).
    This is still a lexical screen: an allowed number may be used wrongly.
    """
    source = _number_set(question)
    if re.search(r"\b(?:percent|percentage|per\s*cent)\b|%", question, flags=re.I):
        source.update((Fraction(1), Fraction(100)))
    if re.search(r"\b(?:twice|double|doubled|half|halved|two\s+times)\b", question, flags=re.I):
        source.add(Fraction(2))
    for word in re.findall(r"[a-z]+", question.lower()):
        if word in NUMBER_WORDS:
            source.add(Fraction(NUMBER_WORDS[word]))
    for first, second, value in UNIT_CONSTANTS:
        if re.search(first, question, flags=re.I) and re.search(second, question, flags=re.I):
            source.add(Fraction(value))
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
