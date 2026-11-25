"""Deterministic *restricted-language* semantic checks for v0.4.

Only the fully anchored, documented arithmetic problem templates are supported.
This is NOT natural-language entailment or general semantic verification.
The check compares the entire computation graph, not merely its final answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re

from .proof import verify_plan, number

NUM = r"(?P<{name}>\d{{1,8}}(?:\.\d{{1,3}})?)"


def n(name: str) -> str:
    return NUM.format(name=name)


def pattern(expr: str) -> re.Pattern:
    return re.compile(expr, re.IGNORECASE | re.ASCII)


# Full-string matching: unrecognized wording, negation, and extra clauses abstain.
TEMPLATES: tuple[tuple[str, re.Pattern], ...] = (
    ("combine_then_distribute", pattern(
        rf"[a-z]+ has {n('a')} (?P<item>[a-z]+), buys {n('b')} more, and "
        rf"(?:puts|shares) them equally into {n('c')} (?:boxes|bags|groups)\. ?"
    )),
    ("combine_then_distribute", pattern(
        rf"[a-z]+ has {n('a')} (?P<item>[a-z]+) and buys {n('b')} more\. "
        rf"she splits them equally into {n('c')} boxes\. "
        rf"how many (?P=item) are in each box\?"
    )),
    ("bus_boarding", pattern(
        rf"a bus has {n('a')} passengers\. at the first stop, {n('b')} leave "
        rf"and {n('c')} board\. how many passengers now\?"
    )),
    ("purchase_change", pattern(
        rf"a store charges \${n('a')} per (?P<item>[a-z]+)\. you buy {n('b')} "
        rf"(?P=item)s and pay with \${n('c')}\. how much change\?"
    )),
    ("hourly_output", pattern(
        rf"a factory produces {n('a')} (?P<item>[a-z]+) per hour for {n('b')} hours\. "
        rf"it discards {n('c')} defective (?P=item)\. how many remain\?"
    )),
)


@dataclass(frozen=True)
class Template:
    name: str
    expected_steps: tuple[tuple[str, tuple[tuple[str, Fraction | int], ...]], ...]
    answer: Fraction


def classify(question: str) -> Template | None:
    """Parse only a closed set of recognized word-problem schemas."""
    if not isinstance(question, str) or len(question) > 4000:
        return None
    # Normalize whitespace, not meaning. Do not remove words or trailing clauses.
    text = " ".join(question.strip().split())
    for name, regex in TEMPLATES:
        match = regex.fullmatch(text)
        if match is None:
            continue
        a, b = number(match["a"]), number(match["b"])
        c = number(match["c"])
        if name == "combine_then_distribute":
            if c == 0:
                return None
            steps = (("add", (("literal", a), ("literal", b))),
                     ("divide", (("ref", 0), ("literal", c))))
            answer = (a + b) / c
        elif name == "bus_boarding":
            steps = (("subtract", (("literal", a), ("literal", b))),
                     ("add", (("ref", 0), ("literal", c))))
            answer = a - b + c
        elif name == "purchase_change":
            # Price per item * quantity, followed by cash minus cost.
            steps = (("multiply", (("literal", a), ("literal", b))),
                     ("subtract", (("literal", c), ("ref", 0))))
            answer = c - a*b
        else:  # hourly_output
            steps = (("multiply", (("literal", a), ("literal", b))),
                     ("subtract", (("ref", 0), ("literal", c))))
            answer = a*b - c
        return Template(name, steps, answer)
    return None


def verify_semantic_plan(question: str, plan: object) -> dict:
    """Exact proof replay plus strict-template graph correspondence.

    Returns `valid` only for verified arithmetic, for backward compatibility.
    `semantically_verified` means *matches the supported template's operation
    graph*, NOT proof that arbitrary English was understood correctly.
    """
    checked = verify_plan(question, plan)
    template = classify(question)
    checked["semantic_supported"] = template is not None
    checked["semantic_template"] = template.name if template else None
    checked["semantically_verified"] = False
    checked["semantic_error"] = None
    if not checked["valid"]:
        checked["semantic_error"] = "Arithmetic proof invalid"
        return checked
    if template is None:
        checked["semantic_error"] = "Unsupported wording: restricted grammar did not match"
        return checked
    steps = plan["steps"]
    if len(steps) != len(template.expected_steps):
        checked["semantic_error"] = "Plan has a different number of operations from the matched template"
        return checked
    for index, (step, expected) in enumerate(zip(steps, template.expected_steps)):
        operation, required_inputs = expected
        if step["op"] != operation:
            checked["semantic_error"] = f"Step {index + 1}: operation does not match the problem template"
            return checked
        actual: list[tuple[str, Fraction | int]] = []
        for operand in step["inputs"]:
            if "literal" in operand:
                actual.append(("literal", number(operand["literal"])))
            else:
                refs = [s["id"] for s in steps[:index]]
                actual.append(("ref", refs.index(operand["ref"])))
        # Order can be swapped only for commutative operations.
        if operation in ("add", "multiply"):
            ok = sorted(actual, key=str) == sorted(required_inputs, key=str)
        else:
            ok = tuple(actual) == required_inputs
        if not ok:
            checked["semantic_error"] = f"Step {index + 1}: operand roles do not match the problem template"
            return checked
    if Fraction(checked["answer"]) != template.answer:
        checked["semantic_error"] = "Computed answer conflicts with template result"
        return checked
    checked["semantically_verified"] = True
    checked["verification_scope"] = (
        "Exact arithmetic and graph correspondence to a fully matched restricted "
        "word-problem template. No general natural-language semantic proof."
    )
    return checked
