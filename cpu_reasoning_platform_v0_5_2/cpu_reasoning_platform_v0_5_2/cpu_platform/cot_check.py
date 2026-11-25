"""Conservative arithmetic checks over *visible, free-form* reasoning text.

Not a semantic proof. Only explicit equations using + - * / and numerical
literals are checkable. A missing equation is NOT positive verification.
Never execute generated Python or arbitrary model text.

v0.8: text is normalized before extraction so ordinary model writing is checkable
(thousands commas, $ amounts, LaTeX \\times/\\frac/\\div, letter x, percentages,
markdown), stated rounding is tolerated ("10 / 3 = 3.33", "≈"), and three
eligibility policies are computed for every chain (see POLICIES).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import re

from cpu_first.core import exact_arithmetic, InvalidExpression, formatted_fraction, parse_number

# Only exact arithmetic claims using an equals sign are eligible. No evaluation
# of prose, function calls, or implicit equations.
RHS = re.compile(r"\s*(?P<sign>=|≈)\s*(?P<rhs>[+-]?\d+(?:\.\d+)?)(?![\w]|\.\d)")
NUMBER_START = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?|\()")
OPERATORS = re.compile(r"[+\-*/]")
ALLOWED_LHS = re.compile(r"^[\d\s+\-*/().]+$")
MAP = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "·": "*", "∗": "*"})
MAX_TEXT = 12000
MAX_EQUATIONS = 40

# strict          v0.7 rule: >=1 equation checked, none wrong (unchecked chains excluded)
# exclude_errors  v0.8 default: excluded only on evidence, i.e. a wrong equation
# answer_supported no wrong equation AND the final answer is the result of a checked one
POLICIES = ("strict", "exclude_errors", "answer_supported")
DEFAULT_POLICY = "exclude_errors"

_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEXT = re.compile(r"\\(?:text|mathrm|mbox)\s*\{([^{}]*)\}")
_PCT_OF = re.compile(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_TIMES_X = re.compile(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def normalize(text: str) -> str:
    """Rewrite common notations into plain arithmetic. Pure text substitution."""
    t = text.translate(MAP)
    for _ in range(3):  # nested \frac
        t = _FRAC.sub(r"((\1)/(\2))", t)
    t = _TEXT.sub(r" \1 ", t)
    t = re.sub(r"\\(?:times|cdot|ast)\b", " * ", t)
    t = re.sub(r"\\div\b", " / ", t)
    t = re.sub(r"\\approx\b", " ≈ ", t)
    t = re.sub(r"\\(?:left|right)\b", "", t)
    t = re.sub(r"\\[()\[\]]", " ", t)      # \( \) \[ \]
    t = re.sub(r"\\[,;! ]", " ", t)        # LaTeX spacing
    t = t.replace("\\$", " ").replace("$", " ")
    t = t.replace("**", " ").replace("__", " ").replace("`", " ")
    t = _THOUSANDS.sub("", t)
    t = _PCT_OF.sub(r"(\1/100*\2)", t)
    t = _PCT.sub(r"(\1/100)", t)
    t = _TIMES_X.sub(" * ", t)
    return t


@dataclass(frozen=True)
class Equation:
    expression: str
    claimed: str
    actual: str | None
    correct: bool | None
    rounded: bool = False      # accepted because the claim is the stated rounding of the result


@dataclass(frozen=True)
class Audit:
    status: str    # checked_pass | arithmetic_error | no_checkable_equations
    checked_count: int
    error_count: int
    equations: tuple[Equation, ...]
    truncated: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def supports(self, answer: str | None) -> bool:
        """True if some correct equation's result equals the answer."""
        if answer is None:
            return False
        try:
            target = Fraction(answer)
        except (ValueError, ZeroDivisionError):
            return False
        return any(eq.correct and eq.actual is not None and Fraction(eq.actual) == target
                   for eq in self.equations)

    def eligible(self, answer: str | None, policy: str = DEFAULT_POLICY) -> bool:
        if answer is None or self.error_count:
            return False
        if policy == "exclude_errors":
            return True
        if policy == "strict":
            return self.checked_count > 0 and not self.truncated
        if policy == "answer_supported":
            return self.supports(answer)
        raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")


def _decimals(text: str) -> int:
    return len(text.split(".", 1)[1]) if "." in text else 0


def _matches(actual: Fraction, claimed: Fraction, claimed_text: str, approx: bool) -> tuple[bool, bool]:
    """(correct, rounded). Exact match, or the claim equals the result rounded to the
    precision the claim states (needs >=1 decimal place, or an explicit ≈)."""
    if actual == claimed:
        return True, False
    places = _decimals(claimed_text.lstrip("+-"))
    if places == 0 and not approx:
        return False, False
    exact = Decimal(actual.numerator) / Decimal(actual.denominator)
    quantum = Decimal(1).scaleb(-places)
    rounded = exact.quantize(quantum, rounding=ROUND_HALF_UP)
    return (Fraction(rounded) == claimed), True


def _candidate_lhs(prefix: str) -> tuple[str, Fraction] | None:
    """Longest valid suffix of text preceding '=' (bounded to 200 chars)."""
    part = prefix[-200:]
    # Hard delimiters: do not reach into previous sentences / lines.
    part = re.split(r"[\n;:=≈!?]", part)[-1]
    options: list[tuple[int, str, Fraction]] = []
    for start in NUMBER_START.finditer(part):
        candidate = part[start.start():].strip()
        if not candidate or not ALLOWED_LHS.fullmatch(candidate):
            continue
        if not OPERATORS.search(candidate):
            continue
        try:
            result = exact_arithmetic(candidate)
        except (InvalidExpression, ValueError, ZeroDivisionError):
            continue
        options.append((len(candidate), candidate, result))
    if not options:
        return None
    _, expression, value = max(options, key=lambda c: c[0])
    return expression, value


def audit_reasoning(content: str) -> Audit:
    """Check every recognized equation; not a general language verifier.

    Unsupported forms are not counted as successes. Returns checked_pass only if
    at least one arithmetic equation was checked and none was wrong.
    """
    raw = content[:MAX_TEXT] if isinstance(content, str) else ""
    normalized = normalize(raw)
    found: list[Equation] = []
    for match in RHS.finditer(normalized):
        if len(found) >= MAX_EQUATIONS:
            break
        lhs = _candidate_lhs(normalized[max(0, match.start() - 200):match.start()])
        if lhs is None:
            continue
        expression, value = lhs
        try:
            claimed = parse_number(match.group("rhs"))
        except (ValueError, TypeError):
            continue
        correct, rounded = _matches(value, claimed, match.group("rhs"), match.group("sign") == "≈")
        found.append(Equation(expression, formatted_fraction(claimed),
                              formatted_fraction(value), correct, rounded and correct))
    errors = sum(eq.correct is False for eq in found)
    status = "arithmetic_error" if errors else ("checked_pass" if found else "no_checkable_equations")
    return Audit(status, len(found), errors, tuple(found),
                 len(content or "") >= MAX_TEXT or len(found) >= MAX_EQUATIONS)
