"""Safe, complete propositional entailment checker for a bounded variable set.

This checks *formal formulas supplied as input*, not the faithfulness of a
natural-language translation. No model calls or eval/exec are used.

Syntax: variables [A-Za-z][A-Za-z0-9_]*, true/false, !, &, |, ->, ( ).
Operator precedence: ! > & > | > -> (implication right-associative).
"""
from __future__ import annotations
from itertools import product
import re

MAX_FORMULA_CHARS = 512
MAX_TOKENS = 160
MAX_PREMISES = 12
MAX_VARIABLES = 12
TOKEN = re.compile(r"\s*(->|[!&|()]|[A-Za-z][A-Za-z0-9_]{0,31})")

class LogicInputError(ValueError):
    pass


def _tokenize(formula: str) -> list[str]:
    if not isinstance(formula, str) or not formula.strip() or len(formula) > MAX_FORMULA_CHARS:
        raise LogicInputError("Formula must be nonempty and at most 512 characters")
    tokens = []
    offset = 0
    while offset < len(formula):
        match = TOKEN.match(formula, offset)
        if match is None:
            if not formula[offset:].strip():
                break
            raise LogicInputError("Unsupported token in formula")
        tokens.append(match.group(1))
        if len(tokens) > MAX_TOKENS:
            raise LogicInputError("Formula contains too many tokens")
        offset = match.end()
    return tokens


class _Parser:
    def __init__(self, formula: str):
        self.tokens = _tokenize(formula)
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def eat(self, token):
        if self.peek() == token:
            self.index += 1
            return True
        return False

    def parse(self):
        expr = self.implies()
        if self.peek() is not None:
            raise LogicInputError("Unexpected token after formula")
        return expr

    def implies(self):
        left = self.or_expr()
        if self.eat("->"):
            return ("implies", left, self.implies())
        return left

    def or_expr(self):
        left = self.and_expr()
        while self.eat("|"):
            left = ("or", left, self.and_expr())
        return left

    def and_expr(self):
        left = self.not_expr()
        while self.eat("&"):
            left = ("and", left, self.not_expr())
        return left

    def not_expr(self):
        if self.eat("!"):
            return ("not", self.not_expr())
        if self.eat("("):
            result = self.implies()
            if not self.eat(")"):
                raise LogicInputError("Unclosed parenthesis")
            return result
        atom = self.peek()
        if atom is None or atom in ("->", "&", "|", "(", ")"):
            raise LogicInputError("Expected a propositional variable")
        self.index += 1
        if atom.lower() in ("true", "false"):
            return ("constant", atom.lower() == "true")
        return ("variable", atom)


def _vars(tree):
    kind = tree[0]
    if kind == "variable":
        return {tree[1]}
    if kind == "constant":
        return set()
    if kind == "not":
        return _vars(tree[1])
    return _vars(tree[1]) | _vars(tree[2])


def _eval(tree, env):
    kind = tree[0]
    if kind == "variable":
        return env[tree[1]]
    if kind == "constant":
        return tree[1]
    if kind == "not":
        return not _eval(tree[1], env)
    if kind == "and":
        return _eval(tree[1], env) and _eval(tree[2], env)
    if kind == "or":
        return _eval(tree[1], env) or _eval(tree[2], env)
    if kind == "implies":
        return not _eval(tree[1], env) or _eval(tree[2], env)
    raise AssertionError("Unknown expression node")


def check_entailment(premises: list[str], conclusion: str) -> dict:
    """Find a countermodel if one exists. Exhaustive for <=12 variables."""
    if not isinstance(premises, list) or len(premises) > MAX_PREMISES or not all(isinstance(p, str) for p in premises):
        raise LogicInputError("premises must be an array of at most 12 formulas")
    parsed = [_Parser(p).parse() for p in premises]
    goal = _Parser(conclusion).parse()
    variables = sorted(set().union(*(_vars(p) for p in parsed), _vars(goal)))
    if len(variables) > MAX_VARIABLES:
        raise LogicInputError("Too many distinct variables (maximum 12)")
    inspected = 0
    satisfying = 0
    for values in product((False, True), repeat=len(variables)):
        env = dict(zip(variables, values))
        inspected += 1
        if all(_eval(p, env) for p in parsed):
            satisfying += 1
            if not _eval(goal, env):
                return {
                    "entailed": False, "counterexample": env,
                    "variables": variables, "assignments_examined": inspected,
                    "satisfying_assignments_examined": satisfying,
                    "premises_satisfiable": True,
                    "scope": "Complete bounded propositional truth-table check; natural-language meaning not checked",
                    "model_calls": 0,
                }
    return {
        "entailed": True, "counterexample": None, "variables": variables,
        "assignments_examined": inspected,
        "satisfying_assignments_examined": satisfying,
        "premises_satisfiable": satisfying > 0,
        "vacuous": satisfying == 0,
        "scope": "Complete bounded propositional truth-table check; natural-language meaning not checked",
        "model_calls": 0,
    }
