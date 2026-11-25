"""Small, dependency-free client and arithmetic-only executor for CPU-first research."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import json
import re
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class InvalidExpression(ValueError):
    pass


class InvalidModelOutput(ValueError):
    pass


def exact_arithmetic(expression: str) -> Fraction:
    """Evaluate ONLY +, -, *, / and parentheses. Never use eval/exec."""
    if not isinstance(expression, str) or not (0 < len(expression) <= 200):
        raise InvalidExpression("Expression must have 1 to 200 characters")
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        raise InvalidExpression("Only digits, decimals and + - * / parentheses allowed")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise InvalidExpression("Expression failed to parse") from exc
    if sum(1 for _ in ast.walk(tree)) > 80:
        raise InvalidExpression("Expression too complex")

    def bounded(fr: Fraction) -> Fraction:
        if abs(fr.numerator) > 10**18 or fr.denominator > 10**18:
            raise InvalidExpression("Intermediate result too large")
        return fr

    def walk(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > 24:
            raise InvalidExpression("Expression too deeply nested")
        if isinstance(node, ast.Expression):
            return walk(node.body, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return bounded(Fraction(str(node.value)))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand, depth + 1)
            return bounded(-value if isinstance(node.op, ast.USub) else value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = walk(node.left, depth + 1), walk(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return bounded(left + right)
            if isinstance(node.op, ast.Sub):
                return bounded(left - right)
            if isinstance(node.op, ast.Mult):
                return bounded(left * right)
            if right == 0:
                raise InvalidExpression("Division by zero")
            return bounded(left / right)
        raise InvalidExpression("Unsafe or unsupported expression")
    return walk(tree)


def json_object(text: str) -> dict:
    """Return first parseable JSON object; handles surrounding model prose."""
    if not isinstance(text, str):
        raise InvalidModelOutput("Non-text output")
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
    raise InvalidModelOutput("No JSON object found")


def parse_number(value: object) -> Fraction:
    if type(value) is not str and type(value) is not int and type(value) is not float:
        raise InvalidModelOutput("Answer must be a numerical string or number")
    txt = str(value).replace(",", "").strip()
    if len(txt) > 60 or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", txt):
        raise InvalidModelOutput("Answer isn't a plain number")
    try:
        return Fraction(Decimal(txt))
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise InvalidModelOutput("Unparseable number") from exc


def gsm8k_gold(value: str) -> Fraction:
    if "####" not in value:
        raise ValueError("Expected GSM8K answer marker ####")
    return parse_number(value.rsplit("####", 1)[1])


def formatted_fraction(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


BASELINE_PROMPT = """Solve the arithmetic word problem. Output ONLY a JSON object of the form {"answer":"NUMBER"}. The NUMBER must be a plain integer or decimal; do not include units, prose, or markdown. If uncertain, try your best."""
HYBRID_PROMPT = """Translate this arithmetic word problem into ONE numerical expression that computes the requested answer. Use ONLY digits, decimal points, parentheses, +, -, *, /, and spaces. Return ONLY JSON {"expression":"EXPR"}. Never calculate the final answer yourself. If a single numerical expression cannot faithfully represent the problem, return {"expression":null}. Do not use variable names, percentage signs, functions, or code."""


@dataclass
class Completion:
    content: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove reasoning-model <think> blocks; an unclosed block (truncated) leaves nothing."""
    text = THINK_BLOCK.sub("", text)
    lower = text.lower()
    if "<think>" in lower:
        text = text[:lower.index("<think>")]
    return text.strip()


class LocalLlama:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: int = 180):
        # Restrict to loopback: benchmark data should never go to remote providers.
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Only a local HTTP llama.cpp server is supported")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        models = self._get("/v1/models")
        available = models.get("data", [])
        if not available:
            raise RuntimeError("Server returned no model. Load a local GGUF model first.")
        self.model = available[0]["id"]

    def _get(self, endpoint: str) -> dict:
        with urlopen(self.base_url + endpoint, timeout=15) as response:
            return json.load(response)

    @property
    def disables_thinking(self) -> bool:
        return "qwen3" in self.model.lower()

    def chat(self, prompt: str, question: str, max_tokens: int = 384, *,
             temperature: float = 0.0, seed: int | None = None) -> Completion:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        system = prompt + ("\n/no_think" if self.disables_thinking else "")
        request_body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            request_body["seed"] = int(seed)
        request = Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (URLError, HTTPError, TimeoutError) as exc:
            raise RuntimeError(f"Local inference request failed: {exc}") from exc
        latency = time.perf_counter() - start
        content = strip_thinking(result["choices"][0]["message"].get("content") or "")
        usage = result.get("usage", {})
        return Completion(content, latency, usage.get("prompt_tokens"), usage.get("completion_tokens"))


def solve(client: LocalLlama, question: str, mode: str, max_tokens: int = 384, *,
          system_prompt: str | None = None, user_prompt: str | None = None,
          temperature: float | None = None, seed: int | None = None) -> dict:
    if mode not in ("baseline", "hybrid"):
        raise ValueError("mode must be baseline or hybrid")
    prompt = BASELINE_PROMPT if mode == "baseline" else HYBRID_PROMPT
    sampling = {}
    if temperature is not None:
        sampling["temperature"] = temperature
    if seed is not None:
        sampling["seed"] = seed
    result = client.chat(system_prompt or prompt, user_prompt or question, max_tokens=max_tokens, **sampling)
    record = {
        "mode": mode,
        "prediction": None,
        "expression": None,
        "latency_s": round(result.latency_s, 4),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "output_valid": False,
        "error": None,
    }
    try:
        obj = json_object(result.content)
        if mode == "baseline":
            if "answer" not in obj:
                raise InvalidModelOutput("Missing answer field")
            number = parse_number(obj["answer"])
        else:
            expression = obj.get("expression")
            if expression is None:
                raise InvalidModelOutput("Model abstained from expression")
            number = exact_arithmetic(expression)
            record["expression"] = expression
        record["prediction"] = formatted_fraction(number)
        record["output_valid"] = True
    except (InvalidExpression, InvalidModelOutput) as exc:
        record["error"] = str(exc)
    # A mathematically valid expression does NOT prove it faithfully reflects the question.
    return record
