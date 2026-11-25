"""v0.6 tests: grammar-constrained plans, wider grounding, reasoning arms, preflight,
launcher UX. Offline; the fake server is not a model and proves nothing about accuracy."""
import contextlib
import io
import json
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import unittest

from cpu_first.core import InvalidModelOutput, LocalLlama, last_json_object
from cpu_platform import experiment6, launch
from cpu_platform.constrained import (build_grammar, constrained_answer, literal_text,
                                      question_literals, to_checked_plan)
from cpu_platform.grounding import allowed_numbers
from cpu_platform.hardware import _same_gpu
from cpu_platform.search import sample

try:
    from fake_llama import FakeLlama
except ImportError:
    from tests.fake_llama import FakeLlama

ROOT = Path(__file__).resolve().parent.parent
STORE = "A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?"


# ---------------------------------------------------------------- mini GBNF engine
# Implements the subset build_grammar() emits, following llama.cpp's grammar syntax:
# rule names [a-zA-Z0-9-]+, "quoted" terminals with \" and \\ escapes, |, ( ... ), ?.

TOKEN = re.compile(r'\s*(?:(?P<name>[a-zA-Z0-9-]+)|(?P<str>"(?:\\.|[^"\\])*")|(?P<sym>::=|[|()?]))')


def parse_gbnf(text):
    rules = {}
    for line in text.strip().splitlines():
        name, sep, body = line.partition("::=")
        name = name.strip()
        assert sep and re.fullmatch(r"[a-zA-Z0-9-]+", name), f"bad rule line: {line}"
        assert name not in rules, f"duplicate rule {name}"
        tokens, pos = [], 0
        while pos < len(body):
            if not body[pos:].strip():
                break
            m = TOKEN.match(body, pos)
            assert m, f"cannot tokenize {body[pos:]!r}"
            pos = m.end()
            if m["str"]:
                tokens.append(("str", bytes(m["str"][1:-1], "utf-8").decode("unicode_escape")))
            elif m["name"]:
                tokens.append(("ref", m["name"]))
            else:
                tokens.append(("sym", m["sym"]))
        expr, rest = _alts(tokens)
        assert not rest, f"trailing tokens in {name}: {rest}"
        rules[name] = expr
    assert "root" in rules
    for expr in rules.values():
        for ref in _refs(expr):
            assert ref in rules, f"undefined rule {ref}"
    return rules


def _alts(tokens):
    seqs, seq = [], []
    while tokens and tokens[0] != ("sym", ")"):
        tok = tokens.pop(0)
        if tok == ("sym", "|"):
            seqs.append(seq); seq = []
        elif tok == ("sym", "("):
            inner, tokens = _alts(tokens)
            assert tokens and tokens.pop(0) == ("sym", ")"), "unbalanced ("
            item = ("group", inner)
            if tokens and tokens[0] == ("sym", "?"):
                tokens.pop(0); item = ("opt", inner)
            seq.append(item)
        else:
            assert tok[0] in ("str", "ref"), f"unexpected {tok}"
            seq.append(tok)
    seqs.append(seq)
    return ("alt", seqs), tokens


def _refs(expr):
    kind, val = expr
    if kind == "alt":
        for seq in val:
            for item in seq:
                yield from _refs(item)
    elif kind in ("group", "opt"):
        yield from _refs(val)
    elif kind == "ref":
        yield val


def _match(rules, expr, text, pos):
    kind, val = expr
    if kind == "str":
        if text.startswith(val, pos):
            yield pos + len(val)
    elif kind == "ref":
        yield from _match(rules, rules[val], text, pos)
    elif kind == "group":
        yield from _match(rules, val, text, pos)
    elif kind == "opt":
        yield pos
        yield from _match(rules, val, text, pos)
    else:
        for seq in val:
            yield from _seq(rules, seq, text, pos)


def _seq(rules, seq, text, pos):
    if not seq:
        yield pos
        return
    for mid in _match(rules, seq[0], text, pos):
        yield from _seq(rules, seq[1:], text, mid)


def accepts(rules, text):
    return any(end == len(text) for end in _match(rules, ("ref", "root"), text, 0))


def generate(rules, rng, expr=("ref", "root")):
    kind, val = expr
    if kind == "str":
        return val
    if kind == "ref":
        return generate(rules, rng, rules[val])
    if kind == "group":
        return generate(rules, rng, val)
    if kind == "opt":
        return generate(rules, rng, val) if rng.random() < 0.6 else ""
    return "".join(generate(rules, rng, item) for item in rng.choice(val))


class TestGrammar(unittest.TestCase):
    def setUp(self):
        self.lits = question_literals(STORE)
        self.text = build_grammar(self.lits)
        self.rules = parse_gbnf(self.text)

    def test_grammar_is_well_formed_gbnf(self):
        for lits in (["3"], ["1", "1.5", "100"], [str(i) for i in range(40)]):
            for steps in (1, 3, 8, 12):
                parse_gbnf(build_grammar(lits, steps))

    def test_accepts_valid_plans(self):
        good = ['{"steps":[{"op":"multiply","a":"12","b":"3"},{"op":"subtract","a":"50","b":"s1"}]}',
                '{"steps":[{"op":"add","a":"3","b":"3"}]}']
        for plan in good:
            self.assertTrue(accepts(self.rules, plan), plan)

    def test_rejects_what_the_old_format_got_wrong(self):
        bad = ['{"steps":[{"op":"add","a":"99","b":"3"}]}',                     # invented number
               '{"steps":[{"op":"add","a":"s1","b":"3"}]}',                     # forward reference
               '{"steps":[]}',                                                   # empty / abstain
               '{"steps":[{"op":"power","a":"3","b":"3"}]}',                    # unknown op
               '{"steps":[{"op":"add","a":"3","b":"3","claim":"6"}]}',          # extra field
               '{"steps": [{"op":"add","a":"3","b":"3"}]}',                     # whitespace
               'Sure! {"steps":[{"op":"add","a":"3","b":"3"}]}']                # prose
        for plan in bad:
            self.assertFalse(accepts(self.rules, plan), plan)

    def test_step_limit(self):
        step = '{"op":"add","a":"3","b":"3"}'
        self.assertTrue(accepts(self.rules, '{"steps":[' + ",".join([step] * 8) + "]}"))
        self.assertFalse(accepts(self.rules, '{"steps":[' + ",".join([step] * 9) + "]}"))

    def test_everything_the_grammar_generates_is_checkable(self):
        rng = random.Random(7)
        for _ in range(400):
            text = generate(self.rules, rng)
            self.assertTrue(accepts(self.rules, text))
            try:
                checked = to_checked_plan(STORE, json.loads(text))
            except InvalidModelOutput as exc:
                self.assertIn("division by zero", str(exc))  # the only legal runtime failure
                continue
            self.assertTrue(checked["valid"], (text, checked["error"]))

    def test_no_numbers_no_grammar(self):
        with self.assertRaises(ValueError):
            build_grammar([])


class TestPlanChecking(unittest.TestCase):
    def test_exact_answer_and_dead_step_pruning(self):
        reply = ('{"steps":[{"op":"multiply","a":"12","b":"3"},{"op":"add","a":"3","b":"3"},'
                 '{"op":"subtract","a":"50","b":"s1"}]}')
        checked = to_checked_plan(STORE, json.loads(reply))
        self.assertTrue(checked["valid"])
        self.assertEqual((checked["answer"], checked["dropped_steps"]), ("14", 1))
        self.assertEqual(constrained_answer(STORE, reply), ("14", True, None))

    def test_rational_and_decimal_operands(self):
        q = "A rope is 7.5 meters long and is cut into 2 equal parts. How long is each part?"
        self.assertEqual(constrained_answer(q, '{"steps":[{"op":"divide","a":"7.5","b":"2"}]}')[0], "15/4")

    def test_rejections_without_grammar(self):
        cases = {'{"steps":[{"op":"add","a":"99","b":"3"}]}': "ungrounded",
                 '{"steps":[{"op":"add","a":"s2","b":"3"}]}': "earlier step",
                 '{"steps":[{"op":"divide","a":"3","b":"s1"},{"op":"add","a":"3","b":"3"}]}': "earlier step",
                 '{"steps":[]}': "1 to 8",
                 'no json here': "No JSON"}
        for reply, fragment in cases.items():
            answer, verified, error = constrained_answer(STORE, reply)
            self.assertIsNone(answer)
            self.assertIn(fragment, error)

    def test_literal_text(self):
        from fractions import Fraction
        self.assertEqual([literal_text(Fraction(x)) for x in ("12", "3.5", "0.25", "1000")],
                         ["12", "3.5", "0.25", "1000"])


class TestWiderGrounding(unittest.TestCase):
    def test_number_words_and_units(self):
        q = "Tom buys three dozen eggs and works for 2 hours. How many minutes did he work?"
        nums = {str(x) for x in allowed_numbers(q)}
        self.assertTrue({"3", "12", "2", "60"} <= nums)
        self.assertNotIn("24", nums)  # 'day' not mentioned
        self.assertIn("7", {str(x) for x in allowed_numbers("She runs 5 km a day, every day of the week.")})

    def test_unit_constant_needs_both_units(self):
        self.assertNotIn("60", {str(x) for x in allowed_numbers("He ran for 3 hours.")})

    def test_words_inside_other_words_do_not_count(self):
        self.assertEqual(allowed_numbers("Someone ate at noon, often."), set())


class TestReasoningReplies(unittest.TestCase):
    def test_last_json_wins(self):
        text = 'Step 1 {"draft": 3}. Maybe {"answer":"5"}? No.\nFinal: {"answer":"7"}'
        self.assertEqual(last_json_object(text, "answer"), {"answer": "7"})
        with self.assertRaises(InvalidModelOutput):
            last_json_object("no answer here {\"x\":1}", "answer")


class TestClientAndSearch(unittest.TestCase):
    def setUp(self):
        self.fake = FakeLlama("Qwen3-0.6B", accuracy=0.5)
        self.model = LocalLlama(self.fake.url)

    def tearDown(self):
        self.fake.close()

    def test_grammar_sent_with_constrained_requests_only(self):
        drawn = sample(self.model, STORE, 6, kind="cplan", seed=3)
        self.assertTrue(all("grammar" in r for r in self.fake.requests))
        self.assertTrue(all(s.verified for s in drawn))       # fake honours the grammar
        self.assertIn("14", [s.answer for s in drawn])
        self.assertTrue(any(s.answer != "14" for s in drawn))  # wrong-but-valid plans exist
        sample(self.model, STORE, 1, kind="cot")
        self.assertNotIn("grammar", self.fake.requests[-1])

    def test_no_grounded_numbers_means_no_model_call(self):
        drawn = sample(self.model, "What is the capital of France?", 4, kind="cplan")
        self.assertEqual(self.fake.requests, [])
        self.assertTrue(all(s.answer is None and s.latency_s == 0 for s in drawn))

    def test_cot_parses_final_answer(self):
        drawn = sample(self.model, STORE, 4, kind="cot", temperature=0.7)
        self.assertTrue(all(s.answer is not None for s in drawn))


V06_ARMS = ("small_greedy", "small_cot", "large_greedy", "large_cot", "small_vote", "small_constrained")


def run_cli(args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        code = experiment6.main(args)
    return code, err.getvalue()


class TestExperiment6V06(unittest.TestCase):
    def test_default_arms_end_to_end(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6)
        large = FakeLlama("Qwen3-8B", accuracy=0.8, delay_s=0.03)
        try:
            with TemporaryDirectory() as tmp:
                out = Path(tmp) / "e6.jsonl"
                code, _ = run_cli(["--dataset", str(ROOT / "data" / "smoke.jsonl"), "--limit", "5",
                                   "--small-url", small.url, "--large-url", large.url,
                                   "--samples", "5", "--output", str(out), "--arms", ",".join(V06_ARMS)])
                self.assertEqual(code, 0)
                res = json.loads(out.with_suffix(".summary.json").read_text())["results"]
        finally:
            small.close()
            large.close()
        self.assertEqual(set(res["curves"]), set(V06_ARMS))
        cmp = res["comparisons"]
        for key in ("small_constrained_vs_large_greedy", "small_constrained_vs_large_cot",
                    "small_vote_vs_large_cot", "verifier_effect_small_constrained_vs_small_vote"):
            self.assertIn(key, cmp)
        self.assertEqual(cmp["grammar_violation_rate"], 0.0)
        self.assertEqual(res["curves"]["small_constrained"][-1]["verified_sample_rate"], 1.0)

    def test_preflight_catches_ignored_grammar(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6, honour_grammar=False)
        try:
            with TemporaryDirectory() as tmp:
                code, err = run_cli(["--limit", "2", "--small-url", small.url, "--large-url", "none",
                                     "--dataset", str(ROOT / "data" / "smoke.jsonl"),
                                     "--output", str(Path(tmp) / "e.jsonl"),
                                     "--arms", "small_constrained"])
        finally:
            small.close()
        self.assertEqual(code, 3)
        self.assertIn("does not conform", err)

    def test_server_down_gives_instructions_not_traceback(self):
        with TemporaryDirectory() as tmp:
            code, err = run_cli(["--limit", "1", "--small-url", "http://127.0.0.1:9", "--large-url", "none",
                                 "--dataset", str(ROOT / "data" / "smoke.jsonl"),
                                 "--output", str(Path(tmp) / "e.jsonl")])
        self.assertEqual(code, 2)
        self.assertIn("SEPARATE window", err)
        self.assertIn("cpu_platform.launch --model qwen3-0.6b --port 9", err)


class TestLauncherUX(unittest.TestCase):
    def test_banner_warns_about_ctrl_c(self):
        text = launch.BANNER.format(port=8080)
        self.assertIn("KEEP THIS WINDOW OPEN", text)
        self.assertIn("Ctrl+C here STOPS", text)

    def test_gpu_names_deduplicated(self):
        self.assertTrue(_same_gpu("Quadro P600", "NVIDIA Quadro P600"))
        self.assertFalse(_same_gpu("Quadro P600", "Intel(R) UHD Graphics 630"))


if __name__ == "__main__":
    unittest.main()
