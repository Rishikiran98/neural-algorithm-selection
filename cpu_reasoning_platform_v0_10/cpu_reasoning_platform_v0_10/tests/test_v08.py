"""v0.8 tests: native thinking with a budget, checker normalization and policies,
shared-sample pairing, preflights, real-dataset default. Offline; the fake server
is not a model and proves nothing about accuracy."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cpu_first.core import Completion, LocalLlama, split_thinking
from cpu_platform import experiment6
from cpu_platform.cot_check import POLICIES, audit_reasoning, normalize
from cpu_platform.search import Sample, check_cot_samples, sample

try:
    from fake_llama import FakeLlama
except ImportError:
    from tests.fake_llama import FakeLlama

ROOT = Path(__file__).resolve().parent.parent
SMOKE = str(ROOT / "data" / "smoke.jsonl")


class TestCheckerNormalization(unittest.TestCase):
    CASES = {  # text -> (status, checked_count)
        "First, 12 * 3 = 36. Then 50 - 36 = 14.": ("checked_pass", 2),
        "Cost is $12 × 3 = $36. Change is $50 - $36 = $14.": ("checked_pass", 2),
        "1,200 * 2 = 2,400": ("checked_pass", 1),
        "15 x 4 = 60, then 60 - 8 = 52.": ("checked_pass", 2),
        "Total: \\( 15 \\times 4 = 60 \\). Remaining: \\( 60 - 8 = 52 \\).": ("checked_pass", 2),
        "$15 \\times 4 = 60$ and $60 \\div 4 = 15$": ("checked_pass", 2),
        "\\frac{20}{5} = 4": ("checked_pass", 1),
        "25% of 80 = 20, so 80 - 20 = 60.": ("checked_pass", 2),
        "80 * 25% = 20": ("checked_pass", 1),
        "**15 * 4 = 60**": ("checked_pass", 1),
        "1. 12 * 3 = 36\n2. 50 - 36 = 14": ("checked_pass", 2),
        "$4.50 * 3 = $13.50": ("checked_pass", 1),
        "10 / 3 = 3.33 per person.": ("checked_pass", 1),
        "10 / 3 ≈ 3": ("checked_pass", 1),
        "10 / 3 \\approx 3.3": ("checked_pass", 1),
        # still errors
        "12 * 3 = 38": ("arithmetic_error", 1),
        "1,200 * 2 = 2,500": ("arithmetic_error", 1),
        "$12 × 3 = $35": ("arithmetic_error", 1),
        "10 / 3 = 3.34": ("arithmetic_error", 1),
        "10 / 3 = 3": ("arithmetic_error", 1),          # rounding to integer needs ≈
        "\\frac{20}{5} = 5": ("arithmetic_error", 1),
        # not checkable (and not counted as passes)
        "20 / 80 = 25%": ("no_checkable_equations", 0),
        "15 toys/hour * 4 hours = 60 toys": ("no_checkable_equations", 0),
        "2x = 10 so x = 5": ("no_checkable_equations", 0),
    }

    def test_cases(self):
        for text, (status, count) in self.CASES.items():
            with self.subTest(text=text):
                result = audit_reasoning(text)
                self.assertEqual((result.status, result.checked_count), (status, count), result)

    def test_rounding_is_marked(self):
        eq = audit_reasoning("10 / 3 = 3.33").equations[0]
        self.assertTrue(eq.correct and eq.rounded)
        self.assertFalse(audit_reasoning("12 * 3 = 36").equations[0].rounded)

    def test_normalize_is_text_only(self):
        hostile = '__import__("os").system("x") = 3 and $\\frac{a}{b}$'
        self.assertEqual(audit_reasoning(hostile).status, "no_checkable_equations")
        self.assertNotIn("$", normalize("$5 + $5 = $10"))


class TestPolicies(unittest.TestCase):
    def test_eligibility_rules(self):
        clean = audit_reasoning("12 * 3 = 36. 50 - 36 = 14.")
        wrong = audit_reasoning("12 * 3 = 38. 50 - 38 = 12.")
        silent = audit_reasoning("It is fourteen.")
        self.assertEqual({p: clean.eligible("14", p) for p in POLICIES},
                         {"strict": True, "exclude_errors": True, "answer_supported": True, "grounded": False})
        self.assertEqual({p: clean.eligible("36", p) for p in POLICIES},
                         {"strict": True, "exclude_errors": True, "answer_supported": True, "grounded": False})
        self.assertFalse(clean.eligible("15", "answer_supported"))
        self.assertEqual({p: wrong.eligible("12", p) for p in POLICIES},
                         {"strict": False, "exclude_errors": False, "answer_supported": False, "grounded": False})
        self.assertEqual({p: silent.eligible("14", p) for p in POLICIES},
                         {"strict": False, "exclude_errors": True, "answer_supported": False, "grounded": False})
        self.assertFalse(clean.eligible(None, "exclude_errors"))
        with self.assertRaises(ValueError):
            clean.eligible("14", "lenient")

    def test_policy_choice_changes_only_verified(self):
        s = [Sample(0, "14", False, None, 0, 10, 1.0, None, raw_content="It is 14.")]
        for policy, expected in (("strict", False), ("exclude_errors", True), ("answer_supported", False),
                                 ("grounded", False)):
            out = check_cot_samples(s, policy)[0]
            self.assertEqual(out.verified, expected, policy)
            self.assertEqual(set(out.verification_policies), set(POLICIES))


class TestThinkingClient(unittest.TestCase):
    def test_split_thinking(self):
        self.assertEqual(split_thinking('<think>a = 1</think>{"answer":"2"}'), ("a = 1", '{"answer":"2"}'))
        self.assertEqual(split_thinking("<think>cut off"), ("cut off", ""))
        self.assertEqual(split_thinking('{"answer":"2"}'), ("", '{"answer":"2"}'))

    def test_requests_state_thinking_explicitly(self):
        fake = FakeLlama("Qwen3-0.6B-Q8_0", accuracy=1.0)
        try:
            client = LocalLlama(fake.url)
            q = "A bus has 25 passengers. At the first stop, 7 leave and 9 board. How many passengers now?"
            on = client.chat("Solve.", q, max_tokens=400, thinking=True, reasoning_budget=128)
            body = fake.requests[-1]
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(body["reasoning_budget_tokens"], 128)
            self.assertFalse(body["messages"][0]["content"].endswith("/no_think"))
            self.assertIn("25 - 7 = 18", on.reasoning)
            client.chat("Solve.", q)                               # default for Qwen3: off
            body = fake.requests[-1]
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertTrue(body["messages"][0]["content"].endswith("/no_think"))
            self.assertNotIn("reasoning_budget_tokens", body)
            with self.assertRaises(ValueError):
                client.chat("Solve.", q, thinking=True, reasoning_budget=-5)
        finally:
            fake.close()

    def test_non_qwen_models_get_no_kwargs_by_default(self):
        fake = FakeLlama("llama-3.2-1b", accuracy=1.0)
        try:
            LocalLlama(fake.url).chat("Solve.", "q")
            self.assertNotIn("chat_template_kwargs", fake.requests[-1])
        finally:
            fake.close()


class ScriptedThinker:
    """Returns thinking with no JSON in the reply, as when the budget cuts it short."""
    def __init__(self):
        self.calls = []

    def chat(self, prompt, question, **options):
        self.calls.append(options)
        return Completion("", 0.5, 60, 300, reasoning='12 * 3 = 36. 50 - 36 = 14. So {"answer":"14"}')


class TestThinkSearch(unittest.TestCase):
    def test_think_samples_keep_reasoning_and_budget(self):
        model = ScriptedThinker()
        drawn = sample(model, "q", 2, kind="think", think_budget=300, max_tokens=64)
        self.assertEqual([c["thinking"] for c in model.calls], [True, True])
        self.assertEqual(model.calls[0]["reasoning_budget"], 300)
        self.assertGreaterEqual(model.calls[0]["max_tokens"], 300 + 256)
        self.assertEqual(drawn[0].answer, "14")                  # recovered from the thinking
        self.assertIn("50 - 36 = 14", drawn[0].raw_content)
        self.assertGreater(drawn[0].thinking_chars, 0)
        checked = check_cot_samples(drawn)
        self.assertTrue(all(s.verified for s in checked))

    def test_other_kinds_explicitly_disable_thinking(self):
        model = ScriptedThinker()
        sample(model, "q", 1, kind="cot")
        self.assertIs(model.calls[0]["thinking"], False)
        self.assertNotIn("reasoning_budget", model.calls[0])


def run_cli(args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        code = experiment6.main(args)
    return code, err.getvalue()


class TestExperimentV08(unittest.TestCase):
    def test_default_arms_end_to_end(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6)
        large = FakeLlama("Qwen3-8B", accuracy=0.85, delay_s=0.02)
        try:
            with TemporaryDirectory() as tmp:
                out = Path(tmp) / "e8.jsonl"
                code, err = run_cli(["--dataset", SMOKE, "--limit", "5", "--small-url", small.url,
                                     "--large-url", large.url, "--samples", "4", "--think-samples", "3",
                                     "--think-budget", "64", "--output", str(out)])
                self.assertEqual(code, 0, err)
                report = json.loads(out.with_suffix(".summary.json").read_text())
                rows = [json.loads(l) for l in out.read_text().splitlines()]
        finally:
            small.close()
            large.close()
        res = report["results"]
        self.assertEqual(set(res["curves"]), set(experiment6.ARMS))
        self.assertEqual(report["settings"]["check_policy"], "exclude_errors")
        for key in ("paired_cot_arithmetic_filter_effect", "paired_think_arithmetic_filter_effect",
                    "small_think_checked_vs_large_cot", "small_cot_checked_vs_large_cot"):
            self.assertIn(key, res["comparisons"])
        think = res["comparisons"]["paired_think_arithmetic_filter_effect"]
        self.assertEqual(set(think["policy_table"]), set(POLICIES))
        # The fake's wrong answers come with visible arithmetic slips, so filtering can only help.
        self.assertGreaterEqual(think["accuracy_after"], think["accuracy_before"])
        self.assertEqual(think["policy_table"]["exclude_errors"]["accuracy"], think["accuracy_after"])
        for r in rows:  # vote and checked arms re-score the same outputs
            for base in ("small_cot", "small_think"):
                self.assertEqual([s["raw_content"] for s in r["arms"][f"{base}_vote"]],
                                 [s["raw_content"] for s in r["arms"][f"{base}_checked"]])
            self.assertEqual(len(r["arms"]["small_think_vote"]), 3)
            self.assertTrue(all(s["thinking_chars"] > 0 for s in r["arms"]["small_think_vote"]))

    def test_thinking_requests_are_not_doubled(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6)
        try:
            with TemporaryDirectory() as tmp:
                code, err = run_cli(["--dataset", SMOKE, "--limit", "2", "--small-url", small.url,
                                     "--large-url", "none", "--think-samples", "3",
                                     "--arms", "small_think_vote,small_think_checked",
                                     "--output", str(Path(tmp) / "e.jsonl")])
            self.assertEqual(code, 0, err)
            thinking = [r for r in small.requests
                        if (r.get("chat_template_kwargs") or {}).get("enable_thinking")
                        and not r["messages"][1]["content"].startswith("PREFLIGHT")]
            self.assertEqual(len(thinking), 2 * 3)
        finally:
            small.close()

    def test_preflight_requires_per_request_thinking(self):
        small = FakeLlama("Qwen3-0.6B", accuracy=0.6, think_capable=False)
        try:
            with TemporaryDirectory() as tmp:
                code, err = run_cli(["--dataset", SMOKE, "--limit", "1", "--small-url", small.url,
                                     "--large-url", "none", "--arms", "small_think",
                                     "--output", str(Path(tmp) / "e.jsonl")])
        finally:
            small.close()
        self.assertEqual(code, 5)
        self.assertIn("v0.8 launcher", err)

    def test_missing_dataset_points_to_fetch(self):
        with TemporaryDirectory() as tmp:
            code, err = run_cli(["--dataset", str(Path(tmp) / "nope.jsonl"), "--large-url", "none",
                                 "--output", str(Path(tmp) / "e.jsonl")])
        self.assertEqual(code, 4)
        self.assertIn("fetch_gsm8k.py", err)


if __name__ == "__main__":
    unittest.main()
