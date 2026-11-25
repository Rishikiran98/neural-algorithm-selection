"""Multi-domain preview regression suite. Fake inference is NOT a model benchmark."""
import http.client
import json
from threading import Thread
import unittest

from cpu_first.core import Completion
from cpu_platform.engine import ReasoningEngine
from cpu_platform.general import GeneralReasoner, _domain_for, _python_syntax
from cpu_platform.logic import LogicInputError, check_entailment
from cpu_platform.server import create_server
from cpu_platform.client import CPUReasoningClient


class FakeGeneralModel:
    def __init__(self, replies=None):
        self.calls = 0
        self.replies = list(replies or ["Binary search halves the search interval."])
        self.params = []

    def chat(self, prompt, question, max_tokens=768, **kwargs):
        self.calls += 1
        self.params.append((prompt, question, max_tokens, kwargs))
        reply = self.replies[min(self.calls-1, len(self.replies)-1)]
        return Completion(reply, .01, 10, 20, None)

    def ready(self):
        return True


class TestGeneralUnit(unittest.TestCase):
    def test_domains(self):
        self.assertEqual(_domain_for("Implement a Python function for sorting"), "code")
        self.assertEqual(_domain_for("Does this argument entail its conclusion?"), "logic")
        self.assertEqual(_domain_for("How many meters in a kilometer?"), "math")
        self.assertEqual(_domain_for("Explain photosynthesis"), "science")
        self.assertEqual(_domain_for("When was the treaty signed?"), "factual")
        self.assertEqual(_domain_for("What are the tradeoffs of caching?"), "general")

    def test_direct_general_answer_not_verified(self):
        model = FakeGeneralModel()
        result = GeneralReasoner(model).ask("Why does binary search work?")
        self.assertEqual(result["status"], "ok")
        self.assertIn("Binary search", result["answer"])
        self.assertEqual(result["usage"]["model_calls"], 1)
        self.assertFalse(result["verified"])
        self.assertEqual(result["checks"]["arithmetic"]["checked_count"], 0)

    def test_exact_arithmetic_shortcut(self):
        model = FakeGeneralModel()
        result = GeneralReasoner(model).ask("What is (12+8)/5?")
        self.assertEqual(result["answer"], "4")
        self.assertEqual(result["usage"]["model_calls"], 0)
        self.assertTrue(result["verified"])
        self.assertEqual(model.calls, 0)

    def test_factual_unverified(self):
        result = GeneralReasoner(FakeGeneralModel()).ask("When was this discovered?")
        self.assertTrue(any("No external sources" in x for x in result["checks"]["issues"]))

    def test_math_error_detect_and_revise(self):
        model = FakeGeneralModel(["12 * 3 = 35. So answer is 35.",
                                  "12 * 3 = 36. So the answer is 36."])
        result = GeneralReasoner(model).ask("A shelf has 12 items in 3 groups. How many?",
                                            strategy="check_and_revise")
        self.assertEqual(result["usage"]["model_calls"], 2)
        self.assertIn("36", result["answer"])
        self.assertEqual(result["checks"]["arithmetic"]["error_count"], 0)
        self.assertFalse(result["verified"])
        self.assertEqual(result["checks"]["attempts"], 2)

    def test_no_repair_when_no_checkable_equations(self):
        model = FakeGeneralModel()
        result = GeneralReasoner(model).ask("Explain a tree", strategy="check_and_revise")
        self.assertEqual(result["usage"]["model_calls"], 1)
        self.assertEqual(result["checks"]["arithmetic"]["status"], "no_checkable_equations")

    def test_no_repair_with_single_call_budget(self):
        model = FakeGeneralModel(["2 * 3 = 7"])
        result = GeneralReasoner(model).ask("What is two times three?", strategy="check_and_revise", max_calls=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["checks"]["arithmetic"]["error_count"], 1)

    def test_empty_answer_is_incomplete(self):
        result = GeneralReasoner(FakeGeneralModel([""])).ask("Tell me about trees")
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["answer"])

    def test_native_thinking_flags(self):
        model = FakeGeneralModel()
        GeneralReasoner(model).ask("Explain loops", thinking=True, think_budget=64)
        kwargs = model.params[0][3]
        self.assertEqual(kwargs["thinking"], True)
        self.assertEqual(kwargs["reasoning_budget"], 64)

    def test_python_syntax_only(self):
        self.assertEqual(_python_syntax("```python\nprint(1)\n``` ")["status"], "syntax_valid")
        self.assertEqual(_python_syntax("```python\ndef foo(:\n  pass\n``` ")["status"], "invalid_syntax")
        self.assertEqual(_python_syntax("```javascript\nthrow new Error()\n``` ")["status"], "not_checked")
        self.assertEqual(_python_syntax("This sounds right")["status"], "not_checked")

    def test_reject_invalid_controls(self):
        engine = GeneralReasoner(FakeGeneralModel())
        for args in (
            {"question": ""}, {"question": "x"*4001}, {"question": "Hi", "domain": "fantasy"},
            {"question": "Hi", "strategy": "infinite"}, {"question": "Hi", "thinking": 1},
            {"question": "Hi", "max_calls": True}, {"question": "Hi", "max_calls": 3},
            {"question": "Hi", "think_budget": 513}, {"question": "Hi", "think_budget": "10"},
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                engine.ask(**args)


class TestPropositionalLogic(unittest.TestCase):
    def test_modus_ponens(self):
        r = check_entailment(["p -> q", "p"], "q")
        self.assertTrue(r["entailed"])
        self.assertTrue(r["premises_satisfiable"])
        self.assertEqual(r["model_calls"], 0)

    def test_counterexample(self):
        r = check_entailment(["p | q"], "p")
        self.assertFalse(r["entailed"])
        self.assertTrue(r["counterexample"]["q"])
        self.assertFalse(r["counterexample"]["p"])

    def test_tautology(self):
        r = check_entailment([], "p | !p")
        self.assertTrue(r["entailed"])

    def test_contradictory_premises_vacuous(self):
        r = check_entailment(["p", "!p"], "q")
        self.assertTrue(r["entailed"])
        self.assertTrue(r["vacuous"])
        self.assertFalse(r["premises_satisfiable"])

    def test_implies_right_associative(self):
        # p -> (q -> r) differs from (p -> q) -> r when p is false and r false.
        r = check_entailment(["p -> q -> r"], "p -> (q -> r)")
        self.assertTrue(r["entailed"])

    def test_constants(self):
        self.assertTrue(check_entailment(["TRUE"], "false -> p")["entailed"])
        self.assertFalse(check_entailment(["true"], "false")["entailed"])

    def test_bad_formulas_are_rejected(self):
        for s in ("", "p && q", "import os", "p; __import__('os')", "p => q", "p &", "(p", "p q"):
            with self.subTest(formula=s), self.assertRaises(LogicInputError):
                check_entailment([s], "q")

    def test_limit_vars(self):
        with self.assertRaises(LogicInputError):
            check_entailment([], " | ".join(f"p{i}" for i in range(13)))

    def test_limit_premises(self):
        with self.assertRaises(LogicInputError):
            check_entailment(["p"]*13, "p")

    def test_no_code_execution(self):
        with self.assertRaises(LogicInputError):
            check_entailment([], "__import__('os').system('whoami')")


class TestGeneralAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FakeGeneralModel()
        cls.server = create_server(0, engine=ReasoningEngine(cls.model), api_key="demo")
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, endpoint, body, key="demo"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        conn.request("POST", endpoint, json.dumps(body), headers)
        response = conn.getresponse()
        out = response.status, json.loads(response.read())
        conn.close()
        return out

    def test_general_answers(self):
        status, result = self.post("/v1/ask", {"question": "Explain caching"})
        self.assertEqual(status, 200)
        self.assertEqual(result["route"], "general_direct")
        self.assertFalse(result["verified"])
        self.assertEqual(result["version"], "0.10.0")

    def test_logic_endpoint_proved_and_counterexample(self):
        status, result = self.post("/v1/verify-logic", {"premises": ["p -> q", "p"], "conclusion": "q"})
        self.assertEqual(status, 200)
        self.assertEqual(result["answer"], "entailed")
        self.assertEqual(result["usage"]["model_calls"], 0)
        self.assertEqual(self.post("/v1/verify-logic", {"premises": ["p | q"], "conclusion": "p"})[1]["answer"], "not entailed")

    def test_api_auth_and_validation(self):
        self.assertEqual(self.post("/v1/ask", {"question": "hi"}, key="")[0], 401)
        self.assertEqual(self.post("/v1/ask", {"question": "hi", "evil": True})[0], 400)
        self.assertEqual(self.post("/v1/ask", {"question": "hi", "thinking": 1})[0], 400)
        self.assertEqual(self.post("/v1/verify-logic", {"premises": ["p"]})[0], 400)
        self.assertEqual(self.post("/v1/verify-logic", {"premises": ["p&"], "conclusion": "p"})[0], 400)

    def test_sdk(self):
        sdk = CPUReasoningClient(f"http://127.0.0.1:{self.server.server_port}", api_key="demo")
        self.assertEqual(sdk.ask("Explain binary search")["status"], "ok")
        self.assertTrue(sdk.verify_logic(["p -> q", "p"], "q")["checks"]["entailed"])


if __name__ == "__main__":
    unittest.main()
