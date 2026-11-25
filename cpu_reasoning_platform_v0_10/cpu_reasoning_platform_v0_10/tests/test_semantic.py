"""Restricted semantic contract with adversarial and end-to-end coverage."""
import http.client
import json
from threading import Thread
import unittest

from cpu_first.core import Completion
from cpu_platform.engine import ReasoningEngine
from cpu_platform.proof import verify_plan
from cpu_platform.semantic import classify, verify_semantic_plan
from cpu_platform.server import create_server
from cpu_platform.client import CPUReasoningClient

Q = "Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes."
GOOD = {
    "steps": [
        {"id":"s1", "op":"add", "inputs":[{"literal":"12"},{"literal":"8"}], "claim":"20"},
        {"id":"s2", "op":"divide", "inputs":[{"ref":"s1"},{"literal":"5"}], "claim":"4"},
    ], "result":"s2"
}
WRONG = {
    "steps": [
        {"id":"s1", "op":"add", "inputs":[{"literal":"12"},{"literal":"8"}], "claim":"20"},
        {"id":"s2", "op":"add", "inputs":[{"ref":"s1"},{"literal":"5"}], "claim":"25"},
    ], "result":"s2"
}


class MockModel:
    def __init__(self, replies):
        self.replies = replies
        self.count = 0

    def chat(self, *args, **kwargs):
        response = self.replies[self.count]
        self.count += 1
        return Completion(response, 0.01, 11, 14)

    def ready(self):
        return True


class RestrictedSemanticTests(unittest.TestCase):
    def test_arithmetic_only_mode_stays_unchanged(self):
        self.assertTrue(verify_plan(Q, WRONG)["valid"])
        self.assertFalse(verify_plan(Q, WRONG)["semantically_verified"])
        self.assertTrue(verify_plan(Q, GOOD)["valid"])
        self.assertFalse(verify_plan(Q, GOOD)["semantically_verified"])

    def test_restricted_template_verifies_right_plan(self):
        checked = verify_semantic_plan(Q, GOOD)
        self.assertTrue(checked["valid"])
        self.assertTrue(checked["semantic_supported"])
        self.assertTrue(checked["semantically_verified"])
        self.assertEqual(checked["semantic_template"], "combine_then_distribute")
        self.assertEqual(checked["answer"], "4")

    def test_wrong_operation_rejected_despite_valid_arithmetic(self):
        checked = verify_semantic_plan(Q, WRONG)
        self.assertTrue(checked["valid"])
        self.assertFalse(checked["semantically_verified"])
        self.assertIn("operation", checked["semantic_error"])

    def test_wrong_roles_rejected_even_same_arithmetic(self):
        altered = json.loads(json.dumps(GOOD))
        altered["steps"][1]["inputs"] = [{"literal":"5"},{"ref":"s1"}]
        altered["steps"][1]["claim"] = "1/4"
        self.assertTrue(verify_plan(Q, altered)["valid"])
        self.assertFalse(verify_semantic_plan(Q, altered)["semantically_verified"])

    def test_commutative_operands_reordered_accepted(self):
        altered = json.loads(json.dumps(GOOD))
        altered["steps"][0]["inputs"].reverse()
        self.assertTrue(verify_semantic_plan(Q, altered)["semantically_verified"])

    def test_extra_clause_no_semantic_claim(self):
        q = Q + " But she loses two apples first."
        checked = verify_semantic_plan(q, GOOD)
        self.assertTrue(checked["valid"])
        self.assertFalse(checked["semantically_verified"])
        self.assertFalse(checked["semantic_supported"])

    def test_bad_numeric_claim_rejected(self):
        broken = json.loads(json.dumps(GOOD))
        broken["steps"][0]["claim"] = "23"
        checked = verify_semantic_plan(Q, broken)
        self.assertFalse(checked["valid"])
        self.assertFalse(checked["semantically_verified"])

    def test_unrecognized_wording_abstains(self):
        self.assertIsNone(classify("Maya has 12 apples and later receives some extras."))
        self.assertIsNone(classify("Maya has 12 apples, buys 8 more, and puts them unequally into 5 boxes."))

    def test_zero_division_template_is_unsupported(self):
        self.assertIsNone(classify("Maya has 12 apples, buys 8 more, and puts them equally into 0 boxes."))

    def test_template_examples_all_parameterized(self):
        cases = [
            ("A bus has 25 passengers. At the first stop, 7 leave and 9 board. How many passengers now?",
             ["subtract", "add"], "27"),
            ("A store charges $12 per notebook. You buy 3 notebooks and pay with $50. How much change?",
             ["multiply", "subtract"], "14"),
            ("A factory produces 15 toys per hour for 4 hours. It discards 8 defective toys. How many remain?",
             ["multiply", "subtract"], "52"),
            ("A bus has 42 passengers. At the first stop, 13 leave and 11 board. How many passengers now?",
             ["subtract", "add"], "40"),
        ]
        for question, ops, answer in cases:
            with self.subTest(question=question):
                spec = classify(question)
                self.assertIsNotNone(spec)
                self.assertEqual([step[0] for step in spec.expected_steps], ops)
                self.assertEqual(str(spec.answer), answer)

    def test_semantic_engine_repairs_semantically_wrong_math(self):
        model = MockModel([json.dumps(WRONG), json.dumps(GOOD)])
        result = ReasoningEngine(model).reason(Q, mode="semantic", max_calls=2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], "4")
        self.assertTrue(result["semantically_verified"])
        self.assertEqual(result["usage"]["model_calls"], 2)
        self.assertFalse(result["attempts"][0]["valid"])
        self.assertTrue(result["attempts"][1]["valid"])

    def test_semantic_engine_abstains_on_unsupported_without_model_call(self):
        model = MockModel([])
        result = ReasoningEngine(model).reason("There are 4 bags and 7 cherries", mode="semantic")
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["stop_reason"], "unsupported_template")
        self.assertEqual(result["usage"]["model_calls"], 0)

    def test_semantic_engine_exhausts_budget_on_wrong_graph(self):
        result = ReasoningEngine(MockModel([json.dumps(WRONG)])).reason(Q, "semantic", max_calls=1)
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["usage"]["model_calls"], 1)


class SemanticAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0, engine=ReasoningEngine(MockModel([json.dumps(GOOD)] * 3)), api_key="testkey")
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(2)
        cls.server.server_close()

    def req(self, path, payload):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        connection.request('POST', path, json.dumps(payload),
                           {"Content-Type": "application/json", "Authorization": "Bearer testkey"})
        response = connection.getresponse()
        code, body = response.status, json.loads(response.read())
        connection.close()
        return code, body

    def test_restricted_endpoint_accepts_right_graph(self):
        status, data = self.req('/v1/verify-semantic-plan', {'question':Q, 'plan':GOOD})
        self.assertEqual(status, 200)
        self.assertEqual(data['usage']['model_calls'], 0)
        self.assertTrue(data['semantically_verified'])

    def test_restricted_endpoint_rejects_wrong_graph(self):
        status, data = self.req('/v1/verify-semantic-plan', {'question':Q, 'plan':WRONG})
        self.assertEqual(status, 422)
        self.assertIsNone(data['answer'])
        self.assertTrue(data['arithmetic_verified'])
        self.assertFalse(data['semantically_verified'])

    def test_unrestricted_endpoint_retains_behavior(self):
        status, data = self.req('/v1/verify-plan', {'question':Q, 'plan':WRONG})
        self.assertEqual(status, 200)
        self.assertEqual(data['answer'], '25')
        self.assertFalse(data['semantically_verified'])

    def test_model_semantic_endpoint(self):
        status, data = self.req('/v1/reason', {'question':Q, 'mode':'semantic', 'max_calls':1})
        self.assertEqual(status, 200)
        self.assertTrue(data['semantically_verified'])
        self.assertEqual(data['usage']['model_calls'], 1)

    def test_sdk_flag(self):
        client = CPUReasoningClient(f'http://127.0.0.1:{self.server.server_port}', api_key='testkey')
        self.assertTrue(client.verify_plan(Q, GOOD, semantic=True)['semantically_verified'])

if __name__ == '__main__':
    unittest.main()
