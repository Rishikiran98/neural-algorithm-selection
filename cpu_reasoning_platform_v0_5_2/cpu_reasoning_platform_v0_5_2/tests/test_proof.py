"""Unit, adversarial and local API verification for v0.3 step graphs."""
import http.client
import json
from threading import Thread
import unittest

from cpu_first.core import Completion
from cpu_platform.engine import ReasoningEngine
from cpu_platform.proof import verify_plan
from cpu_platform.server import create_server
from cpu_platform.client import CPUReasoningClient

Q = "Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes."
GOOD = {
    "steps": [
        {"id": "s1", "op": "add", "inputs": [{"literal": "12"}, {"literal": "8"}], "claim": "20"},
        {"id": "s2", "op": "divide", "inputs": [{"ref": "s1"}, {"literal": "5"}], "claim": "4"},
    ],
    "result": "s2",
}


class FixedModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, question, max_tokens=256):
        self.prompts.append((prompt, question, max_tokens))
        output = self.outputs[self.calls]
        self.calls += 1
        return Completion(output, 0.01, 20, 12)

    def ready(self):
        return True


class TestProof(unittest.TestCase):
    def test_valid_chained_plan(self):
        r = verify_plan(Q, GOOD)
        self.assertTrue(r['valid'])
        self.assertEqual(r['answer'], '4')
        self.assertEqual(len(r['steps']), 2)
        self.assertTrue(all(s['claim_verified'] for s in r['steps']))
        self.assertFalse(r['semantically_verified'])

    def test_bad_claim_rejected(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['claim'] = '21'
        r = verify_plan(Q, plan)
        self.assertFalse(r['valid'])
        self.assertIsNone(r['answer'])
        self.assertIn('claimed 21 but computed 20', r['error'])

    def test_unsupported_number(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['inputs'][1] = {'literal': '999'}
        self.assertIn('ungrounded', verify_plan(Q, plan)['error'])

    def test_forward_reference(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['inputs'][1] = {'ref': 's2'}
        self.assertIn('forward reference', verify_plan(Q, plan)['error'])

    def test_cycle_rejected(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['inputs'][0] = {'ref': 's1'}
        self.assertFalse(verify_plan(Q, plan)['valid'])

    def test_duplicate_ids(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][1]['id'] = 's1'
        self.assertIn('repeated', verify_plan(Q, plan)['error'])

    def test_unused_step(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'].insert(1, {"id": "extra", "op": "add", "inputs": [{"literal": "12"}, {"literal": "8"}], "claim": "20"})
        self.assertIn('unused step', verify_plan(Q, plan)['error'])

    def test_bad_last_result_id(self):
        plan = json.loads(json.dumps(GOOD))
        plan['result'] = 's1'
        self.assertIn('last step', verify_plan(Q, plan)['error'])

    def test_divide_by_zero(self):
        plan = {"steps": [{"id": "s1", "op": "divide", "inputs": [{"literal": "12"}, {"literal": "0"}], "claim": "0"}], "result": "s1"}
        self.assertIn('ungrounded', verify_plan(Q, plan)['error'])
        self.assertIn('division by zero', verify_plan(Q + ' There are 0 oranges.', plan)['error'])

    def test_expression_injection_disallowed(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['inputs'][0] = {'literal': '__import__("os")'}
        self.assertFalse(verify_plan(Q, plan)['valid'])
        plan['steps'][0]['inputs'][0] = {'ref': 's1', 'literal': '12'}
        self.assertFalse(verify_plan(Q, plan)['valid'])

    def test_limits(self):
        self.assertFalse(verify_plan(Q, {'steps': [], 'result': 'x'})['valid'])
        self.assertFalse(verify_plan(Q, {'steps': GOOD['steps'] * 7, 'result': 's2'})['valid'])
        self.assertFalse(verify_plan(Q, {**GOOD, 'extra': True})['valid'])
        self.assertFalse(verify_plan(Q, {'steps': GOOD['steps'], 'result': 12})['valid'])
        self.assertFalse(verify_plan(Q, GOOD['steps'])['valid'])
        self.assertFalse(verify_plan('', GOOD)['valid'])
        self.assertFalse(verify_plan(Q, None)['valid'])

    def test_fractional_answer_and_percentage(self):
        question = '80 dollars with a 25 percent discount'
        plan = {'steps': [
            {'id': 's1', 'op': 'divide', 'inputs': [{'literal': '25'}, {'literal': '100'}], 'claim': '1/4'},
            {'id': 's2', 'op': 'subtract', 'inputs': [{'literal': '1'}, {'ref': 's1'}], 'claim': '3/4'},
            {'id': 's3', 'op': 'multiply', 'inputs': [{'literal': '80'}, {'ref': 's2'}], 'claim': '60'},
        ], 'result': 's3'}
        self.assertEqual(verify_plan(question, plan)['answer'], '60')

    def test_semantically_wrong_but_arithmetically_valid(self):
        # This is a known limitation of formal arithmetic proof checking.
        bad = {'steps': [{'id': 's1', 'op': 'add',
                          'inputs': [{'literal': '12'}, {'literal': '8'}], 'claim': '20'},
                         {'id': 's2', 'op': 'add',
                          'inputs': [{'ref': 's1'}, {'literal': '5'}], 'claim': '25'}], 'result': 's2'}
        result = verify_plan(Q, bad)
        self.assertTrue(result['valid'])
        self.assertEqual(result['answer'], '25')
        self.assertFalse(result['semantically_verified'])

    def test_exact_intermediate_fraction(self):
        q = '3 liters and 2 jars'
        plan = {'steps': [{'id': 's1', 'op': 'divide', 'inputs': [{'literal': '3'}, {'literal': '2'}], 'claim': '3/2'}], 'result': 's1'}
        self.assertEqual(verify_plan(q, plan)['answer'], '3/2')

    def test_no_float_claims_or_booleans(self):
        plan = json.loads(json.dumps(GOOD))
        plan['steps'][0]['claim'] = 20.0
        self.assertFalse(verify_plan(Q, plan)['valid'])
        plan['steps'][0]['claim'] = True
        self.assertFalse(verify_plan(Q, plan)['valid'])


class TestProofModel(unittest.TestCase):
    def test_one_call_success(self):
        model = FixedModel([json.dumps(GOOD)])
        result = ReasoningEngine(model).reason(Q, 'proof')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['usage']['model_calls'], 1)
        self.assertTrue(result['proof_verified'])
        self.assertEqual(len(result['proof_trace']), 2)
        self.assertFalse(result['semantically_verified'])

    def test_second_call_repairs_bad_claim(self):
        broken = json.loads(json.dumps(GOOD))
        broken['steps'][0]['claim'] = '21'
        model = FixedModel([json.dumps(broken), json.dumps(GOOD)])
        result = ReasoningEngine(model).reason(Q, 'proof')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['usage']['model_calls'], 2)
        self.assertEqual(result['attempts'][1]['route'], 'proof_repair')
        self.assertIn('rejected previous plan', model.prompts[1][1])

    def test_budget_exhaustion(self):
        model = FixedModel(['not json'])
        result = ReasoningEngine(model).reason(Q, 'proof', max_calls=1)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['answer'])
        self.assertEqual(result['stop_reason'], 'proof_budget_exhausted')
        self.assertEqual(result['usage']['model_calls'], 1)

    def test_no_numbers_does_not_call_model(self):
        model = FixedModel([])
        result = ReasoningEngine(model).reason('Find the value', 'proof')
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(model.calls, 0)


class TestProofAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FixedModel([json.dumps(GOOD)] * 5)
        cls.server = create_server(0, engine=ReasoningEngine(cls.model), api_key='key')
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def api(self, path, payload, auth=True):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=3)
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer key'
        connection.request('POST', path, json.dumps(payload), headers)
        response = connection.getresponse()
        code = response.status
        data = json.loads(response.read())
        connection.close()
        return code, data

    def test_verify_endpoint_without_model(self):
        before = self.model.calls
        status, result = self.api('/v1/verify-plan', {'question': Q, 'plan': GOOD})
        self.assertEqual(status, 200)
        self.assertEqual(result['answer'], '4')
        self.assertTrue(result['proof_verified'])
        self.assertEqual(result['usage']['model_calls'], 0)
        self.assertEqual(self.model.calls, before)

    def test_verify_endpoint_rejects_wrong_claim(self):
        broken = json.loads(json.dumps(GOOD))
        broken['steps'][1]['claim'] = '7'
        status, result = self.api('/v1/verify-plan', {'question': Q, 'plan': broken})
        self.assertEqual(status, 422)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['answer'])

    def test_verify_endpoint_auth_and_fields(self):
        self.assertEqual(self.api('/v1/verify-plan', {'question': Q, 'plan': GOOD}, auth=False)[0], 401)
        self.assertEqual(self.api('/v1/verify-plan', {'question': Q, 'plan': GOOD, 'extra': 1})[0], 400)

    def test_model_proof_endpoint(self):
        status, result = self.api('/v1/reason', {'question': Q, 'mode': 'proof', 'max_calls': 2})
        self.assertEqual(status, 200)
        self.assertEqual(result['answer'], '4')
        self.assertTrue(result['proof_verified'])

    def test_sdk_proof(self):
        client = CPUReasoningClient(f'http://127.0.0.1:{self.server.server_port}', api_key='key')
        result = client.verify_plan(Q, GOOD)
        self.assertEqual(result['answer'], '4')


if __name__ == '__main__':
    unittest.main()
