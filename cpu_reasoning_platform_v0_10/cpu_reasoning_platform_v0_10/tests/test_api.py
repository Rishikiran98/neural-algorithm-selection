import http.client
import json
from threading import Thread
import unittest

from cpu_first.core import Completion
from cpu_platform.engine import ReasoningEngine
from cpu_platform.server import create_server, MAX_BODY_BYTES
from cpu_platform.client import CPUReasoningClient


class FakeModel:
    def __init__(self):
        self.calls = 0

    def chat(self, prompt, question, max_tokens=256):
        self.calls += 1
        answer = '{"answer":"4"}' if prompt.startswith('Solve ') else '{"expression":"(12+8)/5"}'
        return Completion(answer, .02, 10, 8)

    def ready(self):
        return True


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FakeModel()
        cls.server = create_server(0, engine=ReasoningEngine(cls.model), api_key='test-secret')
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, payload=None, *, auth=True, content_type='application/json'):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        headers = {'Content-Type': content_type}
        if auth:
            headers['Authorization'] = 'Bearer test-secret'
        body = json.dumps(payload) if isinstance(payload, (dict, list)) else payload
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        status, result = resp.status, json.loads(resp.read())
        conn.close()
        return status, result

    def test_playground(self):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        conn.request('GET', '/')
        result = conn.getresponse()
        body = result.read().decode()
        self.assertEqual(result.status, 200)
        self.assertIn('CPU Reasoning Playground', body)
        self.assertIn('content-security-policy', {k.lower() for k, v in result.getheaders()})
        conn.close()

    def test_health_and_ready(self):
        self.assertEqual(self.request('GET', '/healthz', auth=False)[0], 200)
        self.assertEqual(self.request('GET', '/readyz')[1]['model_ready'], True)

    def test_authentication(self):
        self.assertEqual(self.request('POST', '/v1/reason', {'question': '2+3'}, auth=False)[0], 401)
        self.assertEqual(self.request('GET', '/v1/metrics', auth=False)[0], 401)

    def test_math_endpoint(self):
        status, body = self.request('POST', '/v1/reason', {'question': 'calculate: 9*9'})
        self.assertEqual(status, 200)
        self.assertEqual(body['answer'], '81')
        self.assertEqual(body['usage']['model_calls'], 0)
        self.assertFalse(body['semantically_verified'])
        self.assertEqual(len(body['request_id']), 16)

    def test_model_route(self):
        status, body = self.request('POST', '/v1/reason', {'question': '12 apples 8 more divided among 5'})
        self.assertEqual(status, 200)
        self.assertEqual(body['answer'], '4')
        self.assertEqual(body['route'], 'adaptive')
        self.assertEqual(body['usage']['model_calls'], 1)

    def test_validate_payload(self):
        for payload in ({'question': 'hi', 'admin': True}, {'question': ''}, {'mode': 'auto'},
                        {'question': 'hello', 'max_calls': 99}, []):
            with self.subTest(payload=payload):
                self.assertEqual(self.request('POST', '/v1/reason', payload)[0], 400)
        self.assertEqual(self.request('POST','/v1/reason','{not-json}')[0], 400)
        self.assertEqual(self.request('POST','/v1/reason', '{}', content_type='text/plain')[0], 415)

    def test_size_rejection(self):
        status, _ = self.request('POST', '/v1/reason', {'question': 'x'*MAX_BODY_BYTES})
        self.assertEqual(status, 413)

    def test_metrics_no_prompt_content(self):
        # Self-contained: make a request, then allow for the counter being updated just
        # after the response is sent (it is recorded in the handler's finally block).
        self.request('POST', '/v1/reason', {'question': 'calculate: 2+2'})
        import time
        deadline = time.monotonic() + 5
        while True:
            _, body = self.request('GET', '/v1/metrics')
            if body.get('requests_total', 0) >= 1 or time.monotonic() > deadline:
                break
            time.sleep(0.02)
        self.assertGreaterEqual(body['requests_total'], 1)
        self.assertIn('model_calls_total', body)
        self.assertNotIn('apples', json.dumps(body))

    def test_not_found(self):
        self.assertEqual(self.request('GET', '/not-real')[0], 404)

    def test_sdk(self):
        sdk = CPUReasoningClient(f'http://127.0.0.1:{self.port}', 'test-secret')
        result = sdk.reason('calculate: 6*7')
        self.assertEqual(result['answer'], '42')
        self.assertTrue(result['arithmetic_verified'])


if __name__ == '__main__':
    unittest.main()
