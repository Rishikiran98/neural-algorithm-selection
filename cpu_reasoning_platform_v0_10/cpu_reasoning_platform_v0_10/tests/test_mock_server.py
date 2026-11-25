"""Offline end-to-end validation. This is a fake model, NOT an AI benchmark."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from cpu_first.core import LocalLlama, solve


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.reply({"data": [{"id": "mock-model"}]})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(n))
        system = payload["messages"][0]["content"]
        answer = '{"answer":"4"}' if system.startswith("Solve ") else '{"expression":"(12 + 8) / 5"}'
        self.reply({"choices": [{"message": {"content": answer}}],
                    "usage": {"prompt_tokens": 16, "completion_tokens": 8}})

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestMockServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_both_modes(self):
        client = LocalLlama(f"http://127.0.0.1:{self.server.server_port}")
        for mode in ("baseline", "hybrid"):
            result = solve(client, "Maya has 12 apples and gets 8 more. She fills 5 boxes.", mode)
            self.assertEqual(result["prediction"], "4")
            self.assertEqual(result["completion_tokens"], 8)
            self.assertTrue(result["output_valid"])


if __name__ == "__main__":
    unittest.main()
