"""A tiny stand-in for llama.cpp's HTTP server, for offline end-to-end tests.

It answers the bundled smoke questions with a configurable, seed-deterministic
accuracy. It is NOT a model and its numbers are NOT evidence of anything.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import random
from threading import Thread
import time

# Correct answers and correct step plans for data/smoke.jsonl.
GOLD = {
    "Maya has 12 apples": ("4", [("add", "12", "8", "20"), ("divide", "s1", "5", "4")]),
    "A bus has 25 passengers": ("27", [("subtract", "25", "7", "18"), ("add", "s1", "9", "27")]),
    "A store charges $12": ("14", [("multiply", "12", "3", "36"), ("subtract", "50", "s1", "14")]),
    "A factory produces 15": ("52", [("multiply", "15", "4", "60"), ("subtract", "s1", "8", "52")]),
    "A shop discounts a $80": ("60", [("divide", "25", "100", "1/4"), ("subtract", "1", "s1", "3/4"),
                                      ("multiply", "80", "s2", "60")]),
}


def _plan(steps, corrupt=False):
    out = []
    for i, (op, a, b, claim) in enumerate(steps):
        operand = lambda v: {"ref": v} if v.startswith("s") else {"literal": v}
        if corrupt and i == len(steps) - 1:
            claim = claim + "1"  # arithmetically wrong final claim -> verifier rejects
        out.append({"id": f"s{i + 1}", "op": op, "inputs": [operand(a), operand(b)], "claim": claim})
    return {"steps": out, "result": out[-1]["id"]}


class FakeLlama:
    def __init__(self, model_id: str, accuracy: float, delay_s: float = 0.0, think: bool = False):
        self.model_id, self.accuracy, self.delay_s, self.think = model_id, accuracy, delay_s, think
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, code, obj):
                blob = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json(200, {"data": [{"id": fake.model_id}]})
                if self.path == "/health":
                    return self._json(200, {"status": "ok"})
                self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                time.sleep(fake.delay_s)
                self._json(200, fake.complete(body))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        Thread(target=self.httpd.serve_forever, daemon=True).start()

    def complete(self, body: dict) -> dict:
        system, user = body["messages"][0]["content"], body["messages"][1]["content"]
        key = next((k for k in GOLD if user.startswith(k)), None)
        temp = body.get("temperature", 0)
        rng = random.Random(f"{self.model_id}|{user}|{body.get('seed')}|{temp}")
        good = rng.random() < self.accuracy
        if key is None:
            content = '{"answer":"1"}'
        elif "restricted arithmetic-plan generator" in system:
            content = json.dumps(_plan(GOLD[key][1], corrupt=not good))
        else:
            answer = GOLD[key][0] if good else str(int(GOLD[key][0]) + 1)
            content = json.dumps({"answer": answer})
        if self.think:
            content = "<think>Let me {think} about 99 apples.</think>" + content
        return {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": len(content) // 4}}

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
