"""Run the v0.3 exact proof checker with no AI model, internet or server.

From project root: py -m examples.verify_plan
"""
import json
from cpu_platform.proof import verify_plan

question = "Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes."
plan = {
    "steps": [
        {"id": "s1", "op": "add", "inputs": [{"literal": "12"}, {"literal": "8"}], "claim": "20"},
        {"id": "s2", "op": "divide", "inputs": [{"ref": "s1"}, {"literal": "5"}], "claim": "4"},
    ],
    "result": "s2",
}

if __name__ == "__main__":
    print(json.dumps(verify_plan(question, plan), indent=2))
