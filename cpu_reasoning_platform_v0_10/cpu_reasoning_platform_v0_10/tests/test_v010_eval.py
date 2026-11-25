import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cpu_platform.general_eval import score, run

class TestGeneralEval(unittest.TestCase):
    def test_exact_numeric(self):
        r = {"status": "ok", "answer": "Final answer: 16"}
        row = {"grading": "numeric_exact", "expected": "16"}
        self.assertEqual(score(row, r)[0], "correct")
        self.assertEqual(score({**row, "expected": "17"}, r)[0], "incorrect")
        self.assertEqual(score(row, {"status": "ok", "answer": "sixteen"})[0], "unscored")

    def test_manual_is_not_auto_scored(self):
        self.assertEqual(score({"grading": "manual"}, {"status": "ok", "answer": "perfect"})[0], "unscored")
        self.assertEqual(score({"grading": "manual"}, {"status": "incomplete"})[0], "error")

    def test_run_records_unscored_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('''{"id":"x","question":"what is 2+2?","domain":"math","grading":"numeric_exact","expected":"4"}\n{"id":"y","question":"explain caching","domain":"general","grading":"manual"}\n''')
            class FakeSDK:
                def __init__(self,*args,**kwargs): pass
                def ask(self, question, **kwargs):
                    return {"status":"ok", "answer":"4", "usage":{"model_calls":1}}
            with patch('cpu_platform.general_eval.CPUReasoningClient', FakeSDK):
                s=run(str(path),str(Path(tmp)/"out.jsonl"))
            self.assertEqual(s["grades"], {"correct":1, "unscored":1})
            self.assertEqual(s["total_model_calls"], 2)
            self.assertEqual(len((Path(tmp)/"out.jsonl").read_text().splitlines()), 2)

    def test_frozen_eval_blocked_before_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.jsonl"
            path.write_text('{"id":"a","question":"hi","domain":"general","grading":"manual"}\n')
            with patch('cpu_platform.general_eval.split_of', return_value=("eval", {})), \
                 patch('cpu_platform.general_eval.CPUReasoningClient') as sdk:
                with self.assertRaisesRegex(ValueError, "Frozen evaluation"):
                    run(str(path), str(Path(tmp)/"out.jsonl"))
                sdk.assert_not_called()

    def test_duplicate_ids_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.jsonl"
            path.write_text('''{"id":"a","question":"hi","domain":"general","grading":"manual"}\n{"id":"a","question":"x","domain":"general","grading":"manual"}\n''')
            with self.assertRaisesRegex(ValueError,"duplicate"):
                run(str(path),str(Path(tmp)/"out.jsonl"))

if __name__ == '__main__':
    unittest.main()
