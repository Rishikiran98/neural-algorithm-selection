"""Evaluation does not silently omit invalid examples and uses separate gold truth."""
import json
from pathlib import Path
import tempfile
import unittest

from cpu_platform.experiment3 import MODES, run_benchmark, summarize


class ExampleClient:
    def __init__(self):
        self.calls = []

    def reason(self, question, mode, max_calls):
        self.calls.append((question, mode, max_calls))
        answer = '4' if mode != 'baseline' else '25'
        return {
            'status': 'ok', 'answer': answer,
            'proof_verified': mode == 'proof',
            'proof_trace': [{'id': 's1', 'claim_verified': True}, {'id': 's2', 'claim_verified': True}]
                           if mode == 'proof' else [],
            'semantically_verified': False,
            'usage': {'model_calls': 2 if mode == 'proof' else 1,
                      'prompt_tokens': 12, 'completion_tokens': 16,
                      'total_latency_ms': 20 if mode == 'proof' else 10},
        }


class Experiment3Tests(unittest.TestCase):
    def test_paired_results(self):
        qs = [{'id': 'a', 'question': '12 8 5', 'answer': 'working #### 4'},
              {'id': 'b', 'question': '12 8 5 again', 'answer': 'working #### 4'}]
        client = ExampleClient()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'scores.jsonl'
            result = run_benchmark(client, qs, path)
            self.assertEqual(result['paired_tasks'], 2)
            self.assertEqual(result['modes']['proof']['accuracy'], 1)
            self.assertEqual(result['modes']['baseline']['accuracy'], 0)
            self.assertEqual(result['modes']['proof']['verified_step_total'], 4)
            self.assertEqual(result['modes']['proof']['model_calls_total'], 4)
            self.assertEqual(len(path.read_text().splitlines()), 8)
            self.assertEqual(set(mode for _, mode, _ in client.calls), set(MODES))

    def test_incomplete_benchmark_rejected(self):
        rows = [{'id': 'a', 'mode': 'baseline', 'correct': True}]
        with self.assertRaisesRegex(ValueError, 'Every task'):
            summarize(rows)

    def test_duplicate_mode_rejected(self):
        rows = [{'id': 'a', 'mode': 'baseline'}, {'id': 'a', 'mode': 'baseline'}]
        with self.assertRaisesRegex(ValueError, 'Duplicate mode'):
            summarize(rows)

    def test_failed_requests_not_counted_as_benchmark_success(self):
        class Unavailable:
            def reason(self, *args, **kwargs):
                return {'http_status': 503, 'error': 'local_model_unavailable'}
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'HTTP failure'):
                run_benchmark(Unavailable(),
                              [{'question': '2 3', 'answer': '#### 5'}],
                              Path(temp)/'fail.jsonl')


if __name__ == '__main__':
    unittest.main()
