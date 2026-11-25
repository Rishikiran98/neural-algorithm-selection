import tempfile
from pathlib import Path
import unittest

from cpu_platform.experiment2 import MODES, summarize, run_benchmark


class MockAPI:
    """Fake API responses used only to validate benchmark data flow."""
    def reason(self, question, mode, max_calls):
        if mode == 'baseline':
            correct = '9'   # deliberately incorrect for test input
            calls = 1
        else:
            correct = '4'
            calls = 1 if mode == 'hybrid' else 2
        return {
            'status': 'ok', 'answer': correct,
            'arithmetic_verified': mode != 'baseline',
            'numeric_grounding_passed': mode == 'adaptive',
            'semantically_verified': False,
            'stop_reason': 'mock_only',
            'usage': {'model_calls': calls, 'total_latency_ms': calls * 10,
                      'prompt_tokens': calls * 10, 'completion_tokens': calls * 3},
        }


class TestExperiment2(unittest.TestCase):
    def test_all_three_arms_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'benchmark.jsonl'
            q = [
                {'id': '1', 'question': '12 plus 8 divided among 5?', 'answer': '#### 4'},
                {'id': '2', 'question': '20 divide 5?', 'answer': '#### 4'},
            ]
            summary = run_benchmark(MockAPI(), q, path)
            self.assertTrue(path.exists())
            self.assertEqual(len(path.read_text().splitlines()), 6)
            self.assertEqual(set(summary['modes']), set(MODES))
            self.assertEqual(summary['modes']['adaptive']['accuracy'], 1)
            self.assertEqual(summary['modes']['baseline']['accuracy'], 0)
            self.assertEqual(summary['modes']['adaptive']['mean_model_calls_per_task'], 2)
            self.assertEqual(summary['paired_differences']['adaptive_minus_baseline']['accuracy_difference'], 1)

    def test_no_paired_missing_or_duplicates(self):
        row = {'id': '1', 'mode': 'adaptive', 'correct': True}
        with self.assertRaises(ValueError):
            summarize([row])
        with self.assertRaises(ValueError):
            summarize([row, row])

    def test_error_status_is_not_silently_dropped(self):
        class BadMock:
            def reason(self, *a, **kwargs):
                return {'http_status': 503, 'error': 'unavailable'}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_benchmark(BadMock(), [{'question':'12*3?', 'answer':'#### 36'}], Path(tmp)/'x.jsonl')


if __name__ == '__main__':
    unittest.main()
