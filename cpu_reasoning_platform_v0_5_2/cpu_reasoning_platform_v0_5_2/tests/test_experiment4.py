"""Synthetic fixture evaluation is software testing, never evidence of model quality."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from cpu_platform.experiment4 import manifest, summarize, run_benchmark, sha256_file

class MockClient:
    def reason(self, question, mode, max_calls):
        supported = 'unsupported' not in question
        return {'status': 'ok' if (mode != 'semantic' or supported) else 'invalid',
                'answer': '4' if mode != 'baseline' and supported else None,
                'stop_reason': 'unsupported_template' if mode == 'semantic' and not supported else 'accepted_checked_plan',
                'semantically_verified': mode == 'semantic' and supported,
                'usage': {'model_calls': 0 if mode == 'semantic' and not supported else max_calls,
                          'prompt_tokens': 9, 'completion_tokens': 5,
                          'total_latency_ms': 12 if mode == 'baseline' else 18}}

class Benchmark4Tests(unittest.TestCase):
    def test_deterministic_paired_outcome_and_coverage(self):
        questions = [
            {'id':'a','question':'a supported problem','answer':'#### 4'},
            {'id':'b','question':'an unsupported problem','answer':'#### 4'},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'run.jsonl'
            result = run_benchmark(MockClient(), questions, p, modes=('baseline','semantic'), seed=17, repeats=2)
            self.assertEqual(result['paired_observations'], 4)
            self.assertEqual(result['modes']['semantic']['correct'], 2)
            self.assertEqual(result['modes']['semantic']['abstentions_or_invalid'], 2)
            self.assertEqual(result['modes']['semantic']['semantic_coverage'], .5)
            self.assertEqual(len(p.read_text().splitlines()), 8)
            self.assertIsNotNone(result['paired_accuracy_differences']['semantic_minus_baseline'])

    def test_missing_pair_detected(self):
        row = dict(id='x', repeat=0, mode='baseline')
        with self.assertRaisesRegex(ValueError, 'Every paired task'):
            summarize([row], ('baseline','semantic'))

    def test_duplicate_pair_detected(self):
        row = dict(id='x', repeat=0, mode='baseline')
        with self.assertRaisesRegex(ValueError, 'Duplicate mode'):
            summarize([row,row], ('baseline',))

    def test_failed_backend_is_not_counted_as_abstention(self):
        class Broken:
            def reason(self, *args, **kwargs):
                return {'http_status':503,'error':'offline'}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'HTTP failure'):
                run_benchmark(Broken(), [{'question':'x','answer':'#### 4'}], Path(d)/'a.jsonl', modes=('baseline',))

    def test_manifest_records_fingerprints_and_cpu_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'data.jsonl'
            p.write_text('{"question":"two + two","answer":"#### 4"}\n', encoding='utf-8')
            m = manifest(p, model_id='Qwen3-test', quantization='Q8_0', threads=4,
                         model_file=None, modes=('baseline','semantic'), seed=17,
                         repeats=1, gpu_layers=0, context_size=2048)
            self.assertEqual(m['dataset']['sha256'], hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(m['model']['gpu_layers_declared'], 0)
            self.assertFalse(m['model']['cpu_only_execution_attested'])
            self.assertEqual(m['source']['sha256'].__len__(), 64)
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                manifest(p, model_id='', quantization='', threads=None,
                         model_file=None, modes=('baseline',), seed=0, repeats=1,
                         gpu_layers=1, context_size=None)

if __name__ == '__main__':
    unittest.main()
