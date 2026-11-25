"""v0.7: paired free-form CoT voting and conservative arithmetic checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cpu_first.core import Completion
from cpu_platform.cot_check import audit_reasoning
from cpu_platform.experiment6 import run_experiment, summarize, prefix_votes_from
from cpu_platform.search import Sample, check_cot_samples


class CoTArithmeticChecks(unittest.TestCase):
    def test_good_multistep(self):
        v = audit_reasoning('12 * 3 = 36\n50 - 36 = 14\n{"answer":"14"}')
        self.assertEqual(v.status, 'checked_pass')
        self.assertEqual((v.checked_count, v.error_count), (2, 0))

    def test_detects_bad_step_even_if_later_good(self):
        v = audit_reasoning('12 * 3 = 35; 50 - 35 = 15 {"answer":"15"}')
        self.assertEqual(v.status, 'arithmetic_error')
        self.assertEqual((v.checked_count, v.error_count), (2, 1))

    def test_handles_parentheses_unicode_negative_decimals(self):
        for text in ('50 − (12 × 3) = 14', '7 ÷ 4 = 1.75', '3 - 8 = -5', '2.5*4 = 10'):
            with self.subTest(text=text):
                result = audit_reasoning(text)
                self.assertEqual(result.status, 'checked_pass', result)

    def test_correct_equation_can_still_support_wrong_answer(self):
        # The checker MUST NOT be described as a semantic or entailment verifier.
        result = audit_reasoning('2+2=4. The answer is 99. {"answer":"99"}')
        self.assertEqual(result.status, 'checked_pass')
        self.assertEqual(result.error_count, 0)

    def test_no_equation_not_verified(self):
        self.assertEqual(audit_reasoning('I computed the result. {"answer":"4"}').status,
                         'no_checkable_equations')
        self.assertEqual(audit_reasoning('3 squared is 9').status, 'no_checkable_equations')

    def test_wrong_division(self):
        self.assertEqual(audit_reasoning('38/3 = 12').status, 'arithmetic_error')

    def test_percentages_checked_since_v08(self):
        # v0.8 normalizes "10% of 240" to (10/100*240); still nothing executed.
        self.assertEqual(audit_reasoning('10% of 240 = 24').status, 'checked_pass')
        self.assertEqual(audit_reasoning('10% of 240 = 25').status, 'arithmetic_error')

    def test_text_is_not_executed(self):
        result = audit_reasoning('__import__("os").system("echo BAD") = 3')
        self.assertEqual(result.status, 'no_checkable_equations')

    def test_raises_no_large_memory_load_for_long_text(self):
        result = audit_reasoning('narrative ' * 3000)
        self.assertTrue(result.truncated)
        self.assertEqual(result.checked_count, 0)

    def test_sample_filter_policy(self):
        base = [
            Sample(0,'14',False,None,0,100,1.0,2.0,raw_content='12*3=36; 50-36=14 {"answer":"14"}'),
            Sample(1,'14',False,None,0,100,1.0,2.0,raw_content='12*3=35 {"answer":"14"}'),
            Sample(2,'14',False,None,0,100,1.0,2.0,raw_content='Looks like 14. {"answer":"14"}'),
        ]
        reviewed = check_cot_samples(base, policy='strict')   # the v0.7 rule
        self.assertEqual([x.verified for x in reviewed], [True, False, False])
        default = check_cot_samples(base)                      # v0.8: exclude only on evidence
        self.assertEqual([x.verified for x in default], [True, False, True])
        self.assertEqual(default[2].verification_policies,
                         {'strict': False, 'exclude_errors': True, 'answer_supported': False})
        self.assertEqual([x.verification_status for x in reviewed],
                         ['checked_pass', 'arithmetic_error', 'no_checkable_equations'])
        self.assertEqual([x.verification_equations for x in reviewed], [2,1,0])
        self.assertEqual([x.raw_content for x in base], [x.raw_content for x in reviewed])
        self.assertEqual([x.verified for x in base], [False, False, False])
        self.assertGreaterEqual(reviewed[0].cpu_seconds, base[0].cpu_seconds)


class ScriptedModel:
    def __init__(self):
        self.calls = []
        self.content = [
            '12*3=35. 50-35=15. {"answer":"15"}',
            '12*3=36. 50-36=14. {"answer":"14"}',
            '12*3=36. 50-36=14. {"answer":"14"}',
        ]

    def chat(self, prompt, question, **options):
        self.calls.append((prompt, question, options))
        return Completion(self.content[(len(self.calls)-1) % 3], 0.1, 80, 40)


class PairedExperiment(unittest.TestCase):
    def test_cot_arms_share_same_generation_not_two_draws(self):
        model = ScriptedModel()
        questions = [{'id':'test', 'question':'A store charges $12 per notebook. '
                      'You buy 3 notebooks and pay with $50. How much change?',
                      'answer':'work #### 14'}]
        with TemporaryDirectory() as d:
            records = run_experiment({'small':model}, questions, Path(d)/'experiment7.jsonl',
                                     arms=('small_cot_vote','small_cot_checked'),
                                     samples=3, warmup=False)
            self.assertEqual(len(model.calls), 3, 'paired comparisons must not double model calls')
            x = records[0]['arms']
            self.assertEqual([s['raw_content'] for s in x['small_cot_vote']],
                             [s['raw_content'] for s in x['small_cot_checked']])
            self.assertEqual([s['answer'] for s in x['small_cot_vote']],
                             [s['answer'] for s in x['small_cot_checked']])
            self.assertEqual([s['verified'] for s in x['small_cot_checked']],
                             [False, True, True])
            self.assertEqual(prefix_votes_from(records[0], 'small_cot_vote')[-1], '14')
            self.assertEqual(prefix_votes_from(records[0], 'small_cot_checked')[-1], '14')
            result = summarize(records, ('small_cot_vote','small_cot_checked'),
                               cost_unit='model_wall_seconds')
            self.assertTrue(result['comparisons']['paired_cot_arithmetic_filter_effect']['same_model_outputs'])
            self.assertEqual(result['comparisons']['paired_cot_arithmetic_filter_effect']['sample_checkable_rate'], 1)
            self.assertEqual(len(json.loads((Path(d)/'experiment7.jsonl').read_text())['arms']), 2)

    def test_only_checked_arm_needs_no_extra_control_calls(self):
        model = ScriptedModel()
        questions = [{'question':'Calculate 12*3.', 'answer':'foo #### 36'}]
        with TemporaryDirectory() as d:
            record = run_experiment({'small': model}, questions, Path(d)/'x.jsonl',
                                    arms=('small_cot_checked',), samples=2, warmup=False)
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(len(record[0]['arms']['small_cot_checked']), 2)


if __name__ == '__main__':
    unittest.main()
