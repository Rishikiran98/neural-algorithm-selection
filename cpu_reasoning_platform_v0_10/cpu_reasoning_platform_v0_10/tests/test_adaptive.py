import unittest
from cpu_first.core import Completion
from cpu_platform.engine import ReasoningEngine
from cpu_platform.grounding import check_numeric_grounding


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0
        self.prompts = []

    def chat(self, prompt, question, max_tokens=256):
        self.prompts.append((prompt, question))
        result = self.outputs[self.calls]
        self.calls += 1
        return Completion(result, 0.02, 19, 9)


class GroundingTests(unittest.TestCase):
    def test_seen_numbers_and_negative(self):
        self.assertTrue(check_numeric_grounding('There were 12; then 8 left and 5 came.', '12-8+5').passed)
        self.assertTrue(check_numeric_grounding('Temp is -2 degrees and rose 7', '-2+7').passed)
        self.assertTrue(check_numeric_grounding('1,250 at 0.5 rate', '1250*0.5').passed)

    def test_unseen_constants(self):
        check = check_numeric_grounding('There are 12 apples and 8 boxes', '12+8+999')
        self.assertFalse(check.passed)
        self.assertEqual(check.ungrounded, ('999',))

    def test_percentage_and_half_allow_helpers(self):
        self.assertTrue(check_numeric_grounding('80 with 25 percent off', '80*(1-25/100)').passed)
        self.assertTrue(check_numeric_grounding('20 apples, half the amount', '20/2').passed)
        self.assertFalse(check_numeric_grounding('20 apples cost 25', '20*(1-25/100)').passed)

    def test_not_proof_of_word_problem_meaning(self):
        self.assertTrue(check_numeric_grounding('12 apples, buy 8, split 5 ways', '12+8+5').passed)


class AdaptiveEngineTests(unittest.TestCase):
    def question(self):
        return 'Maya has 12 apples, buys 8 and shares into 5 boxes.'

    def test_success_early_stop(self):
        model = ScriptedModel(['{"expression":"(12+8)/5"}', '{"expression":"999"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['stop_reason'], 'accepted_expression')
        self.assertTrue(result['numeric_grounding_passed'])
        self.assertTrue(result['arithmetic_verified'])
        self.assertFalse(result['semantically_verified'])
        self.assertEqual(result['usage']['model_calls'], 1)
        self.assertEqual(model.calls, 1)

    def test_repair_invalid_json(self):
        model = ScriptedModel(['I cannot.', '{"expression":"(12+8)/5"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['usage']['model_calls'], 2)
        self.assertEqual(result['attempts'][0]['route'], 'hybrid')
        self.assertEqual(result['attempts'][1]['route'], 'repair')
        self.assertIn('failed validation', model.prompts[1][1])
        self.assertEqual(result['usage']['completion_tokens'], 18)

    def test_ungrounded_number_triggers_retry(self):
        model = ScriptedModel(['{"expression":"(12+8)/999"}', '{"expression":"(12+8)/5"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['usage']['model_calls'], 2)
        self.assertIn('999', result['attempts'][0]['error'])
        self.assertFalse(result['attempts'][0]['numeric_grounding_passed'])

    def test_ungrounded_exhaustion_not_accepted(self):
        model = ScriptedModel(['{"expression":"(12+8)/999"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive', 1)
        self.assertEqual(result['status'], 'invalid')
        self.assertIsNone(result['answer'])
        self.assertEqual(result['numeric_grounding_passed'], False)
        self.assertEqual(result['stop_reason'], 'call_budget_exhausted')
        self.assertEqual(model.calls, 1)

    def test_bad_expression_stays_invalid(self):
        model = ScriptedModel(['{"expression":"1/0"}', '{"expression":"__import__(\\"os\\")"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive')
        self.assertEqual(result['status'], 'invalid')
        self.assertFalse(result['arithmetic_verified'])
        self.assertEqual(result['usage']['model_calls'], 2)

    def test_no_repair_for_valid_but_semantically_wrong(self):
        model = ScriptedModel(['{"expression":"12+8+5"}'])
        result = ReasoningEngine(model).reason(self.question(), 'adaptive')
        self.assertEqual(result['answer'], '25')
        self.assertTrue(result['numeric_grounding_passed'])
        self.assertFalse(result['semantically_verified'])
        self.assertEqual(model.calls, 1)

    def test_adaptive_with_no_digits_rejects(self):
        model = ScriptedModel([])
        result = ReasoningEngine(model).reason('How many are left?', 'adaptive')
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['stop_reason'], 'unsupported_input')
        self.assertEqual(model.calls, 0)

    def test_auto_no_digits_uses_baseline(self):
        model = ScriptedModel(['{"answer":"3"}'])
        result = ReasoningEngine(model).reason('How many apples are left?', 'auto')
        self.assertEqual(result['route'], 'baseline')
        self.assertEqual(result['usage']['model_calls'], 1)


if __name__ == '__main__':
    unittest.main()
