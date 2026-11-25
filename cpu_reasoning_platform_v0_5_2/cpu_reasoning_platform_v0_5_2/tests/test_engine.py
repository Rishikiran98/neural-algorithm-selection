import unittest
from cpu_platform.engine import ReasoningEngine, Settings
from cpu_first.core import Completion


class FakeModel:
    def __init__(self, hybrid='{"expression":"(12+8)/5"}', baseline='{"answer":"4"}'):
        self.hybrid = hybrid
        self.baseline = baseline
        self.calls = 0
        self.last_max_tokens = None

    def chat(self, prompt, question, max_tokens=384):
        self.calls += 1
        self.last_max_tokens = max_tokens
        answer = self.baseline if prompt.startswith('Solve ') else self.hybrid
        return Completion(answer, .01, 12, 8)

    def ready(self):
        return True


class SequenceModel(FakeModel):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def chat(self, prompt, question, max_tokens=384):
        result = self.outputs[self.calls]
        self.calls += 1
        self.last_max_tokens = max_tokens
        return Completion(result, .01, 12, 8)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.engine = ReasoningEngine(self.model)

    def test_exact_local_has_zero_model_calls(self):
        result = self.engine.reason('calculate: 12*(7+3)')
        self.assertEqual(result['answer'], '120')
        self.assertTrue(result['arithmetic_verified'])
        self.assertFalse(result['semantically_verified'])
        self.assertEqual(result['route'], 'exact_arithmetic')
        self.assertEqual(result['usage']['model_calls'], 0)
        self.assertEqual(self.model.calls, 0)

    def test_question_prefixed_math(self):
        self.assertEqual(self.engine.reason('What is (2 + 3) / 2?')['answer'], '5/2')

    def test_local_expression_cannot_execute_python(self):
        # No model is used to execute Python. This is routed to the model instead.
        self.assertIsNone(self.engine._explicit_math('__import__("os").system("echo hi")'))
        self.assertEqual(self.model.calls, 0)

    def test_word_problem_goes_hybrid(self):
        result = self.engine.reason('Maya has 12 apples and buys 8 then shares among 5')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['route'], 'adaptive')
        self.assertTrue(result['arithmetic_verified'])
        self.assertFalse(result['semantically_verified'])
        self.assertEqual(result['usage']['model_calls'], 1)
        self.assertEqual(self.model.last_max_tokens, 256)

    def test_text_goes_baseline(self):
        result = self.engine.reason('Say the number four')
        self.assertEqual(result['route'], 'baseline')
        self.assertFalse(result['arithmetic_verified'])

    def test_repair_only_on_invalid_output(self):
        model = SequenceModel(['oops', '{"expression":"(12+8)/5"}'])
        result = ReasoningEngine(model).reason('12 plus 8 divided by 5', 'auto')
        self.assertEqual(result['route'], 'adaptive')
        self.assertEqual(result['answer'], '4')
        self.assertEqual(result['usage']['model_calls'], 2)
        self.assertTrue(result['arithmetic_verified'])
        self.assertEqual(result['stop_reason'], 'accepted_expression')
        self.assertEqual(len(result['attempts']), 2)

    def test_budget_disables_fallback(self):
        model = FakeModel(hybrid='oops')
        result = ReasoningEngine(model).reason('12 plus 8 divided by 5', 'auto', 1)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['usage']['model_calls'], 1)

    def test_mode_hybrid_has_no_fallback(self):
        model = FakeModel(hybrid='oops')
        result = ReasoningEngine(model).reason('12 plus 8 divided by 5', 'hybrid')
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(model.calls, 1)

    def test_invalid_args(self):
        for question, mode, budget in [('', 'auto', 2), ('hi', 'hack', 2), ('hi', 'auto', 0),
                                       ('hi', 'auto', 3), ('hi', 'auto', True), ('x'*4001, 'auto', 2)]:
            with self.subTest(question=question[:30], mode=mode, budget=budget):
                with self.assertRaises(ValueError):
                    self.engine.reason(question, mode, budget)

    def test_local_expression_invalid_does_not_fallback(self):
        result = self.engine.reason('calculate: 2/0')
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['usage']['model_calls'], 0)
        self.assertEqual(self.model.calls, 0)

    def test_semantic_gap_remains_explicit(self):
        model = FakeModel(hybrid='{"expression":"12+8+5"}')
        result = ReasoningEngine(model).reason('12 apples plus 8 apples shared by 5')
        self.assertEqual(result['answer'], '25')
        self.assertTrue(result['arithmetic_verified'])
        self.assertFalse(result['semantically_verified'])
        self.assertIn('NOT verified', result['verification_scope'])


if __name__ == '__main__':
    unittest.main()
