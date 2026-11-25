"""Run: py -m examples.verify_semantic (zero neural/GPU calls)."""
from cpu_platform.semantic import verify_semantic_plan

QUESTION = 'Maya has 12 apples, buys 8 more, and puts them equally into 5 boxes.'
CORRECT = {'steps': [
    {'id': 's1', 'op': 'add', 'inputs': [{'literal': '12'}, {'literal': '8'}], 'claim': '20'},
    {'id': 's2', 'op': 'divide', 'inputs': [{'ref': 's1'}, {'literal': '5'}], 'claim': '4'},
], 'result': 's2'}
WRONG = {'steps': [
    {'id': 's1', 'op': 'add', 'inputs': [{'literal': '12'}, {'literal': '8'}], 'claim': '20'},
    {'id': 's2', 'op': 'add', 'inputs': [{'ref': 's1'}, {'literal': '5'}], 'claim': '25'},
], 'result': 's2'}


def main() -> None:
    good = verify_semantic_plan(QUESTION, CORRECT)
    bad = verify_semantic_plan(QUESTION, WRONG)
    print('Correct plan:', {'arithmetic_verified': good['arithmetic_verified'],
                            'restricted_semantics_verified': good['semantically_verified'],
                            'answer': good['answer']})
    print('Incorrect plan:', {'arithmetic_verified': bad['arithmetic_verified'],
                              'restricted_semantics_verified': bad['semantically_verified'],
                              'reason': bad['semantic_error']})
    assert good['semantically_verified'] and not bad['semantically_verified']
    assert bad['arithmetic_verified']


if __name__ == '__main__':
    main()
