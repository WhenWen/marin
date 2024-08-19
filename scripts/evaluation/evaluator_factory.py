from scripts.evaluation.evaluator import Evaluator
from scripts.evaluation.simple_evaluator import SimpleEvaluator
from scripts.evaluation.eleuther_tpu_evaluator import EleutherTpuEvaluator

# Supported evaluators
NAME_TO_EVALUATOR = {
    "simple": SimpleEvaluator,
    "eleuther": EleutherTpuEvaluator,
}


def get_evaluator(evaluator_name: str) -> Evaluator:
    """
    Returns the evaluator for the given name.
    """
    assert evaluator_name in NAME_TO_EVALUATOR, f"Unknown evaluator: {evaluator_name}"
    return NAME_TO_EVALUATOR[evaluator_name]()
