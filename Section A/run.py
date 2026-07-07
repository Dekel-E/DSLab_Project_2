"""
Fixed entry point — do NOT modify this file.

evaluation.py calls main(seed) defined here.
Implement your logic in strategy.py.
"""

from strategy import run_active_learning


def main(seed: int):
    """Run active learning for the given seed and return a trained RandomForestClassifier."""
    return run_active_learning(seed)
