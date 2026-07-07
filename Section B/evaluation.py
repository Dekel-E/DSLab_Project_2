"""
Evaluation script — do NOT modify this file.

Runs run.main(seed) for seeds 1–3 and reports per-seed test accuracy,
the HW2 penalty formula, and the Section B score out of 40 points.
"""

from __future__ import annotations

import ast
import sys
import warnings
from pathlib import Path

from run import main
from utils import (
    ACCURACY_THRESHOLD,
    CONFIG,
    SEEDS,
    SECTION_B_POINTS,
    compute_section_b_score,
)

_STUDENT_DIR = Path(__file__).resolve().parent
_GNN_PATH = _STUDENT_DIR / "gnn.py"


def _load_allowed_imports() -> set[str]:
    return set(CONFIG.get("allowed_imports", []))


def _module_root(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def check_imports(path: Path = _GNN_PATH) -> None:
    """Ensure gnn.py only uses allowed top-level imports."""
    allowed = _load_allowed_imports() | {"__future__"}
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _module_root(alias.name)
                if root not in allowed:
                    raise ImportError(
                        f"Disallowed import '{alias.name}' in {path.name}. "
                        f"Allowed top-level packages: {sorted(allowed - {'__future__'})}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = _module_root(node.module)
            if root not in allowed:
                raise ImportError(
                    f"Disallowed import from '{node.module}' in {path.name}. "
                    f"Allowed top-level packages: {sorted(allowed - {'__future__'})}"
                )


def run_evaluation() -> dict[int, float]:
    check_imports()

    test_scores: dict[int, float] = {}
    for seed in SEEDS:
        test_scores[seed] = main(seed=seed)
        print(f"seed={seed}  test_accuracy={test_scores[seed]:.3f}")

    score, penalty = compute_section_b_score(test_scores)
    mean_accuracy = sum(test_scores.values()) / len(test_scores)

    print(f"\nMean test accuracy over seeds {SEEDS}: {mean_accuracy:.3f}")
    print(f"Accuracy threshold: {ACCURACY_THRESHOLD:.2f}")
    print(f"Penalty points: {penalty}")
    print(f"Section B score (before Section A bonus): {score:.0f}/{SECTION_B_POINTS:.0f}")
    return test_scores


if __name__ == "__main__":
    warnings.filterwarnings("default")
    try:
        run_evaluation()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise
