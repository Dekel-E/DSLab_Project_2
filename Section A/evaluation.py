"""
Evaluation script — do NOT modify this file.

Runs run.main(seed) for seeds 1–3, saves and reloads each model,
reports mean F1(Left), and enforces the runtime limit from constants.yaml.
"""

from __future__ import annotations

import ast
import sys
import time
import warnings
from pathlib import Path

import joblib

from run import main
from utils import CONFIG, SEEDS, evaluate_model, reset_oracle, set_active_seed

_STUDENT_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _STUDENT_DIR / ".eval_cache"
_STRATEGY_PATH = _STUDENT_DIR / "strategy.py"


def _load_allowed_imports() -> set[str]:
    return set(CONFIG.get("allowed_imports", []))


def _module_root(module_name: str) -> str:
    return module_name.split(".", 1)[0]


def check_imports(path: Path = _STRATEGY_PATH) -> None:
    """Ensure strategy.py only uses allowed top-level imports."""
    allowed = _load_allowed_imports() | {"utils", "__future__"}
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

    max_runtime = CONFIG["max_runtime_sec"]
    _CACHE_DIR.mkdir(exist_ok=True)
    scores: dict[int, float] = {}

    for seed in SEEDS:
        reset_oracle()
        set_active_seed(seed)

        model_path = _CACHE_DIR / f"model_seed_{seed}.joblib"
        t0 = time.perf_counter()
        model = main(seed=seed)
        elapsed = time.perf_counter() - t0

        if elapsed > max_runtime:
            raise TimeoutError(
                f"Seed {seed} exceeded runtime limit: {elapsed:.2f}s > {max_runtime}s"
            )

        joblib.dump(model, model_path)
        loaded_model = joblib.load(model_path)
        scores[seed] = evaluate_model(loaded_model, seed=seed)

        print(
            f"seed={seed}  F1(Left)={scores[seed]:.4f}  "
            f"runtime={elapsed:.2f}s  model={model_path.name}"
        )

    mean_f1 = sum(scores.values()) / len(scores)
    print(f"\nMean F1(Left) over seeds {SEEDS}: {mean_f1:.4f}")
    return scores


if __name__ == "__main__":
    warnings.filterwarnings("default")
    try:
        run_evaluation()
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise
