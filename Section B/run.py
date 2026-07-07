"""
Fixed entry point — do NOT modify this file.

evaluation.py calls main(seed) defined here.
Implement your logic in gnn.py.
"""

from __future__ import annotations

import torch

from gnn import GraphSAGE, train
from utils import (
    BEST_MODEL_PATH,
    EPOCHS,
    OPTIMIZER_CONFIG,
    build_graph_data,
    cleanup_artifacts,
    evaluate,
    get_device,
    seed_all,
)


def main(seed: int) -> float:
    """Train GraphSAGE for one seed and return test-set accuracy."""
    cleanup_artifacts()
    seed_all(seed)

    data, output_dim = build_graph_data()
    device = get_device()
    data = data.to(device)

    model = GraphSAGE(data.x.shape[1], output_dim, seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=OPTIMIZER_CONFIG["lr"],
        weight_decay=OPTIMIZER_CONFIG["weight_decay"],
    )

    train(data, model, optimizer, EPOCHS, evaluate)
    try:
        best_model = torch.load(BEST_MODEL_PATH, weights_only=False)
    except TypeError:
        best_model = torch.load(BEST_MODEL_PATH)
    return evaluate(best_model, data, data.test_mask)
