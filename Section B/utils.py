"""
Provided framework utilities — do NOT modify this file.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

_STUDENT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _STUDENT_DIR / "constants.yaml"

with open(_CONFIG_PATH, encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

EPOCHS = CONFIG["epochs"]
ACCURACY_THRESHOLD = CONFIG["accuracy_threshold"]
SECTION_B_POINTS = CONFIG["section_b_points"]
SEEDS = CONFIG["seeds"]
OPTIMIZER_CONFIG = CONFIG["optimizer"]
BEST_MODEL_PATH = _STUDENT_DIR / "best_model.pt"


def _data_path(relative_path: str) -> Path:
    return _STUDENT_DIR / relative_path


def read_data(nodes_df_path: str, edges_df_path: str, subject_mapping_path: str):
    nodes_df = pd.read_csv(_data_path(nodes_df_path))
    edges_df = pd.read_csv(_data_path(edges_df_path))
    with open(_data_path(subject_mapping_path), "rb") as f:
        subject_mapping = pickle.load(f)
    return nodes_df, edges_df, subject_mapping


def get_node_id_mapping(nodes_df: pd.DataFrame):
    node_id_mapping: dict[int, int] = {}
    inverse_node_id_mapping: dict[int, int] = {}
    for i, node_id in enumerate(nodes_df["nodeId"]):
        node_id_mapping[i] = node_id
        inverse_node_id_mapping[node_id] = i
    return node_id_mapping, inverse_node_id_mapping


def build_graph_data():
    """Load CSV/pickle artifacts and construct a PyG Data object."""
    from gnn import get_edges, get_feature_vectors, get_labels

    data_cfg = CONFIG["data"]
    nodes_df, edges_df, subject_mapping = read_data(
        data_cfg["nodes"],
        data_cfg["edges"],
        data_cfg["subject_mapping"],
    )
    node_id_mapping, inverse_node_id_mapping = get_node_id_mapping(nodes_df)

    x = get_feature_vectors(nodes_df)
    edge_index = get_edges(edges_df, inverse_node_id_mapping)
    y = get_labels(nodes_df, subject_mapping)

    with open(_data_path(data_cfg["indices_dict"]), "rb") as f:
        indices_dict = pickle.load(f)

    num_nodes = x.shape[0]
    train_mask = torch.tensor(
        [node_id_mapping[i] in indices_dict["train_indices"] for i in range(num_nodes)],
        dtype=torch.bool,
    )
    valid_mask = torch.tensor(
        [node_id_mapping[i] in indices_dict["valid_indices"] for i in range(num_nodes)],
        dtype=torch.bool,
    )
    test_mask = torch.tensor(
        [node_id_mapping[i] in indices_dict["test_indices"] for i in range(num_nodes)],
        dtype=torch.bool,
    )

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )
    output_dim = len(subject_mapping)
    return data, output_dim


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model: torch.nn.Module, data: Data, rel_mask: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=1)
        correct = (preds[rel_mask] == data.y[rel_mask]).sum()
        return round(int(correct) / int(rel_mask.sum()), 3)


def compute_section_b_score(test_scores: dict[int, float]) -> tuple[float, int]:
    """Return (score out of section_b_points, total penalty points)."""
    penalty = 0
    for seed in SEEDS:
        accuracy = test_scores[seed]
        if accuracy < ACCURACY_THRESHOLD:
            penalty += int((ACCURACY_THRESHOLD - accuracy) // 0.01)
    score = max(0.0, SECTION_B_POINTS - penalty)
    return score, penalty


def cleanup_artifacts() -> None:
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()
