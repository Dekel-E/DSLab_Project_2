"""
Student implementation file — implement all TODO sections below.

This is the only Section B file you should submit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def get_feature_vectors(nodes_df):
    """Return node feature matrix x as a float torch.Tensor."""
    # Each `features` cell is a string like "[0, 0, 1, ...]" with 1,433 binary ints.
    # `np.fromstring` parses the comma-separated numbers (ast/json are not allowed imports).
    feats = np.stack(
        [np.fromstring(s.strip("[]"), sep=",") for s in nodes_df["features"]]
    )
    x = torch.tensor(feats, dtype=torch.float)
    # Row-normalize the bag-of-words features (standard preprocessing for Cora);
    # gives a small, consistent accuracy boost and more margin above threshold.
    x = x / x.sum(dim=1, keepdim=True).clamp(min=1.0)
    return x


def get_edges(edges_df, inverse_node_id_mapping):
    """Return edge_index as a long torch.Tensor of shape [2, num_edges]."""
    # Edges are keyed by original nodeId; remap both endpoints to internal indices.
    src = edges_df["sourceNodeId"].map(inverse_node_id_mapping).to_numpy()
    dst = edges_df["targetNodeId"].map(inverse_node_id_mapping).to_numpy()
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def get_labels(nodes_df, subject_mapping):
    """Return node labels y as a long torch.Tensor."""
    y = nodes_df["subject"].map(subject_mapping).to_numpy()
    return torch.tensor(y, dtype=torch.long)


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, output_dim, seed):
        super().__init__()
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        # Note: the framework passes data.x.shape[1] (the input dim, 1433) as
        # `hidden_channels`, so this argument is the number of input features.
        in_dim = hidden_channels
        hidden = 256
        self.dropout = 0.5
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(data, model, optimizer, epochs, evaluate_fn):
    """
    Train the model for the given number of epochs.

    Use evaluate_fn(model, data.valid_mask) to track validation accuracy.
    Save the best checkpoint to 'best_model.pt' in the current working directory.
    """
    best_val_acc = -1.0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Track validation accuracy; save the best full model (run.py reloads the
        # whole object and calls it directly).
        val_acc = evaluate_fn(model, data, data.valid_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model, "best_model.pt")

    # Guarantee a checkpoint exists even if validation never improved.
    if best_val_acc < 0:
        torch.save(model, "best_model.pt")
