from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from em_connectome.adapters import TreeBatch, TreeLSTMAdapter, collate_treelstm_samples
from em_connectome.models import MLP, GNNAlpha, TreeLSTMArch3

from .test_adapters import fixture_record

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIRECTORY = REPOSITORY_ROOT / "notebooks" / "reference"


def notebook_definitions(
    filename: str,
    cell_index: int,
    names: set[str],
    namespace: dict[str, object],
) -> dict[str, object]:
    notebook = json.loads((REFERENCE_DIRECTORY / filename).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][cell_index]["source"])
    tree = ast.parse(source)
    selected = [
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    exec(compile("\n\n".join(selected), filename, "exec"), namespace)
    return namespace


def assert_same_state(
    reference: nn.Module,
    packaged: nn.Module,
) -> None:
    reference_state = reference.state_dict()
    packaged_state = packaged.state_dict()
    assert reference_state.keys() == packaged_state.keys()
    for key in reference_state:
        assert reference_state[key].shape == packaged_state[key].shape
        torch.testing.assert_close(reference_state[key], packaged_state[key])
    assert sum(parameter.numel() for parameter in reference.parameters()) == sum(
        parameter.numel() for parameter in packaged.parameters()
    )


def test_mlp_parameters_logits_and_loss_match_notebook() -> None:
    namespace = notebook_definitions(
        "V.alpha_MLP_clean_champion_reproduction.ipynb",
        8,
        {"activation_layer", "ChampionMLP"},
        {
            "CHAMPION": {
                "hidden_dims": (96, 48),
                "activation": "leaky_relu",
                "dropout": 0.05,
            },
            "nn": nn,
            "torch": torch,
        },
    )
    torch.manual_seed(123)
    reference = namespace["ChampionMLP"](4)
    torch.manual_seed(123)
    packaged = MLP(4)
    assert_same_state(reference, packaged)
    assert sum(parameter.numel() for parameter in packaged.parameters()) == 11092

    features = torch.linspace(-1.0, 1.0, 3 * 64).reshape(3, 64)
    labels = torch.tensor([0, 1, 3])
    reference.eval()
    packaged.eval()
    reference_logits = reference(features)
    packaged_logits = packaged(features)
    torch.testing.assert_close(reference_logits, packaged_logits)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    torch.testing.assert_close(
        criterion(reference_logits, labels),
        criterion(packaged_logits, labels),
    )


def test_gnn_parameters_logits_and_loss_match_valpha_notebook() -> None:
    namespace = notebook_definitions(
        "V.alpha_GNN.ipynb",
        1,
        {"knn", "get_graph_feature", "TropicalMorphoGNN"},
        {"F": F, "nn": nn, "torch": torch},
    )
    torch.manual_seed(456)
    reference = namespace["TropicalMorphoGNN"](4)
    torch.manual_seed(456)
    packaged = GNNAlpha(4)
    assert_same_state(reference, packaged)

    generator = torch.Generator().manual_seed(7)
    point_cloud = torch.randn(2, 6, 32, generator=generator)
    spectrum = torch.rand(2, 64, generator=generator)
    labels = torch.tensor([0, 3])
    reference.eval()
    packaged.eval()
    reference_logits = reference(point_cloud[:, :3], spectrum)
    packaged_logits = packaged(point_cloud, spectrum)
    torch.testing.assert_close(reference_logits, packaged_logits)
    criterion = nn.CrossEntropyLoss()
    torch.testing.assert_close(
        criterion(reference_logits, labels),
        criterion(packaged_logits, labels),
    )


def test_treelstm_parameters_logits_and_loss_match_notebook() -> None:
    namespace = notebook_definitions(
        "V.alpha_TreeLSTM_arch3.ipynb",
        3,
        {"ChildSumTreeLSTMCell", "NativeTreeLSTM", "TopologicalTreeLSTM"},
        {
            "GraphBatch": TreeBatch,
            "List": list,
            "nn": nn,
            "torch": torch,
        },
    )
    torch.manual_seed(789)
    reference = namespace["TopologicalTreeLSTM"](4)
    torch.manual_seed(789)
    packaged = TreeLSTMArch3(4)
    assert_same_state(reference, packaged)

    record = fixture_record()
    batch = collate_treelstm_samples(TreeLSTMAdapter().transform_many([record, record]))
    labels = batch.labels
    reference.eval()
    packaged.eval()
    reference_logits = reference(batch)
    packaged_logits = packaged(batch)
    torch.testing.assert_close(reference_logits, packaged_logits)
    criterion = nn.CrossEntropyLoss()
    torch.testing.assert_close(
        criterion(reference_logits, labels),
        criterion(packaged_logits, labels),
    )
