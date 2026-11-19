from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from em_connectome.adapters import (
    GNNAdapter,
    MLPAdapter,
    SpectrumStandardizer,
    TreeLSTMAdapter,
    collate_gnn_samples,
    collate_treelstm_samples,
)
from em_connectome.data.records import NeuronRecord
from em_connectome.data.swc import parse_swc
from em_connectome.spectral import ArakelovGreenSpectrum

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "simple.swc"
GOLDEN = REPOSITORY_ROOT / "tests" / "golden" / "adapters_v1.json"


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def fixture_record() -> NeuronRecord:
    morphology = parse_swc(FIXTURE)
    record = NeuronRecord(
        sample_id="act4/simple.swc",
        dataset="act4",
        swc_path=FIXTURE,
        swc_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        label_id=0,
        label_name="Isocortex_layer23",
        fold=0,
        phase="train",
        morphology=morphology,
    )
    return record.with_spectrum(ArakelovGreenSpectrum().extract_tree(morphology))


def test_mlp_adapter_and_train_only_standardization() -> None:
    record = fixture_record()
    sample = MLPAdapter().transform(record)
    np.testing.assert_array_equal(sample.features, record.spectrum)
    assert sample.features.dtype == np.float64
    assert sample.label == record.label_id

    training = np.vstack([sample.features, sample.features + 2.0])
    standardizer = SpectrumStandardizer.fit(training)
    normalized = standardizer.transform(training)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-12)


def test_gnn_adapter_matches_formal_baseline_cache_source() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    sample = GNNAdapter().transform(fixture_record())
    expected = golden["gnn"]
    assert list(sample.point_cloud.shape) == expected["point_cloud_shape"]
    assert array_hash(sample.point_cloud) == expected["point_cloud_sha256"]
    assert array_hash(sample.tree_parent) == expected["tree_parent_sha256"]
    assert sample.spectrum.shape == (64,)
    assert sample.label == 0


def test_treelstm_adapter_matches_notebook_cell() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    sample = TreeLSTMAdapter().transform(fixture_record())
    expected = golden["treelstm"]
    assert list(sample.x.shape) == expected["x_shape"]
    assert array_hash(sample.x.numpy()) == expected["x_sha256"]
    assert sample.parent_indices.tolist() == expected["parent_indices"]
    assert [level.tolist() for level in sample.levels] == expected["levels"]
    assert sample.root_idx == expected["root_idx"]
    assert array_hash(sample.spectral_sig.numpy()) == expected["spectral_sha256"]
    assert sample.label == expected["label"]


def test_collators_preserve_sample_spectrum_label_alignment() -> None:
    first = fixture_record()
    second = replace(first, sample_id="act4/simple-copy.swc", label_id=1)

    gnn_batch = collate_gnn_samples(GNNAdapter(num_points=32).transform_many([first, second]))
    assert gnn_batch["sample_id"] == [first.sample_id, second.sample_id]
    assert tuple(gnn_batch["point_cloud"].shape) == (2, 6, 32)
    assert gnn_batch["label"].tolist() == [0, 1]
    torch.testing.assert_close(
        gnn_batch["spectral_sig"][0],
        torch.tensor(first.spectrum, dtype=torch.float32),
    )

    tree_batch = collate_treelstm_samples(TreeLSTMAdapter().transform_many([first, second]))
    assert tree_batch.sample_ids == [first.sample_id, second.sample_id]
    assert tree_batch.labels.tolist() == [0, 1]
    assert tree_batch.root_indices.numel() == 2
    assert tree_batch.total_nodes == tree_batch.x.shape[0]
    assert all(
        index == tree_batch.total_nodes
        for index in tree_batch.parent_indices[tree_batch.root_indices].tolist()
    )
