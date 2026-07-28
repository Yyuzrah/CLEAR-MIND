from __future__ import annotations

from pathlib import Path

from em_connectome.config import FORMAL_SEEDS, resolve_experiment

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_all_nine_experiment_configs_resolve_with_formal_protocol() -> None:
    paths = sorted((REPOSITORY_ROOT / "configs" / "experiments").glob("*.yaml"))
    assert len(paths) == 9
    combinations = set()
    for path in paths:
        config = resolve_experiment(path)
        combinations.add((config.dataset.key, config.model.name))
        assert config.seeds == FORMAL_SEEDS
        assert len(config.config_hash) == 64
        assert all(len(value) == 64 for value in config.source_hashes.values())
        assert config.spectral.method == "arakelov_green"
        assert config.spectral.top_k == 64
    assert combinations == {
        (dataset, model)
        for dataset in ("act4", "jml4", "bil6")
        for model in ("gnn_alpha", "treelstm_arch3", "mlp")
    }


def test_model_specific_result_hyperparameters_are_locked() -> None:
    mlp = resolve_experiment(REPOSITORY_ROOT / "configs" / "experiments" / "bil6_mlp.yaml")
    assert (
        mlp.trainer.epochs,
        mlp.trainer.patience,
        mlp.trainer.batch_size,
        mlp.trainer.learning_rate,
        mlp.trainer.weight_decay,
        mlp.trainer.label_smoothing,
    ) == (800, 250, 32, 0.003, 0.000001, 0.05)

    tree = resolve_experiment(
        REPOSITORY_ROOT / "configs" / "experiments" / "bil6_treelstm_arch3.yaml"
    )
    assert (
        tree.trainer.epochs,
        tree.trainer.patience,
        tree.trainer.batch_size,
        tree.trainer.learning_rate,
        tree.trainer.weight_decay,
        tree.trainer.gradient_clip,
    ) == (150, 25, 128, 0.001, 0.0001, 1.0)

    gnn = resolve_experiment(REPOSITORY_ROOT / "configs" / "experiments" / "bil6_gnn_alpha.yaml")
    assert (
        gnn.trainer.epochs,
        gnn.trainer.patience,
        gnn.trainer.batch_size,
        gnn.trainer.jitter_std,
        gnn.trainer.jitter_clip,
    ) == (150, 25, 64, 0.001, 0.005)
    assert gnn.model.parameters["num_points"] == 1024
