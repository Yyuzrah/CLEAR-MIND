from __future__ import annotations

from pathlib import Path

from em_connectome.cli import build_parser, main
from em_connectome.verify import verify_results

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_required_cli_command_tree_is_registered() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in ("data", "spectral", "train", "reproduce", "verify"):
        assert command in help_text


def test_result_verification_command_is_registered() -> None:
    parser = build_parser()
    args = parser.parse_args(["verify", "results"])
    assert args.target == "results"


def test_published_formal_matrix_passes_regression_gate() -> None:
    result = verify_results(REPOSITORY_ROOT)
    assert result["ok"] is True
    assert result["found_runs"] == 45
    assert len(result["experiments"]) == 9


def test_adapter_verification_runs_without_raw_data(capsys) -> None:
    status = main(
        [
            "--repository-root",
            str(REPOSITORY_ROOT),
            "verify",
            "adapters",
        ]
    )
    captured = capsys.readouterr()
    assert status == 0
    assert '"ok": true' in captured.out
