from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from em_connectome.data.swc import parse_swc

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_swc_preserves_file_order_and_parent_indices() -> None:
    tree = parse_swc(FIXTURES / "simple.swc")

    np.testing.assert_array_equal(tree.node_ids, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(tree.node_types, [1, 3, 3, 3, 3])
    np.testing.assert_array_equal(tree.parent_ids, [-1, 1, 2, 2, 2])
    np.testing.assert_array_equal(tree.parent_indices, [-1, 0, 1, 1, 1])
    np.testing.assert_allclose(
        tree.coordinates,
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ],
    )
    assert tree.root_index == 0


def test_parse_swc_rejects_duplicate_node_ids(tmp_path: Path) -> None:
    bad = tmp_path / "duplicate.swc"
    bad.write_text("1 1 0 0 0 1 -1\n1 3 1 0 0 1 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate node IDs"):
        parse_swc(bad)
