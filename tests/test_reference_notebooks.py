from __future__ import annotations

from pathlib import Path

import nbformat

from em_connectome.data import sha256_file

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "reference"


def test_reference_notebooks_are_valid_and_byte_identical() -> None:
    expected = {}
    for line in (REFERENCE_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest

    assert set(expected) == {path.name for path in REFERENCE_ROOT.glob("*.ipynb")}
    for filename, digest in expected.items():
        path = REFERENCE_ROOT / filename
        assert sha256_file(path) == digest
        nbformat.validate(nbformat.read(path, as_version=4))
