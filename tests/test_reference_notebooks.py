from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat

from em_connectome.data import sha256_file

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "notebooks" / "reference"
REPOSITORY_ROOT = REFERENCE_ROOT.parents[1]
MLP_NOTEBOOK = "V.alpha_MLP_clean_champion_reproduction.ipynb"
MLP_COMPUTATIONAL_SHA256 = "12c49567f9fb0d5f085742743fa51a45b663e3b696f59662fd3b7ea74d159ad7"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def test_reference_notebooks_are_valid_and_match_public_hashes() -> None:
    expected = {}
    for line in (REFERENCE_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        expected[filename] = digest

    assert set(expected) == {path.name for path in REFERENCE_ROOT.glob("*.ipynb")}
    for filename, digest in expected.items():
        path = REFERENCE_ROOT / filename
        assert sha256_file(path) == digest
        nbformat.validate(nbformat.read(path, as_version=4))


def test_mlp_notebook_computational_record_matches_historical_source() -> None:
    notebook = nbformat.read(REFERENCE_ROOT / MLP_NOTEBOOK, as_version=4)
    payload = {
        "nbformat": notebook.nbformat,
        "nbformat_minor": notebook.nbformat_minor,
        "metadata": notebook.metadata,
        "cells": [
            {
                "cell_type": "markdown",
                "id": cell.get("id"),
                "metadata": cell.metadata,
            }
            if cell.cell_type == "markdown"
            else cell
            for cell in notebook.cells
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == MLP_COMPUTATIONAL_SHA256


def test_public_text_contains_no_cjk_characters() -> None:
    text_suffixes = {
        ".csv",
        ".ipynb",
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    excluded = {".git", ".pytest_cache", ".ruff_cache", "cache", "runs", "__pycache__"}
    offenders = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in excluded for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if CJK_PATTERN.search(text):
            offenders.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert offenders == []
