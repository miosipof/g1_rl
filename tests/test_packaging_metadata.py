from __future__ import annotations

import pathlib
import tomllib


def test_pyproject_declares_src_layout_packages() -> None:
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["project"]["name"] == "g1-rl"
    assert data["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert "g1_nav_rl*" in data["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "adapters*" in data["tool"]["setuptools"]["packages"]["find"]["include"]


def test_core_packages_importable() -> None:
    import adapters  # noqa: F401
    import g1_nav_rl  # noqa: F401
