"""Packaging safeguards must reject private material without extracting it."""

import io
from pathlib import Path
import runpy
import tarfile
import zipfile

import pytest


check_archive = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "check_release_artifacts.py"),
)["check_archive"]


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
@pytest.mark.parametrize("member", [
    "ruin_theory/__init__.py", "docs/PAPER.PDF", "ressources/book.txt",
    "private/notes.md", "resources/book.txt",
])
def test_release_archive_rejects_private_material(tmp_path, suffix, member):
    artifact = tmp_path / ("ruin_theory-0.1.0" + suffix)
    content = b"test fixture"
    if suffix == ".whl":
        with zipfile.ZipFile(artifact, "w") as archive:
            archive.writestr(member, content)
    else:
        with tarfile.open(artifact, "w:gz") as archive:
            info = tarfile.TarInfo(member)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    if member.endswith(".py"):
        check_archive(artifact)
    else:
        with pytest.raises(ValueError, match="private reference material"):
            check_archive(artifact)


def test_release_archive_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unsupported release artifact"):
        check_archive(tmp_path / "unknown.zip")
