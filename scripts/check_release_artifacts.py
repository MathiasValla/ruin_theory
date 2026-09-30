"""Reject private reference material in wheel and source distributions."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def check_archive(path: Path) -> None:
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
    else:
        raise ValueError(f"unsupported release artifact: {path}")
    forbidden = []
    for name in names:
        entry = PurePosixPath(name.lower())
        if entry.suffix == ".pdf" or {"ressources", "resources", "private"}.intersection(entry.parts):
            forbidden.append(name)
    if forbidden:
        raise ValueError(f"private reference material in {path}: {', '.join(forbidden)}")
    print(f"checked {path.name}: {len(names)} entries, no PDF or private reference directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.artifacts:
        check_archive(path)


if __name__ == "__main__":
    main()
