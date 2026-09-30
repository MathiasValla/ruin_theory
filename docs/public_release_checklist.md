# Public Release Checklist

This checklist prepares a public package release. It does not replace
scientific review.

## Package Metadata

- Confirm package name on PyPI.
- Replace any placeholder repository URL in project metadata and documentation
  once the public repository exists.
- Confirm `LICENSE`, `CITATION.cff`, and `.zenodo.json`.
- Build and inspect source/wheel distributions:

```bash
python -m build
python -m twine check dist/*
```

## Tests And Coverage

- Run the full test suite with coverage:

```bash
pytest --cov=ruin_theory --cov-report=term-missing --cov-report=html
```

- Run validation notebooks:

```bash
python scripts/check_validation_notebooks.py
```

- Build the documentation site:

```bash
mkdocs build --strict
```

- Enable GitHub Pages from GitHub Actions once the public repository exists.
  The `Docs` workflow deploys `mkdocs.yml` to Pages from the `main` branch.

## Private Material

- Do not publish `../ressources/` or any copyrighted PDFs, books, manuscript
  drafts, or private notes.
- Publish citations, tests, notebooks and derived validation tables instead of
  copyrighted source files.
- Check the release archive before upload:

```bash
python scripts/check_release_artifacts.py dist/*.tar.gz dist/*.whl
```

This check fails on PDFs and private reference directories in either artifact.
Generated PDF figures can stay outside the release archives; publish them
separately as supplementary material when appropriate.

## Zenodo

- Enable Zenodo only after the public repository is ready.
- Mint the DOI from a tagged release.
- Add the final DOI to `CITATION.cff`, README, and the manuscript.

## Transparency

- Keep the README development-status warning visible.
- Keep the AI/Codex assistance disclosure visible.
- Clearly identify methods that are simulation-based, approximate, or still
  under scientific review.
