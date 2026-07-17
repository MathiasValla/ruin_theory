# Contributing

Thank you for considering a contribution. `ruin-theory` is currently a public
research-preview package, so the most valuable contributions are focused,
traceable, and scientifically documented.

## Ways To Contribute

- Report mathematical, numerical, or API issues.
- Add validation examples that reproduce a result from a paper.
- Improve teaching examples, notebooks, plotting diagnostics, and error
  messages.
- Add tests for an existing method before proposing broader refactors.

## Reviewer Feedback Rubric

When reviewing a method or notebook, please separate comments into:

- Mathematical correctness: assumptions, formulas, boundary conventions,
  numerical stability, and faithfulness to the source paper.
- API/usability: function names, inputs, return objects, error messages, and
  whether the example can be adapted easily.
- Teaching value: suitability for courses, tutorials, student projects, and
  practical sessions.

## Development Setup

```bash
python -m pip install -e ".[dev,docs,notebooks]"
pytest --cov=ruin_theory --cov-report=term-missing
python scripts/check_validation_notebooks.py
mkdocs build --strict
```

## Scientific Provenance

New methods should include:

- a source reference in `docs/references.md`;
- at least one unit test with a known result, hand check, or limiting case;
- a short example or validation notebook if the method will be discussed in the
  software paper;
- clear notes when a method is approximate, simulation-based, or restricted to a
  special case.

## Private Source Material

Do not commit copyrighted papers, books, or private research notes. Keep them
outside the public package and cite the original publications instead.
