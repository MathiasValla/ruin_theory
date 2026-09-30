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

## Review And Refactoring Workflow

The 2026 package review applied the Python review, regression-first refactoring
and verification workflow from [Everything Claude Code](https://github.com/affaan-m/ECC),
and adapted the [deep `/cleanse` procedure](https://github.com/ulfaslak/saas_tmplt/blob/main/.claude/commands/cleanse.md)
to this scientific Python library. No external hooks or services are needed.
See the [review report](docs/code_review_2026_09.md) for numerical changes,
verification results, benchmarks and outstanding limits.

For future changes:

1. Compare the implementation against its documented assumptions and provenance
   row. Identify strict/non-strict ruin boundaries, truncation and normalization.
2. Reproduce a defect with an independent calculation, known distribution or
   limiting case before correcting it. Agreement between two related functions
   is not sufficient scientific validation by itself.
3. Preserve public names and test edge cases before consolidating duplicated
   internal computations. Public exports are not dead code merely because no
   internal caller uses them. Keep independent reproduction tests.
4. Measure optimization claims on fixed inputs and seeds. The sensitivity
   benchmark is `python scripts/benchmark_sensitivity.py`; timings are reported,
   not used as machine-dependent pass/fail thresholds.
5. Run lint, the full tests with coverage, validation notebooks, the strict docs
   build and distribution checks. Describe numerical changes and remaining
   limitations in the review report.

For release verification, install the `release` extra, then run
`python -m build`, `python -m twine check dist/*` and
`python scripts/check_release_artifacts.py dist/*.tar.gz dist/*.whl`.

## Private Source Material

Do not commit copyrighted papers, books, or private research notes. Keep them
outside the public package and cite the original publications instead.
