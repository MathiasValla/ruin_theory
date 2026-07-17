# Validation Notebooks

These notebooks provide small, paper-specific numerical reproductions for
methods that were previously validated mostly through internal consistency
checks.

Each notebook asks reviewers to separate comments into three categories:

- mathematical correctness;
- API/usability;
- teaching value.

Current notebooks:

- `markov_modulated_multirisk_reproduction.ipynb`
- `multirisk_ctmc_dividends_reproduction.ipynb`
- `loisel_privault_sensitivity_reproduction.ipynb`
- `inar_binar_byclaim_reproduction.ipynb`

From the repository root, regenerate the notebooks with:

```bash
python scripts/create_validation_notebooks.py
```

Run a lightweight execution check without Jupyter using:

```bash
python scripts/check_validation_notebooks.py
```
