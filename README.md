# A-STAR Firm Dynamics

Exploratory analysis of Singapore firm survival dynamics using reconstructed ACRA data.

The repository is organized so reusable analysis code lives in `firm_dynamics/`, active notebooks live in `notebooks/`, generated CSV summaries live in `outputs/`, and discontinued experiments live in `legacy/`.

## Layout

- `firm_dynamics/data.py`: load and prepare ACRA-derived data.
- `firm_dynamics/survival.py`: compute age bins, survivor counts, and survival fractions.
- `firm_dynamics/models/`: constant, power-law, Hill, and one-tail perturbation survival models.
- `firm_dynamics/fitting.py`: shared likelihood, AIC, and BIC helpers.
- `notebooks/`: active exploratory notebooks.
- `legacy/`: old notebooks and discontinued two-tail perturbation code.
- `outputs/`: saved model and clustering summary CSVs.
- `tests/`: lightweight smoke/math tests for the reusable code.

## Data

`ACRA_w_SO.csv` is intentionally ignored and is not included in the repository. Place it at the repository root before running the notebooks or `prepare_df()`.

## Quick Checks

```powershell
python -m unittest discover -s tests -v
```
