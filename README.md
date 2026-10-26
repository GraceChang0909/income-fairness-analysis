# Income Prediction & Fairness Analysis

> 🚧 Work in progress. Sections marked TODO will be filled in as the analysis is completed.

**Question:** Can we predict whether someone earns more than $50K a year, and does the model work equally well for men and women and across racial groups?

Models like this are used in lending, hiring and marketing. A model can look accurate overall while systematically failing one group. This project builds an income classifier from scratch, audits it for group-level gaps, and tests ways to reduce them.

## Results

TODO: headline numbers (overall accuracy, largest fairness gap before → after mitigation) and one key figure.

## Approach

| Notebook | What it does |
|---|---|
| [01 Data cleaning & EDA](notebooks/01_data_cleaning_eda.ipynb) | Clean the raw census files, explore income gaps by group, detect proxy features |
| [02 Logistic regression from scratch](notebooks/02_logistic_regression_from_scratch.ipynb) | NumPy implementation verified against scikit-learn; tuned baseline model |
| [03 Fairness evaluation & mitigation](notebooks/03_fairness_evaluation.ipynb) | Per-group audit, then compare unawareness, reweighting and threshold adjustment |

Reusable code lives in [`src/`](src/): the logistic regression class and the group-fairness metrics.

## Key findings

TODO: 3–5 bullets.

## Data

[UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult), 48,842 people from the 1994 US Census, CC BY 4.0. Details, column descriptions and loading quirks: [data/README.md](data/README.md).

## Run it yourself

```bash
pip install -r requirements.txt
python data/download_data.py
jupyter lab
```

Run the notebooks in order from inside the `notebooks/` folder.

## Tech stack

Python · NumPy · pandas · scikit-learn · Matplotlib · Jupyter
