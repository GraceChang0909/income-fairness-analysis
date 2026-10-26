# Data: UCI Adult (Census Income)

Extracted from the 1994 US Census by Barry Becker. The task is to predict whether a person earns more than $50K per year.

## Sources (for more detail)

| What | URL |
|---|---|
| Official dataset page (variable descriptions, citation, license) | https://archive.ics.uci.edu/dataset/2/adult |
| Direct download used by `download_data.py` | https://archive.ics.uci.edu/static/public/2/adult.zip |
| Kaggle mirror (CSV version, community notebooks and discussion) | https://www.kaggle.com/datasets/uciml/adult-census-income |
| Original paper: Kohavi (1996), *Scaling Up the Accuracy of Naive-Bayes Classifiers* | https://cdn.aaai.org/KDD/1996/KDD96-033.pdf |
| Known issues with this dataset for fairness research: Ding et al. (2021), *Retiring Adult* | https://arxiv.org/abs/2108.04884 |

**License:** CC BY 4.0. Cite: Becker, B. & Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Files (in `raw/`, not committed; run `python data/download_data.py`)

| File | Rows | Notes |
|---|---|---|
| `adult.data` | 32,561 | Training set |
| `adult.test` | 16,281 | Official test set |
| `adult.names` | – | Original documentation |

## Columns

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Age in years |
| `workclass` | categorical | Employer type (Private, Self-emp, government, …) |
| `fnlwgt` | numeric | Census sampling weight: how many people this row represents. **Not** a personal attribute |
| `education` | categorical | Highest education level |
| `education-num` | numeric | Education level as a number (duplicates `education`) |
| `marital-status` | categorical | |
| `occupation` | categorical | |
| `relationship` | categorical | Husband, Wife, Own-child, … (strongly linked to `sex`) |
| `race` | categorical | **Sensitive attribute** |
| `sex` | categorical | **Sensitive attribute** |
| `capital-gain`, `capital-loss` | numeric | Mostly zeros, heavily skewed |
| `hours-per-week` | numeric | |
| `native-country` | categorical | |
| `income` | target | `<=50K` or `>50K` |

## Quirks to handle when loading

- No header row: column names must be supplied (use the table above, in that order)
- Values have a leading space → `skipinitialspace=True`
- Missing values are written as `?` → `na_values="?"`. Training set: `workclass` 1,836, `occupation` 1,843, `native-country` 583
- `adult.test` starts with a junk line (`|1x3 Cross validator`) → `skiprows=1`
- Test labels end with a period (`<=50K.`, `>50K.`) while training labels don't
- 24 exact duplicate rows in the training set

## First look at the groups (training set)

- Share earning >50K: **30.6% of men vs 10.9% of women**
- Race counts: White 27,816 · Black 3,124 · Asian-Pac-Islander 1,039 · Amer-Indian-Eskimo 311 · Other 271. The smallest groups are small, so their metrics will be noisy: report uncertainty
