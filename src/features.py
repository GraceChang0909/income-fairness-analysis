"""Turn the cleaned Adult tables into numeric feature matrices."""
import numpy as np
import pandas as pd

NUMERIC = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
SENSITIVE = ["sex", "race"]
TARGET = "income"


class FeatureBuilder:
    """One-hot encode categoricals and standardize everything, using training statistics only.

    Parameters
    ----------
    include_sensitive : bool
        Whether `sex` and `race` are used as model inputs.
    exclude : tuple of str
        Categorical columns to leave out (e.g. proxies for a sensitive attribute).
    """

    def __init__(self, include_sensitive=True, exclude=()):
        self.include_sensitive = include_sensitive
        self.exclude = tuple(exclude)

    def _raw(self, df):
        X = df[NUMERIC].astype(float).copy()
        # Capital gains/losses are mostly zero with a long right tail
        X["capital-gain"] = np.log1p(X["capital-gain"])
        X["capital-loss"] = np.log1p(X["capital-loss"])
        cats = [c for c in CATEGORICAL if c not in self.exclude] + (SENSITIVE if self.include_sensitive else [])
        dummies = pd.get_dummies(df[cats], prefix_sep="=", dtype=float)
        return pd.concat([X, dummies], axis=1)

    def fit(self, df):
        X = self._raw(df)
        self.columns_ = X.columns
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0)
        return self

    def transform(self, df):
        # Categories unseen in training become all-zero dummies
        X = self._raw(df).reindex(columns=self.columns_, fill_value=0.0)
        return ((X - self.mean_) / self.std_).to_numpy()

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def sensitive_columns(self):
        """Indices of the columns derived from sex/race (empty if not included)."""
        return [i for i, c in enumerate(self.columns_) if c.split("=")[0] in SENSITIVE]


def load_processed(path="../data/processed"):
    train = pd.read_csv(f"{path}/adult_train_clean.csv")
    test = pd.read_csv(f"{path}/adult_test_clean.csv")
    return train, test


def race_group(race):
    """Merge the two smallest race categories (Amer-Indian-Eskimo, Other; ~580 people together)
    so that group metrics are not pure noise."""
    return race.where(race.isin(["White", "Black", "Asian-Pac-Islander"]), "Other")
