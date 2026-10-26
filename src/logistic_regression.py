"""Logistic regression implemented from scratch with NumPy.

TODO (you): fill in the methods. Your DSM assignment 3 softmax classifier is a good
starting point; the binary case only needs a sigmoid instead of a softmax.
"""
import numpy as np


class LogisticRegressionScratch:
    """Binary logistic regression trained with mini-batch gradient descent.

    Parameters
    ----------
    learning_rate : float
    epochs : int
    batch_size : int
    l2 : float
        L2 penalty strength (do not penalize the intercept).
    random_state : int or None
    """

    def __init__(self, learning_rate=0.1, epochs=20, batch_size=64, l2=0.0, random_state=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Train on X (n, p) and y (n,) with values in {0, 1}.

        `sample_weight` (n,) is needed later for group reweighting: multiply each
        observation's gradient contribution by its weight.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """Return P(y = 1 | x) for each row, shape (n,)."""
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        """Return 0/1 predictions, shape (n,)."""
        raise NotImplementedError
