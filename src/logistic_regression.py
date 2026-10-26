"""Logistic regression implemented from scratch with NumPy."""
import numpy as np


def sigmoid(z):
    # Clip to avoid overflow in exp for very large |z|
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionScratch:
    """Binary logistic regression trained with mini-batch gradient descent.

    Minimizes the (weighted) mean cross-entropy plus an L2 penalty on the
    coefficients (the intercept is not penalized):

        L(w, b) = sum_i s_i * CE(y_i, sigmoid(x_i w + b)) / sum_i s_i + (l2 / 2) * ||w||^2

    Parameters
    ----------
    learning_rate : float
    epochs : int
    batch_size : int
    l2 : float
        L2 penalty strength.
    random_state : int or None
    """

    def __init__(self, learning_rate=0.1, epochs=20, batch_size=64, l2=0.0, random_state=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    def _loss(self, X, y, s):
        p = np.clip(self.predict_proba(X), 1e-12, 1 - 1e-12)
        ce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.sum(s * ce) / np.sum(s) + 0.5 * self.l2 * self.coef_ @ self.coef_

    def fit(self, X, y, sample_weight=None):
        """Train on X (n, p) and y (n,) with values in {0, 1}.

        `sample_weight` (n,) scales each observation's contribution to the loss,
        which is used for group reweighting.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        s = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        s = s * n / s.sum()  # normalize so the average weight is 1

        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros(p)
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                Xb, yb, sb = X[idx], y[idx], s[idx]
                # Gradient of the weighted cross-entropy: X^T (s * (p - y)) / sum(s)
                err = sb * (sigmoid(Xb @ self.coef_ + self.intercept_) - yb)
                grad_w = Xb.T @ err / sb.sum() + self.l2 * self.coef_
                grad_b = err.sum() / sb.sum()
                self.coef_ -= self.learning_rate * grad_w
                self.intercept_ -= self.learning_rate * grad_b
            self.loss_history_.append(self._loss(X, y, s))
        return self

    def predict_proba(self, X):
        """Return P(y = 1 | x) for each row, shape (n,)."""
        return sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        """Return 0/1 predictions, shape (n,)."""
        return (self.predict_proba(X) >= threshold).astype(int)
