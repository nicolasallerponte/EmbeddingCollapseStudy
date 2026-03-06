"""
Linear evaluation protocol.
Trains a logistic regression on frozen representations.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 1000,
) -> dict:
    """
    Fits a linear probe on frozen embeddings and returns accuracy.

    Args:
        z_train: (N_train, D) train embeddings
        y_train: (N_train,) train labels
        z_test:  (N_test, D) test embeddings
        y_test:  (N_test,) test labels

    Returns:
        dict with train_acc and test_acc
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=0.1),
    )
    clf.fit(z_train, y_train)

    return {
        "train_acc": clf.score(z_train, y_train),
        "test_acc": clf.score(z_test, y_test),
    }
