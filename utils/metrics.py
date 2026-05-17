import numpy as np


def getScores(y_true, y_pred):
    y_true = np.concatenate([np.asarray(x).reshape(-1) for x in y_true])
    y_pred = np.concatenate([np.asarray(x).reshape(-1) for x in y_pred])
    if y_true.size == 0:
        return 0.0, 0.0

    try:
        from sklearn.metrics import average_precision_score, f1_score
        ap = average_precision_score(y_true, y_pred) if np.any(y_true == 1) else 0.0
        f1 = f1_score(y_true, y_pred >= 0.5, zero_division=0)
        return float(ap), float(f1)
    except Exception:
        pred = y_pred >= 0.5
        true = y_true.astype(bool)
        tp = np.logical_and(pred, true).sum()
        fp = np.logical_and(pred, ~true).sum()
        fn = np.logical_and(~pred, true).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return float(precision), float(f1)
