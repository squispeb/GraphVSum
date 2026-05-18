import numpy as np
from sklearn.metrics import average_precision_score


def getScores(y_true, y_pred):
    if isinstance(y_true, list) and y_true and isinstance(y_true[0], np.ndarray):
        y_true = np.concatenate(y_true)
    elif isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list) and y_pred and isinstance(y_pred[0], np.ndarray):
        y_pred = np.concatenate(y_pred)
    elif isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true) == 0:
        return 0.0, 0.0

    ap = average_precision_score((y_true > 0.5).astype(int), y_pred)
    pred_binary = (y_pred > 0.5).astype(int)
    true_binary = (y_true > 0.5).astype(int)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return ap, f1
