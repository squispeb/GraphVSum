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


def knapsack(values, weights, capacity):
    capacity = int(capacity)
    n_items = len(values)
    if capacity <= 0 or n_items == 0:
        return []

    table = np.zeros((n_items + 1, capacity + 1), dtype=np.float32)
    keep = np.zeros((n_items + 1, capacity + 1), dtype=np.bool_)
    for i in range(1, n_items + 1):
        weight = int(weights[i - 1])
        value = float(values[i - 1])
        for cap in range(capacity + 1):
            table[i, cap] = table[i - 1, cap]
            if weight <= cap:
                candidate = table[i - 1, cap - weight] + value
                if candidate > table[i, cap]:
                    table[i, cap] = candidate
                    keep[i, cap] = True

    selected = []
    cap = capacity
    for i in range(n_items, 0, -1):
        if keep[i, cap]:
            selected.append(i - 1)
            cap -= int(weights[i - 1])
    selected.reverse()
    return selected


def build_tvsum_summary(pred_scores, change_points, n_frame_per_seg, picks, n_frames,
                        budget_ratio=0.15):
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    change_points = np.asarray(change_points, dtype=np.int64)
    n_frame_per_seg = np.asarray(n_frame_per_seg, dtype=np.int64)
    picks = np.asarray(picks, dtype=np.int64)
    n_frames = int(n_frames)

    valid_picks = picks >= 0
    picks = picks[valid_picks]
    pred_scores = pred_scores[: len(picks)]
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    if len(picks):
        frame_scores[picks] = pred_scores

    shot_scores = []
    for start, end in change_points:
        if start < 0 or end < 0:
            continue
        end = min(int(end), n_frames - 1)
        start = min(int(start), end)
        shot_scores.append(float(frame_scores[start:end + 1].mean()))

    n_segments = min(len(shot_scores), len(n_frame_per_seg))
    shot_scores = np.asarray(shot_scores[:n_segments], dtype=np.float32)
    shot_lengths = np.asarray(n_frame_per_seg[:n_segments], dtype=np.int64)
    budget = int(np.floor(n_frames * budget_ratio))
    selected = knapsack(shot_scores, shot_lengths, budget)

    summary = np.zeros(n_frames, dtype=np.float32)
    for idx in selected:
        start, end = change_points[idx]
        if start < 0 or end < 0:
            continue
        summary[int(start):min(int(end) + 1, n_frames)] = 1
    return summary


def evaluate_summary(pred_summary, user_summary):
    user_summary = np.asarray(user_summary, dtype=np.float32)
    if user_summary.ndim == 1:
        user_summary = user_summary[None, :]

    scores = []
    for user in user_summary:
        length = min(len(pred_summary), len(user))
        pred = pred_summary[:length] > 0
        gt = user[:length] > 0
        overlap = np.logical_and(pred, gt).sum()
        precision = overlap / (pred.sum() + 1e-8)
        recall = overlap / (gt.sum() + 1e-8)
        scores.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(scores)) if scores else 0.0


def getTVSumScores(y_true, y_pred, metadata, budget_ratio=0.15):
    ap, _ = getScores(y_true, y_pred)
    f_scores = []
    for scores, meta in zip(y_pred, metadata):
        pred_summary = build_tvsum_summary(
            scores,
            meta["change_points"],
            meta["n_frame_per_seg"],
            meta["picks"],
            meta["n_frames"],
            budget_ratio=budget_ratio,
        )
        f_scores.append(evaluate_summary(pred_summary, meta["user_summary"]))
    return ap, float(np.mean(f_scores)) if f_scores else 0.0
