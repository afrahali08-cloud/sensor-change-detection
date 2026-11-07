"""Mean-shift segmentation and isolated-outlier detection."""
import numpy as np


def signal(values):
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("Expected a nonempty finite univariate signal")
    return y


def cusum(values, mean, scale, drift=0.5, threshold=8):
    """Return the first two-sided alarm relative to a fixed reference."""
    y = signal(values)
    if scale <= 0 or threshold <= 0 or drift < 0:
        raise ValueError("Expected positive scale and threshold and nonnegative drift")
    up = down = 0.0
    for i, z in enumerate((y - mean) / scale):
        up = max(0, up + z - drift)
        down = max(0, down - z - drift)
        if max(up, down) >= threshold:
            return i
    return None


def segment(values, penalty=40, min_size=25, backend="exact"):
    """Return internal segment starts, excluding the terminal index."""
    y = signal(values)
    n = len(y)
    if penalty <= 0 or not 1 <= min_size <= n:
        raise ValueError("Expected positive penalty and a feasible minimum length")
    if backend == "pelt":
        import ruptures
        return ruptures.Pelt(model="l2", min_size=min_size, jump=1).fit_predict(
            y, pen=penalty)[:-1]
    if backend != "exact":
        raise ValueError("Expected exact or pelt backend")

    # Center first to reduce cancellation in the prefix-sum SSE formula.
    y = y - y.mean()
    sums = np.r_[0, np.cumsum(y)]
    squares = np.r_[0, np.cumsum(y**2)]
    cost = np.full(n + 1, np.inf)
    cost[0] = -penalty
    previous = np.zeros(n + 1, dtype=int)
    for end in range(min_size, n + 1):
        starts = np.arange(end - min_size + 1)
        sse = squares[end] - squares[starts] - (sums[end] - sums[starts])**2 / (end - starts)
        candidates = cost[starts] + np.maximum(sse, 0) + penalty
        best = int(np.argmin(candidates))
        previous[end] = starts[best]
        cost[end] = candidates[best]
    changes = []
    while previous[n]:
        n = int(previous[n])
        changes.append(n)
    return changes[::-1]


def spikes(values, window=15, threshold=6, noise_floor=0.5):
    """Score isolated outliers against neighboring samples, excluding the target."""
    y = signal(values)
    if window < 3 or window % 2 == 0 or threshold <= 0 or noise_floor <= 0:
        raise ValueError("Expected an odd window >= 3 and positive thresholds")
    half = window // 2
    if len(y) < window:
        return np.array([], dtype=int)
    windows = np.lib.stride_tricks.sliding_window_view(y, window)
    neighbors = np.concatenate([windows[:, :half], windows[:, half + 1:]], axis=1)
    median = np.median(neighbors, axis=1)
    mad = np.median(np.abs(neighbors - median[:, None]), axis=1)
    scores = np.abs(windows[:, half] - median) / np.maximum(1.4826 * mad, noise_floor)
    return np.flatnonzero(scores >= threshold) + half

