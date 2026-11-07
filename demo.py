"""Benchmark synthetic mean shifts and isolated sensor spikes."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from sensor import cusum, segment, spikes


def generate(seed):
    rng = np.random.default_rng(seed)
    changes = [180, 360, 530]
    clean = np.repeat([0., 2.2, -0.9, 1.2], [180, 180, 170, 170])
    y = clean + rng.normal(0, 0.55, clean.size)
    candidates = [i for i in range(65, len(y) - 8)
                  if all(abs(i - change) > 8 for change in changes)]
    anomalies = sorted(rng.choice(candidates, 10, replace=False))
    y[anomalies] += rng.choice([-1, 1], len(anomalies)) * rng.uniform(4, 7, len(anomalies))
    return y, changes, anomalies


def change_errors(truth, detected, tolerance=8):
    if not truth or not detected:
        return []
    errors = np.abs(np.subtract.outer(truth, detected))
    penalty = (min(errors.shape) + 1) * tolerance
    rows, cols = linear_sum_assignment(np.where(errors <= tolerance, errors, penalty))
    return [int(errors[i, j]) for i, j in zip(rows, cols) if errors[i, j] <= tolerance]


def metrics(tp, predicted, expected):
    return dict(tp=tp, fp=predicted - tp, fn=expected - tp,
                precision=tp / max(predicted, 1), recall=tp / expected,
                f1=2 * tp / (predicted + expected))


def run(out, backend):
    out.mkdir(parents=True, exist_ok=True)
    errors, delays = [], []
    predicted = spike_predictions = spike_hits = early = missed = 0
    elapsed = 0
    for seed in range(21, 41):
        y, truth, anomalies = generate(seed)
        start = perf_counter()
        flagged = spikes(y)
        cleaned = y.copy()
        for i in flagged:
            cleaned[i] = np.median(np.r_[y[i-7:i], y[i+1:i+8]])
        detected = segment(cleaned, backend=backend)
        # CUSUM sees raw streaming samples after the initial calibration period.
        alarm = cusum(y[60:], y[:60].mean(), y[:60].std(ddof=1))
        elapsed += perf_counter() - start
        errors.extend(change_errors(truth, detected))
        predicted += len(detected)
        spike_predictions += len(flagged)
        spike_hits += len(set(flagged) & set(anomalies))
        if alarm is None or alarm + 60 >= truth[1]:
            missed += 1
        elif alarm + 60 < truth[0]:
            early += 1
        else:
            delays.append(alarm + 60 - truth[0])
        if seed == 21:
            with (out / "example.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample", "observed", "cleaned", "true_change",
                                 "detected_change", "true_spike", "detected_spike"])
                writer.writerows((i, y[i], cleaned[i], int(i in truth), int(i in detected),
                                  int(i in anomalies), int(i in flagged)) for i in range(len(y)))
            fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
            ax.plot(y, label="Observed", alpha=.55)
            ax.plot(cleaned, label="Cleaned", lw=.9)
            ax.scatter(flagged, y[flagged], marker="x", c="black", label="Flagged spike")
            for index, c in enumerate(truth):
                ax.axvline(c, color="green", linestyle="--", label="True change" if index == 0 else None)
            for index, c in enumerate(detected):
                ax.axvline(c, color="red", linestyle=":", label="Detected change" if index == 0 else None)
            ax.set(xlabel="Sample", ylabel="Sensor reading")
            ax.legend(ncol=2)
            fig.savefig(out / "example.png", dpi=140)
            plt.close(fig)
    report = dict(backend=backend, seeds=list(range(21, 41)), samples=14000,
                  offline_changes=metrics(len(errors), predicted, 60),
                  matched_change_mae_samples=float(np.mean(errors)) if errors else None,
                  point_anomalies=metrics(spike_hits, spike_predictions, 200),
                  first_change_cusum=dict(on_time=len(delays), early=early, missed=missed,
                                          median_delay_samples=float(np.median(delays)) if delays else None),
                  pipeline_seconds=elapsed)
    output = json.dumps(report, indent=2)
    (out / "metrics.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--backend", choices=["exact", "pelt"], default="exact")
    args = parser.parse_args()
    run(args.out, args.backend)

