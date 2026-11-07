# Change-point and anomaly detection on sensor time series

This interview-preparation project compares online first-alarm detection with offline segmentation of persistent mean shifts. A separate robust local detector flags isolated spikes.

## Run

From this repository, using Python 3.10 or later:

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python demo.py
```

The command writes a metrics JSON, an annotated plot, and an example CSV with observations, cleaned values, and labels. `benchmark/` contains a checked-in run. Reuse `cusum`, `segment` or `spikes` from `sensor.py` with finite, one-dimensional readings.

The optional `ruptures` backend targets the same squared-error objective with PELT pruning. It is wired up but was unavailable for local verification. With `ruptures` installed, run `python demo.py --backend pelt --out results-pelt` to evaluate that backend separately.

## Under the hood

- **CUSUM** standardizes readings using the first 60 samples. It maintains positive and negative sums, accumulating standardized residuals beyond a drift allowance and flooring each sum at zero. Crossing a threshold reports an alarm time. Increasing the threshold usually reduces false alarms at the cost of longer delay. This example stops at the first alarm.
- **Offline segmentation** minimizes total within-segment squared error plus a fixed cost for each breakpoint. Prefix sums compute the cost for any interval in constant time. Dynamic programming evaluates every legal last breakpoint, stores the lowest cost for each prefix, and backtracks to recover the segmentation. Minimum length is 25, penalty is 40, and the exact solver takes quadratic time and linear auxiliary memory. PELT prunes candidate starts that cannot improve a later solution.
- **Point anomalies** use the median and scaled median absolute deviation of a centered window that excludes the point under test. A positive scale floor prevents division by nearly zero. Spikes flagged by this offline step are replaced with a neighbor median for segmentation. CUSUM sees raw values. A centered window needs future samples and skips the first and last seven positions.

## Benchmark protocol

Seeds 21–40 generate 20 independent 700-sample signals, each with three fixed mean shifts and ten randomly placed spikes. Spikes are kept away from true transitions and the initial calibration period. Offline change points count as true positives when assigned one-to-one within eight samples. Spikes require exact index matches. First-alarm CUSUM reports early alarms, alarms between the first and second changes, misses, and median delay for alarms in that interval. Pipeline runtime includes detection and segmentation and excludes data generation, scoring, and plotting.

Parameters and benchmark were developed together. Results apply to this synthetic setup. Autocorrelated noise, gradual drift, variance changes, spikes near transitions, or a contaminated calibration window could change performance.

[PELT implementation](https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/pelt.py)

