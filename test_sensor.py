import itertools
import unittest

import numpy as np

from sensor import cusum, segment, spikes


class SensorTests(unittest.TestCase):
    def test_cusum_detects_both_directions_and_quiet_reference(self):
        for direction in (-1, 1):
            self.assertEqual(cusum(np.r_[np.zeros(20), np.full(20, 3 * direction)], 0, 1), 23)
        self.assertIsNone(cusum(np.zeros(60), 0, 1))

    def test_exact_matches_brute_force(self):
        y = np.array([0., 1., 0., 4., 4., 5., -1., 0.])
        def objective(changes):
            groups = np.split(y, changes)
            return sum(((group - group.mean())**2).sum() for group in groups) + 2 * len(changes)
        feasible = [changes for k in range(4)
                    for changes in itertools.combinations(range(2, len(y)-1), k)
                    if min(np.diff((0, *changes, len(y)))) >= 2]
        self.assertEqual(segment(y, penalty=2, min_size=2), [3, 6])
        self.assertAlmostEqual(objective(segment(y, 2, 2)), min(map(objective, feasible)))

    def test_spike_and_constant_signal(self):
        y = np.zeros(40)
        y[20] = 8
        np.testing.assert_array_equal(spikes(y), [20])
        self.assertEqual(segment(np.zeros(60)), [])

    def test_invalid_signal(self):
        with self.assertRaises(ValueError):
            spikes([1, np.nan])

