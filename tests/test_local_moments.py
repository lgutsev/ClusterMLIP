import unittest

import numpy as np

from cluster_mlip.local_moments import canonicalize, set_local_moments


class CanonicalizeTests(unittest.TestCase):
    def test_positive_sum_unchanged_and_values_kept(self):
        m = [3.8, 3.7, -5.611729, 4.17]
        np.testing.assert_array_equal(canonicalize(m), np.array(m))

    def test_reversed_vector_maps_to_the_same_input(self):
        m = np.array([3.8, 3.7, -5.611729, 4.17])
        np.testing.assert_array_equal(canonicalize(-m), canonicalize(m))

    def test_zero_sum_is_refused(self):
        with self.assertRaises(ValueError):
            canonicalize([1.0, -1.0])


class SetLocalMomentsTests(unittest.TestCase):
    def test_sets_array_and_magmoms(self):
        try:
            from ase import Atoms
        except ImportError:  # base install without ase
            self.skipTest("ase not installed")
        a = Atoms("Fe3", positions=[[0, 0, 0], [2.4, 0, 0], [0, 2.4, 0]])
        set_local_moments(a, [-3.0, -3.0, 2.0])
        np.testing.assert_array_equal(a.arrays["local_moment"], [3.0, 3.0, -2.0])
        np.testing.assert_array_equal(a.get_initial_magnetic_moments(), [3.0, 3.0, -2.0])
        with self.assertRaises(ValueError):
            set_local_moments(a, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
