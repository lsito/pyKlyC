"""Checks for final trajectory and fundamental current diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diagnostics.beam import (
    check_boundary_reaching, check_negative_velocity, find_peak_modulation,
)


def histories():
    # Two active samples and one inactive column. Rows after an exit stay inf.
    return SimpleNamespace(
        Zn=np.array([[0., 0., 0.], [1., 1., np.inf],
                     [2., np.inf, np.inf], [np.inf, np.inf, np.inf]]),
        Vn=np.array([[1., .8, np.inf], [.9, .7, np.inf],
                     [.8, np.inf, np.inf], [np.inf, np.inf, np.inf]]),
        Y=np.array([0, 1]), Yflag=2,
    )


class TrajectoryDiagnosticsTests(unittest.TestCase):
    def test_last_row_uses_active_samples_and_matlab_numbering(self):
        dynamics = histories()
        dynamics.Zn[-1, 2] = 20.  # Inactive column must not set n.
        old_z, old_v = dynamics.Zn.copy(), dynamics.Vn.copy()
        result = check_boundary_reaching(dynamics)
        velocity = check_negative_velocity(dynamics, SimpleNamespace(s=.2))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["status"], "Normal")
        self.assertAlmostEqual(velocity["minimum_v_over_c"], .14)
        self.assertFalse(velocity["negative_velocity"])
        np.testing.assert_array_equal(dynamics.Zn, old_z)
        np.testing.assert_array_equal(dynamics.Vn, old_v)

    def test_storage_limit_has_priority_even_when_all_exits_are_counted(self):
        for Yflag in (1, 2):
            with self.subTest(Yflag=Yflag):
                dynamics = histories()
                dynamics.Zn[-1, 0] = 2.5
                dynamics.Vn[-1, 0] = .6
                dynamics.Yflag = Yflag
                result = check_boundary_reaching(dynamics)
                self.assertEqual(result["n"], 4)
                self.assertEqual(result["status"], "Storage limit")

    def test_missing_exit_does_not_require_recorded_negative_velocity(self):
        dynamics = histories()
        dynamics.Yflag = 1
        result = check_boundary_reaching(dynamics)
        velocity = check_negative_velocity(dynamics, SimpleNamespace(s=.2))
        self.assertEqual(result["status"], "Reflected electrons")
        self.assertFalse(velocity["negative_velocity"])

    def test_backward_motion_can_precede_a_successful_downstream_exit(self):
        dynamics = histories()
        dynamics.Vn[1, 0] = -.25
        result = check_boundary_reaching(dynamics)
        velocity = check_negative_velocity(dynamics, SimpleNamespace(s=.2))
        self.assertEqual(result["status"], "Normal")
        self.assertTrue(velocity["negative_velocity"])
        self.assertAlmostEqual(velocity["minimum_v_over_c"], -.05)

    def test_minimum_omits_nan_like_matlab(self):
        dynamics = histories()
        dynamics.Vn[0, 0] = np.nan
        result = check_negative_velocity(dynamics, SimpleNamespace(s=.2))
        self.assertAlmostEqual(result["minimum_v_over_c"], .14)


class PeakModulationTests(unittest.TestCase):
    def test_phase_wrap_and_active_particle_column(self):
        # The first harmonic peaks at plane 1, with target phase zero.
        # A larger second harmonic elsewhere must not affect the selection.
        beam_state = SimpleNamespace(Iz=np.array([[0.2, 100], [2j, 0], [1, 0]]))
        dynamics = SimpleNamespace(
            Tnz=np.zeros((3, 6)), Y=np.array([4, 1, 5]),
        )
        dynamics.Tnz[1, [4, 1, 5]] = [1.0, 2 * np.pi - 0.02, 0.3]
        # Inactive columns have phase zero but must not be selected.
        old_Iz = beam_state.Iz.copy()
        old_Tnz = dynamics.Tnz.copy()
        result = find_peak_modulation(beam_state, dynamics)
        self.assertEqual(result, {"Izmax": 2.0, "ntemp": 1, "ncenter": 1})
        np.testing.assert_array_equal(beam_state.Iz, old_Iz)
        np.testing.assert_array_equal(dynamics.Tnz, old_Tnz)

    def test_first_maximum_and_first_particle_in_a_tie(self):
        beam_state = SimpleNamespace(Iz=np.array([[1.0], [1.0], [0.2]]))
        dynamics = SimpleNamespace(
            Tnz=np.full((3, 5), np.pi / 2), Y=np.array([3, 1]),
        )
        result = find_peak_modulation(beam_state, dynamics)
        self.assertEqual(result, {"Izmax": 1.0, "ntemp": 0, "ncenter": 3})


if __name__ == "__main__":
    unittest.main()
