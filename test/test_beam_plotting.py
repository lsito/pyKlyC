"""Check plot data against simple trajectories and MATLAB conventions."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plotting.beam import plot_applegate, plot_current_harmonics


class BeamPlottingTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(zi=np.array([-2., 0., 2.]),
                                    z=np.array([-1., 0., 1.]), betae=0.5)
        self.beam = SimpleNamespace(Nt=5, tn1=np.arange(5) * 0.2)
        self.cavities = SimpleNamespace(nh=np.array([1, 0, 2]), zshift=np.array([0., 1., 2.]))
        self.fields = SimpleNamespace(ez=np.array([[1., 99., -1.], [2., 99., 1.], [0., 99., 1.]]))
        Zn = np.vstack([np.tile(self.grid.z[:, None], (1, 5)), np.full((1, 5), np.inf)])
        Tn = np.zeros_like(Zn)
        Tn[:3] = self.beam.tn1 + self.grid.z[:, None] - self.grid.z[0]
        self.dynamics = SimpleNamespace(Zn=Zn, Tn=Tn, Y=np.array([3, 0, 1]))

    def tearDown(self):
        plt.close("all")

    def test_drift_sampling_background_and_input_preservation(self):
        old_fields = self.fields.ez.copy()
        # Nt/Ntp=2.5 rounds UP to 3 in MATLAB: select particle columns 0, 3.
        with np.errstate(invalid="raise"):
            fig, (ax, ax_field) = plot_applegate(
                self.dynamics, self.beam, self.grid, self.cavities, self.fields,
                ncenter=0, Ntp=2, zmax=1.5,
            )
        self.assertEqual([line.get_label() for line in ax.lines], ["particle 0", "particle 3"])
        for line, phase in zip(ax.lines, [1., 1.6]):
            np.testing.assert_allclose(line.get_xdata(), [-2, 0, 2])
            np.testing.assert_allclose(line.get_ydata(), phase)
        np.testing.assert_allclose(ax_field.lines[0].get_ydata(), [0, 3, 1])
        np.testing.assert_array_equal(self.fields.ez, old_fields)
        np.testing.assert_allclose(ax.get_xlim(), [-2, 1.5])

    def test_phase_wrapping_and_all_particles_default(self):
        # Relative to center=1: +pi maps to -pi; whole periods disappear.
        self.dynamics.Tn[:3, 0] = self.grid.z + 1 + np.array([np.pi, 2*np.pi + 0.1, -2*np.pi - 0.2])
        self.fields.ez[:] = 0
        fig, (ax, ax_field) = plot_applegate(
            self.dynamics, self.beam, self.grid, self.cavities, self.fields, ncenter=0,
        )
        self.assertEqual(len(ax.lines), 3)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [1-np.pi, 1.1, 0.8])
        self.assertLess(ax_field.get_ylim()[0], ax_field.get_ylim()[1])

    def test_harmonics_are_magnitudes_and_markers_match_matlab(self):
        state = SimpleNamespace(Iz=np.array([[3+4j, -2j], [-5, 1], [0, 0]], complex))
        old_current = state.Iz.copy()
        fig, ax = plot_current_harmonics(state, self.grid, self.cavities)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [5, 5, 0])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [2, 1, 0])
        self.assertEqual([line.get_linewidth() for line in ax.lines[:2]], [4, 2])
        for line, zshift in zip(ax.lines[2:], self.cavities.zshift):
            np.testing.assert_array_equal(line.get_xdata(), [zshift, zshift])
            np.testing.assert_array_equal(line.get_ydata(), [0, 2])
        self.assertEqual(len(ax.get_legend().get_texts()), 2)
        np.testing.assert_array_equal(state.Iz, old_current)


if __name__ == "__main__":
    unittest.main()
