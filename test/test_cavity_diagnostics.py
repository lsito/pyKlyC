"""Physical-unit checks for MATLAB's final cavity power accounting."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import eta0
from diagnostics.cavity import calculate_cavity_power


class CavityPowerTests(unittest.TestCase):
    def setUp(self):
        self.cavities = SimpleNamespace(
            type=np.array([0, 0, -1, -1, 1, 1]),
            nh=np.array([1, 1, 1, 1, 1, 0]),
            Qin=np.full(6, 100.0),
            gaman=np.array([0, 0, 0, 0.5j, 0, 0]),
            zshift=np.zeros(6),
        )
        self.reference = SimpleNamespace(U0=10.0, I0=2.0)
        self.beam = SimpleNamespace(Nbeam=3)
        self.grid = SimpleNamespace(zi=np.array([-1., 0., 1.]), k0=np.full(6, 4.0))
        # Triangular profiles have integral 2; dividing by sqrt(k0) gives 1.
        self.fields = SimpleNamespace(ez=np.tile([[0.], [2.], [0.]], (1, 6)))
        self.state = SimpleNamespace(
            Pin=np.array([100., 200., 0., 0., 0., 0.]),
            Vin=np.zeros(6), Vnp=np.zeros(6, dtype=complex),
            Vnc=np.full(6, 0.02 + 0.01j),
        )
        self.state.Vin = np.sqrt(2 * eta0 * self.state.Pin) / 10000
        # Half-voltage drive: incident powers are 25 W and 50 W.
        self.state.Vnp[:2] = self.state.Vin[:2] * 0.5 * np.array([0.2, 0.3j])
        # Outgoing powers of 1000 W and 2000 W before load reflection.
        self.state.Vnp[2:4] = np.sqrt(2 * eta0 * np.array([1000., 2000.])) / 10000

    def calculate(self, drivefactor=0.5):
        return calculate_cavity_power(
            self.cavities, self.fields, self.state, self.beam,
            self.grid, self.reference, drivefactor,
        )

    def test_power_in_watts_and_last_cavity_scalars(self):
        result = self.calculate()
        self.assertAlmostEqual(result["RFinput"], 25 * 0.96 + 50 * 0.91)
        self.assertAlmostEqual(result["refcavity"], 0.3j)
        # Total beam power is 3 * 10 kV * 2 A = 60000 W.
        self.assertAlmostEqual(result["effcavity"], 2000 * 0.75 / 60000)
        # For the same outgoing waves, full drive changes only incident power.
        full = self.calculate(drivefactor=1.0)
        self.assertAlmostEqual(full["RFinput"] - result["RFinput"], 225.0)
        self.assertAlmostEqual(full["refcavity"], 0.15j)

    def test_gap_phase_wall_loss_and_spatial_steps(self):
        old_voltage = self.state.Vnc.copy()
        old_fields = self.fields.ez.copy()
        # DC gap voltage must be independent of its field-map integral and k0.
        self.fields.ez[:, -1] = 0
        self.grid.k0[-1] = 0
        result = self.calculate()
        np.testing.assert_allclose(result["Vng"], np.full(6, 0.2 + 0.1j))
        # Physical mode amplitude is (200 + 100j) V; divide its squared
        # magnitude by 2 * eta0 * Qin to obtain wall dissipation in W.
        wall_fraction = 50000 / (2 * eta0 * 100) / 60000
        np.testing.assert_allclose(result["effloss"], [wall_fraction] * 5 + [0])
        np.testing.assert_allclose(result["lossacc"],
                                   [0, 2.5 * wall_fraction, 5 * wall_fraction])
        np.testing.assert_array_equal(self.state.Vnc, old_voltage)
        np.testing.assert_array_equal(self.fields.ez[:, :-1], old_fields[:, :-1])
        np.testing.assert_array_equal(self.fields.ez[:, -1], 0)

    def test_no_ports_leave_scalar_results_zero(self):
        self.cavities.type[:] = 1
        result = self.calculate()
        self.assertEqual(result["refcavity"], 0)
        self.assertEqual(result["effcavity"], 0)
        self.assertEqual(result["RFinput"], 0)

    def test_traveling_wave_requires_its_own_model(self):
        self.cavities.type[0] = 2
        with self.assertRaises(NotImplementedError):
            self.calculate()


if __name__ == "__main__":
    unittest.main()
