"""Analytic checks for space-charge work and the final RF power balance."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import esign
from diagnostics.space_charge import calculate_space_charge_power
from diagnostics.power_balance import calculate_power_balance


class SpaceChargePowerTests(unittest.TestCase):
    def setUp(self):
        # Physical intervals are 2 mm; normalized intervals are 0.5.
        self.grid = SimpleNamespace(
            zi=np.array([0., 2., 4., 6.]),
            z=np.array([0., 0.5, 1., 1.5]), betae=0.25, dz=2.,
        )
        self.beam_state = SimpleNamespace(
            IzDC=np.full(4, 2.), Iz=np.ones((4, 3), dtype=complex),
        )
        self.space_charge_state = SimpleNamespace(
            Esz0=np.zeros(4, dtype=complex), Esz=np.zeros((4, 3), dtype=complex),
        )

    def calculate(self):
        return calculate_space_charge_power(
            self.beam_state, self.space_charge_state, self.grid,
        )

    def test_zero_fields_give_no_energy_exchange(self):
        result = self.calculate()
        for value in result.values():
            np.testing.assert_array_equal(value, np.zeros(4))

    def test_constant_dc_exchange_and_unchanged_inputs(self):
        self.space_charge_state.Esz0[:] = -0.3
        old_current = self.beam_state.IzDC.copy()
        old_field = self.space_charge_state.Esz0.copy()
        result = self.calculate()
        # Two units of current times -0.3 field over normalized distance z.
        np.testing.assert_allclose(result["effSC"], -0.6 * esign * self.grid.z)
        np.testing.assert_array_equal(self.beam_state.IzDC, old_current)
        np.testing.assert_array_equal(self.space_charge_state.Esz0, old_field)

    def test_left_endpoint_rule_and_complex_dc_are_preserved(self):
        self.beam_state.IzDC[:] = 1
        self.space_charge_state.Esz0[:] = [1 + 0.1j, 3, -2, 1000]
        result = self.calculate()
        # Final sample is not integrated; each interval uses its left sample.
        expected = esign * np.array([0, 0.5 + 0.05j, 2 + 0.05j, 1 + 0.05j])
        np.testing.assert_allclose(result["effSCDC"], expected)

    def test_rf_conjugation_phase_and_harmonic_sum(self):
        self.beam_state.Iz[:] = [2j, 1, 1]
        # First harmonic: aligned phases, mean product 4.
        # Second: quadrature, zero mean. Third: opposite phases, mean -1.
        self.space_charge_state.Esz[:] = [4j, 3j, -2]
        result = self.calculate()
        np.testing.assert_allclose(result["effSCRF"], 3 * esign * self.grid.z)
        np.testing.assert_allclose(result["effSC"], result["effSCRF"])


class PowerBalanceTests(unittest.TestCase):
    def setUp(self):
        self.grid = SimpleNamespace(zi=np.array([0., 2., 4.]))
        self.numerical = SimpleNamespace(efficiency_end_offset_mm=1.)
        self.beam = SimpleNamespace(Nbeam=3)
        self.reference = SimpleNamespace(U0=10., I0=2.)
        self.beam_state = SimpleNamespace(effe=np.array([0., 0.2, 0.4]))
        self.space_charge_power = {"effSC": np.array([0., 0.04, 0.08], dtype=complex)}
        self.cavity_power = {"effloss": np.array([0.01, 0.02]), "RFinput": 600.}

    def calculate(self):
        return calculate_power_balance(
            self.beam_state, self.space_charge_power, self.cavity_power,
            self.numerical, self.beam, self.grid, self.reference,
        )

    def test_interpolation_and_power_balance_in_watts(self):
        result = self.calculate()
        # At 3 mm: 18 kW beam power reduction, 3.6 kW space-charge
        # contribution, 1.8 kW wall losses and 0.6 kW net RF input.
        self.assertAlmostEqual(result["effend"], 0.3)
        self.assertAlmostEqual(result["effSCend"], 0.06)
        self.assertAlmostEqual(result["effendmodified"], 13200 / 60000)

    def test_zero_offset_samples_the_exit(self):
        self.numerical.efficiency_end_offset_mm = 0
        result = self.calculate()
        self.assertAlmostEqual(result["effend"], 0.4)
        self.assertAlmostEqual(result["effSCend"], 0.08)
        self.assertAlmostEqual(result["effendmodified"], 0.3)

    def test_outside_domain_returns_nan_without_clamping(self):
        for offset in (-1., 5.):
            with self.subTest(offset=offset):
                self.numerical.efficiency_end_offset_mm = offset
                result = self.calculate()
                for value in result.values():
                    self.assertTrue(np.isnan(value))


if __name__ == "__main__":
    unittest.main()
