"""Mode-field normalization, coherent addition, and circuit power exchange."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import esign, eta0
from solver.cavity_fields import CavityFields
from solver.cavity_state import CavityState
from solver.beam_state import BeamState
from diagnostics.cavity import calculate_cavity_power
from test_large_signal_1d import inputs


class CavityModeFieldTests(unittest.TestCase):
    def test_harmonics_phase_scaling_and_fresh_arrays(self):
        numerical = SimpleNamespace(Nc=4, Ns=2)
        grid = SimpleNamespace(z=np.arange(3.), betae=2., k0=np.array([1., 4., 1., 1.]))
        cavities = SimpleNamespace(type=np.array([0, 1, -1, 1]), nh=np.array([1, 2, 1, 3]))
        ez = np.array([[1., 4., 2., 99.], [2., 0., 1., 99.], [0., 2., 3., 99.]])
        fields = CavityFields(ez=ez.copy(), C3=ez.copy(), vpz=ez*0, alfaz=ez*0)
        state = SimpleNamespace(Vnc=np.array([2j, 1-1j, 1., 100.]))
        Emz, Emz0 = fields.calculate_mode_fields(numerical, cavities, grid, state)
        expected = np.array([[1+1j, 1-1j], [0.5+2j, 0], [1.5, 0.5-0.5j]])
        np.testing.assert_allclose(Emz, expected)
        np.testing.assert_array_equal(Emz0, 0)
        # Multiplication by 2j doubles amplitudes and rotates phase by pi/2.
        state.Vnc *= 2j
        following, _ = fields.calculate_mode_fields(numerical, cavities, grid, state)
        np.testing.assert_allclose(following, 2j * expected)
        np.testing.assert_array_equal(fields.ez, ez)
        np.testing.assert_allclose(Emz, expected)  # No mutation/accumulation.
        self.assertFalse(np.shares_memory(Emz, following))

    def test_physical_field_integrates_to_gap_voltage(self):
        numerical, beam, cavities, grid, reference, fields = inputs()
        numerical = replace(numerical, Ns=2)
        cavities = replace(cavities, nh=np.array([1, 2]), type=np.array([1, 1]))
        state = CavityState.initialize(numerical, cavities, beam, reference)
        state.Vnc[:] = [0.12+0.04j, -0.03+0.07j]
        Emz, _ = fields.calculate_mode_fields(numerical, cavities, grid, state)
        physical = reference.U0 * grid.betae * Emz  # kV/mm.
        gap = np.trapezoid(physical, x=grid.zi, axis=0)
        power = calculate_cavity_power(cavities, fields, state, beam, grid, reference)
        np.testing.assert_allclose(gap, power["Vng"], rtol=1e-13, atol=1e-14)
        from_C3 = fields.C3 * state.Vnc * reference.U0 * 1000 / eta0 / reference.I0
        np.testing.assert_allclose(Emz, from_C3, rtol=1e-13)

    def test_beam_work_matches_passive_cavity_wall_loss(self):
        numerical, beam, cavities, grid, reference, fields = inputs(modulation=0.3)
        # Passive, uncoupled bunching modes have only intrinsic losses. The
        # field should extract exactly their wall-loss power from the source
        # current used in the circuit, including detuning and complex phase.
        cavities = replace(cavities, type=np.ones(2, dtype=int), Pin=np.zeros(2))
        beam_state = BeamState.initialize(numerical, beam, grid, reference)
        beam_state.Iz *= np.exp(-1j * (grid.z[:, None] - grid.z[0]) * np.arange(1, 3))
        state = CavityState.initialize(numerical, cavities, beam, reference)
        state.solve(numerical, cavities, fields, beam, beam_state, grid)
        Emz, _ = fields.calculate_mode_fields(numerical, cavities, grid, state)
        work = esign * 0.5 * np.trapezoid(
            np.sum(np.real(np.conj(beam_state.Iz) * Emz), axis=1), x=grid.z,
        )
        power = calculate_cavity_power(cavities, fields, state, beam, grid, reference)
        self.assertGreater(work, 0)
        np.testing.assert_allclose(work, np.sum(power["effloss"]), rtol=1e-12)

    def test_dc_and_traveling_wave_are_not_silently_ignored(self):
        numerical, beam, cavities, grid, reference, fields = inputs()
        state = CavityState.initialize(numerical, cavities, beam, reference)
        for changed in [replace(cavities, nh=np.array([0, 1])),
                        replace(cavities, type=np.array([0, 2]))]:
            with self.assertRaises(NotImplementedError):
                fields.calculate_mode_fields(numerical, changed, grid, state)


if __name__ == "__main__":
    unittest.main()
