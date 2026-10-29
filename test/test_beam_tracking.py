"""Small regression checks for tracking orchestration and MATLAB 516--530.

Run from pyKlyC: python -B -m unittest discover -s test -p test_beam_tracking.py
These checks do not overwrite the notebook/MATLAB comparison results.
"""

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.settings import NumericalSettings
from constants import c, etom
from solver.beam_dynamics import BeamDynamics
from solver.beam_state import BeamState


def inputs(Ns=3, iterfactor=0.5, threshold=0.005, Ntimes=10):
    numerical = NumericalSettings(
        f=1.0, Nc=1, Ns=Ns, Nz=128, Nlimit=10, Ntimes=Ntimes,
        Rlimit=1e-4, iterfactor=iterfactor,
        evolutionmethod=np.array([threshold, 2.0]),
        upstream_discard_distance_mm=0.0,
        efficiency_end_offset_mm=0.0,
    )
    t0 = np.arange(4) * np.pi / 2
    beam = SimpleNamespace(
        Nt=4, t0=t0, tn1=t0 + 0.13, Ut=np.full(4, 20.0),
        Izt=1.0 + 0.2 * np.cos(t0), Y0=np.arange(4),
    )
    grid = SimpleNamespace(z=np.arange(25) * 0.125,
                           zi=np.arange(25) * 0.125, dt=0.125)
    reference = SimpleNamespace(s=0.2, Iz0=1.0, vn1=np.full(4, 1.0 / 0.91))
    return numerical, beam, grid, reference


class ChargeReconstructionTests(unittest.TestCase):
    def test_linear_harmonics_and_unchanged_dc(self):
        for Ns in [1, 3]:
            with self.subTest(Ns=Ns):
                numerical, beam, grid, reference = inputs(Ns=Ns)
                dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
                kk = np.arange(1, Ns + 1)
                slopes = (0.4 + 0.7j) * kk**2
                dynamics.Izr = 2.0j + grid.z.reshape(-1, 1) * slopes
                old_current = dynamics.Izr.copy()
                old_dc = dynamics.QQDCr.copy()
                dynamics.calculate_QQr(numerical, grid)
                expected = np.tile(-slopes / (1.0j * kk), (len(grid.z), 1))
                np.testing.assert_allclose(dynamics.QQr, expected, atol=1e-13)
                np.testing.assert_array_equal(dynamics.Izr, old_current)
                np.testing.assert_array_equal(dynamics.QQDCr, old_dc)
                self.assertEqual(dynamics.QQr.shape, (len(grid.z), Ns))

    def test_matlab_gradient_stencil(self):
        for Ns in [1, 4]:
            for N in [2, 25]:
                with self.subTest(Ns=Ns, N=N):
                    numerical, beam, grid, reference = inputs(Ns=Ns)
                    grid.z = grid.z[:N]
                    dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
                    rng = np.random.default_rng(42)
                    dynamics.Izr = rng.normal(size=(N, Ns)) + 1j * rng.normal(size=(N, Ns))
                    # MATLAB gradient: central interior and one-sided endpoints.
                    Fy = np.empty_like(dynamics.Izr)
                    Fy[0] = (dynamics.Izr[1] - dynamics.Izr[0]) / grid.dt
                    Fy[-1] = (dynamics.Izr[-1] - dynamics.Izr[-2]) / grid.dt
                    Fy[1:-1] = (dynamics.Izr[2:] - dynamics.Izr[:-2]) / (2 * grid.dt)
                    expected = -Fy / (1j * np.arange(1, Ns + 1))
                    dynamics.calculate_QQr(numerical, grid)
                    np.testing.assert_allclose(dynamics.QQr, expected, atol=1e-13)

    def test_constant_current_has_zero_charge_harmonics(self):
        numerical, beam, grid, reference = inputs()
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        dynamics.Izr[:] = 0.3 + 0.2j
        dynamics.calculate_QQr(numerical, grid)
        np.testing.assert_array_equal(dynamics.QQr, np.zeros_like(dynamics.QQr))


class TrackingCoordinatorTests(unittest.TestCase):
    def test_matches_explicit_tracking_sequence(self):
        cases = [
            ("spatial_only", 0.0, 0.0, 0.005, 0.5, 1, 10, False),
            ("mixed_rf_dc", 0.0001, 0.0002, 0.005, 0.5, 1, 10, False),
            ("time_handoff", 0.0, 0.2, -1.0, 0.5, 1, 10, True),
            ("reflection", 3.0, 0.0, 0.005, 0.5, 1, 10, True),
            ("time_limit", 0.0, 0.0, -1.0, 0.5, 1, 1, True),
            ("field_correction", 0.0, 0.2, -1.0, -0.5, 6, 10, True),
        ]
        for name, dc, rf, threshold, iterfactor, nn, Ntimes, handoff in cases:
            with self.subTest(case=name):
                numerical, beam, grid, reference = inputs(
                    threshold=threshold, iterfactor=iterfactor, Ntimes=Ntimes,
                )
                expected = BeamDynamics.initialize(numerical, beam, grid, reference)
                actual = BeamDynamics.initialize(numerical, beam, grid, reference)
                Emz = np.full((len(grid.z), numerical.Ns), rf * (1 + 0.2j))
                Esz = 0.2 * Emz
                Esz0 = np.full(len(grid.z), 0.25 * dc)
                Emz0 = np.full(len(grid.z), 0.75 * dc)
                expected_dc = Esz0 + Emz0
                Iz = np.full_like(Emz, 0.3 + 0.2j)
                expected_field = Esz + Emz
                source_fields = [Esz, Esz0, Emz, Emz0]
                originals = [source.copy() for source in source_fields]
                entrance_current = actual.Izr[0].copy()
                kzstart = expected.track_z(
                    numerical, beam, grid, reference, expected_field, expected_dc,
                    Esz=Esz, Emz=Emz, nn=nn, iterf=np.sign(iterfactor), Iz=Iz,
                )
                self.assertEqual(kzstart is not None, handoff)
                if kzstart is not None:
                    expected.track_t(numerical, beam, grid, reference,
                                     expected_field, expected_dc, kzstart)
                expected.calculate_QQr(numerical, grid)
                with patch.object(actual, "track_t", wraps=actual.track_t) as continuation:
                    actual_field, actual_dc = actual.track(
                        numerical, beam, grid, reference,
                        Esz=Esz, Esz0=Esz0, Emz=Emz, Emz0=Emz0, nn=nn, Iz=Iz,
                    )
                self.assertEqual(continuation.call_count, int(handoff))
                for field in fields(BeamDynamics):
                    np.testing.assert_array_equal(getattr(actual, field.name),
                                                  getattr(expected, field.name))
                np.testing.assert_array_equal(actual_field, expected_field)
                np.testing.assert_array_equal(actual_dc, expected_dc)
                for source, original in zip(source_fields, originals):
                    np.testing.assert_array_equal(source, original)
                    self.assertFalse(np.shares_memory(actual_field, source))
                    self.assertFalse(np.shares_memory(actual_dc, source))
                if name == "field_correction":
                    np.testing.assert_allclose(actual_field[0], Esz[0] * entrance_current / Iz[0] + Emz[0])
                    # Handoff after the first push: time tracking still needs
                    # all remaining planes populated with the ordinary sum.
                    np.testing.assert_array_equal(actual_field[1:], Esz[1:] + Emz[1:])
                else:
                    np.testing.assert_array_equal(actual_field, Esz + Emz)
                if name in ["reflection", "time_limit"]:
                    self.assertLess(actual.Yflag, beam.Nt)


class BeamRelaxationTests(unittest.TestCase):
    def test_weights_and_independent_arrays(self):
        pairs = [("IzDC", "IzDCr"), ("Iz", "Izr"), ("QQDC", "QQDCr"), ("QQ", "QQr")]
        for weight in [0.0, 0.3, 1.0, -0.3]:
            with self.subTest(iterfactor=weight):
                numerical, beam, grid, reference = inputs(iterfactor=weight)
                state = BeamState.initialize(numerical, beam, grid, reference)
                dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
                for index, (name, new_name) in enumerate(pairs):
                    getattr(dynamics, new_name)[:] = index + 0.6
                    if name in ["Iz", "QQ"]:
                        getattr(dynamics, new_name)[:] += 0.7j
                previous = {name: getattr(state, name).copy() for name, _ in pairs}
                tracked = {name: getattr(dynamics, name).copy() for _, name in pairs}
                state.update(numerical, beam, reference, dynamics)
                for name, new_name in pairs:
                    expected = (1 - abs(weight)) * previous[name] + abs(weight) * tracked[new_name]
                    np.testing.assert_array_equal(getattr(state, name), expected)
                    np.testing.assert_array_equal(getattr(dynamics, new_name), tracked[new_name])
                    self.assertFalse(np.shares_memory(getattr(state, name), getattr(dynamics, new_name)))


class DerivedBeamProfileTests(unittest.TestCase):
    def test_field_free_tracking_preserves_energy_and_speed(self):
        numerical, beam, grid, reference = inputs()
        # Make the entrance voltage consistent with the prescribed velocity.
        gamma = (1.0 - (reference.s / reference.vn1[0]) ** 2) ** -0.5
        beam.Ut[:] = c**2 / etom * 1e-3 * (gamma - 1)
        state = BeamState.initialize(numerical, beam, grid, reference)
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        field = np.zeros((len(grid.z), numerical.Ns), dtype=complex)
        dynamics.track(numerical, beam, grid, reference,
                       Esz=field, Esz0=np.zeros(len(grid.z)),
                       Emz=field, Emz0=np.zeros(len(grid.z)))
        state.update(numerical, beam, reference, dynamics)
        np.testing.assert_allclose(state.effe, 0.0, atol=1e-14)
        np.testing.assert_allclose(state.vez, 1.0 / reference.vn1[0], atol=1e-14)
        np.testing.assert_allclose(state.gamaz, gamma, atol=1e-14)

    def test_energy_loss_and_partial_weighted_entrance(self):
        numerical, beam, grid, reference = inputs(iterfactor=1.0)
        beam.Y0 = np.array([0, 2])
        beam.Izt = np.array([-2.0, -100.0, -5.0, -200.0])
        beam.Ut = np.array([100.0, 900.0, 180.0, 800.0])
        reference.Iz0 = np.sum(beam.Izt[beam.Y0]) / beam.Nt
        state = BeamState.initialize(numerical, beam, grid, reference)
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        # Active-sample current-weighted entrance voltage, not the simple mean.
        U0 = (2.0 * 100.0 + 5.0 * 180.0) / 7.0
        fraction = np.linspace(1.0, 0.4, len(grid.z))
        gamma = 1.0 + etom * 1e3 / c**2 * U0 * fraction
        dynamics.IzDCr[:] = 1.0
        dynamics.Pe[:] = 2 * np.pi * (gamma - 1.0)
        old_pe = dynamics.Pe.copy()
        state.update(numerical, beam, reference, dynamics)
        np.testing.assert_allclose(state.effe, 1 - fraction, atol=1e-14)
        np.testing.assert_allclose(state.gamaz, gamma, atol=1e-14)
        np.testing.assert_allclose(state.vez, np.sqrt(1 - gamma**-2) / reference.s)
        np.testing.assert_array_equal(dynamics.Pe, old_pe)

    def test_raw_energy_with_relaxed_current_including_zero_weight(self):
        for weight in [0.0, 0.3, -0.3, 1.0]:
            with self.subTest(iterfactor=weight):
                numerical, beam, grid, reference = inputs(iterfactor=weight)
                state = BeamState.initialize(numerical, beam, grid, reference)
                dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
                state.IzDC[:] = 0.8
                dynamics.IzDCr[:] = 0.4
                dynamics.Pe[:] = 2 * np.pi * np.linspace(0.01, 0.1, len(grid.z))
                relaxed_current = (1 - abs(weight)) * 0.8 + abs(weight) * 0.4
                gamma = 1 + dynamics.Pe / (2 * np.pi * relaxed_current)
                state.update(numerical, beam, reference, dynamics)
                np.testing.assert_allclose(state.gamaz, gamma, atol=1e-14)
                np.testing.assert_allclose(state.vez, np.sqrt(1 - gamma**-2) / reference.s)
                # With mean entrance current/reference current = 1, denominator
                # is 2*pi*20 kV. Derived profiles are not relaxed themselves.
                np.testing.assert_allclose(state.effe,
                    1 - c**2 / etom * 1e-3 * dynamics.Pe / (2 * np.pi * 20.0))

    def test_zero_and_small_net_current_are_not_clipped(self):
        numerical, beam, grid, reference = inputs(iterfactor=1.0)
        state = BeamState.initialize(numerical, beam, grid, reference)
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        dynamics.IzDCr[:] = 1.0
        dynamics.Pe[:] = 0.2 * np.pi
        dynamics.IzDCr[:3] = [0.0, 0.0, 1e-20]
        dynamics.Pe[:3] = [0.0, 0.1, 2 * np.pi * 1e-21]
        with np.errstate(divide="ignore", invalid="ignore"):
            state.update(numerical, beam, reference, dynamics)
        self.assertTrue(np.isnan(state.vez[0]))
        self.assertTrue(np.isnan(state.gamaz[0]))
        self.assertTrue(np.isinf(state.gamaz[1]))
        np.testing.assert_allclose(state.gamaz[2], 1.1)
        np.testing.assert_allclose(state.vez[2], np.sqrt(1 - 1.1**-2) / reference.s)

    def test_negative_radicand_retains_matlab_complex_root(self):
        numerical, beam, grid, reference = inputs(iterfactor=1.0)
        state = BeamState.initialize(numerical, beam, grid, reference)
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        dynamics.IzDCr[:] = 1.0
        dynamics.Pe[:] = -np.pi  # gamma_eff = 0.5: no physical forward speed.
        state.update(numerical, beam, reference, dynamics)
        np.testing.assert_allclose(state.vez, 1j * np.sqrt(3.0) / reference.s)
        np.testing.assert_allclose(state.gamaz, 0.5)

    def test_zero_injected_energy_remains_undefined(self):
        numerical, beam, grid, reference = inputs(iterfactor=1.0)
        beam.Ut[:] = 0.0
        state = BeamState.initialize(numerical, beam, grid, reference)
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        dynamics.IzDCr[:] = 1.0
        dynamics.Pe[:] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            state.update(numerical, beam, reference, dynamics)
        self.assertTrue(np.all(np.isnan(state.effe)))


if __name__ == "__main__":
    unittest.main()
