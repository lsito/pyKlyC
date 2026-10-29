"""End-to-end tests of the supplied-source nonlinear iteration.

The expected zero cavity-to-beam field is tested explicitly; these tests do
not establish physical klystron gain. No notebook outputs are overwritten.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.beam import BeamParameters
from model.cavity import CavityParameters
from model.settings import NumericalSettings
from model.tube import TubeParameters
from solver.reference import ReferenceParameters
from solver.grid import GridParameters
from solver.cavity_fields import CavityFields
from solver.beam_dynamics import BeamDynamics
from solver.convergence import ConvergenceRecord
from solver.large_signal_1d import solve, calculate_drivefactor


def inputs(modulation=0.0, Nlimit=20):
    numerical = NumericalSettings(
        f=1.0, Nc=2, Ns=2, Nz=64, Nlimit=Nlimit, Ntimes=5,
        Rlimit=1e-4, iterfactor=0.5, evolutionmethod=np.array([0.005, 2.0]),
        upstream_discard_distance_mm=0.0, efficiency_end_offset_mm=10.0,
    )
    t0 = 2 * np.pi * np.arange(16) / 16
    beam = BeamParameters(
        Nbeam=1, rm2=6., rm1=0., rc=10., xb=0., yb=0., xd=0., yd=0.,
        Nt=16, t0=t0, Ut=np.full(16, 180.),
        Izt=-1.0 * (1 + modulation * np.cos(t0)), tn1=t0.copy(), Y0=np.arange(16),
    )
    cavities = CavityParameters(
        type=np.array([0, -1]), nh=np.array([1, 1]), f0=np.array([1., 1.01]),
        Qext=np.array([100., 80.]), Qin=np.array([10000., 10000.]),
        zshift=np.array([20., 80.]), Pin=np.array([10., 0.]),
        gaman=np.zeros(2, dtype=complex), Lsb=np.full(2, np.pi),
        cavityname="input output", kai=np.zeros((2, 2), dtype=complex),
    )
    reference = ReferenceParameters.calculate_reference_parameters(beam)
    tube = TubeParameters(z0=0., L=100., rcv=np.array([[-10., 110.], [1., 1.]]))
    grid = GridParameters.calculate(numerical, cavities, tube, beam, reference, 5.)
    # Small prescribed normalized profiles isolate loop wiring from map imports
    # (already covered by the dedicated cavity-field notebook comparisons).
    C3 = 0.01 * np.exp(-((grid.zi.reshape(-1, 1) - cavities.zshift) / 10.)**2)
    fields = CavityFields(ez=C3.copy(), C3=C3, vpz=np.zeros_like(C3), alfaz=np.zeros_like(C3))
    return numerical, beam, cavities, grid, reference, fields


def run(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The available MATLAB source.*", category=RuntimeWarning)
        return solve(*args, **kwargs)


class NonlinearSolveTests(unittest.TestCase):
    def test_no_space_charge_and_no_cavity_acceleration(self):
        args = inputs(modulation=0.15, Nlimit=40)
        result = run(*args, factor=0.0)
        self.assertTrue(result.control.stop)
        self.assertEqual(result.control.reason, "converged")
        self.assertEqual(result.control.iterationsuc, 1)
        self.assertEqual(result.space_charge_state.SCmethod, 1)
        self.assertEqual(result.SCmethods[0], 0)
        self.assertEqual(result.SCmethods[1], 1)
        np.testing.assert_array_equal(result.Emz, 0)
        np.testing.assert_array_equal(result.Emz0, 0)
        np.testing.assert_array_equal(result.Esmz, 0)
        np.testing.assert_array_equal(result.Esmz0, 0)
        np.testing.assert_allclose(result.beam_state.effe, 0., atol=1e-14)
        np.testing.assert_allclose(result.beam_state.vez, 1., atol=1e-14)
        np.testing.assert_allclose(result.dynamics.Utn, args[1].Ut)
        self.assertGreater(np.max(np.abs(result.cavity_state.Vnc)), 0)
        self.assertEqual(len(result.history.records), len(result.Vnc_history))
        self.assertEqual(len(result.history.records), len(result.drivefactors))

    def test_source_limitation_is_reported(self):
        with self.assertWarnsRegex(RuntimeWarning, "cavity-to-beam acceleration"):
            solve(*inputs(Nlimit=1), factor=0.)

    def test_iteration_callback_is_observational_and_skips_ramp_gaps(self):
        args = inputs(modulation=0.15, Nlimit=80)
        baseline = run(*args, factor=0., power_ramp_mode=True, Npin=4)
        snapshots = []
        def capture(record, beam_state, cavity_state, space_charge_state, dynamics):
            snapshots.append((record.iteration, space_charge_state.SCmethod,
                              cavity_state.Vnc.copy(), beam_state.Iz.copy()))
            np.testing.assert_allclose(beam_state.effe, 0.0, atol=1e-14)
        result = run(*args, factor=0., power_ramp_mode=True, Npin=4, on_iteration=capture)
        self.assertEqual([snapshot[0] for snapshot in snapshots],
                         [record.iteration for record in result.history.records])
        np.testing.assert_array_equal([snapshot[1] for snapshot in snapshots], result.SCmethods)
        np.testing.assert_array_equal([snapshot[2] for snapshot in snapshots], result.Vnc_history)
        np.testing.assert_array_equal(result.Vnc_history, baseline.Vnc_history)
        np.testing.assert_array_equal(result.skipped_iterations, baseline.skipped_iterations)
        np.testing.assert_array_equal(result.beam_state.Iz, baseline.beam_state.Iz)
        self.assertGreater(len(result.skipped_iterations), 0)
        self.assertFalse(np.shares_memory(snapshots[0][3], result.beam_state.Iz))

    def test_iteration_limit(self):
        result = run(*inputs(modulation=0.2, Nlimit=1), factor=0.)
        self.assertEqual(result.control.reason, "iteration_limit")
        self.assertEqual(result.control.iterationsuc, 0)
        self.assertEqual(result.nn, 1)
        self.assertEqual(len(result.history.records), 1)

    def test_local_only_mode(self):
        numerical, *rest = inputs(modulation=0.15, Nlimit=40)
        numerical = replace(numerical, evolutionmethod=np.array([0.005, numerical.Rlimit]))
        result = run(numerical, *rest, factor=0.)
        self.assertEqual(result.control.reason, "converged")
        np.testing.assert_array_equal(result.SCmethods, 0)
        self.assertEqual(result.space_charge_state.SCmethod, 0)

    def test_actual_space_charge_updates_and_fresh_passes(self):
        args = inputs(modulation=0.1, Nlimit=8)
        fresh = []
        initialize = BeamDynamics.initialize
        def capture(*args):
            state = initialize(*args)
            fresh.append(state)
            return state
        with patch.object(BeamDynamics, "initialize", side_effect=capture):
            result = run(*args, factor=0.01)
        self.assertEqual(len(fresh), len(result.history.records))
        for previous, following in zip(fresh, fresh[1:]):
            self.assertFalse(np.shares_memory(previous.Zn, following.Zn))
        self.assertGreater(np.max(np.abs(result.space_charge_state.Esz)), 0.)
        np.testing.assert_array_equal(result.Esmz0, result.space_charge_state.Esz0 + result.Emz0)
        self.assertTrue(np.all(np.isfinite(result.beam_state.Iz)))
        self.assertTrue(np.all(np.isfinite(result.Vnc_history)))

    def test_time_continuation_in_outer_loop(self):
        numerical, *rest = inputs(modulation=0.1, Nlimit=3)
        numerical = replace(numerical, evolutionmethod=np.array([-1., 2.]))
        track_t = BeamDynamics.track_t
        calls = []
        def capture(state, *args):
            calls.append(args[-1])
            return track_t(state, *args)
        with patch.object(BeamDynamics, "track_t", new=capture):
            result = run(numerical, *rest, factor=0.)
        self.assertEqual(len(calls), len(result.history.records))
        self.assertTrue(all(start == 1 for start in calls))
        self.assertEqual(result.dynamics.Yflag, rest[0].Nt)

    def test_power_ramp_schedule_and_skips(self):
        args = inputs(modulation=0.15, Nlimit=80)
        result = run(*args, factor=0., power_ramp_mode=True, Npin=4)
        self.assertEqual(result.drivefactors[0], 0.5)
        self.assertEqual(result.drivefactors[-1], 1.0)
        self.assertEqual(result.control.reason, "converged")
        self.assertGreater(len(result.skipped_iterations), 0)
        evaluated = [record.iteration for record in result.history.records]
        self.assertEqual(sorted(evaluated + result.skipped_iterations.tolist()), list(range(1, result.nn + 1)))
        for record, drive in zip(result.history.records, result.drivefactors):
            self.assertEqual(drive, calculate_drivefactor(record.iteration, args[0], 4, True))

    def test_limit_reached_via_skips(self):
        numerical, *rest = inputs(Nlimit=6)
        numerical = replace(numerical, evolutionmethod=np.array([0.005, 1e-6]))
        record = ConvergenceRecord(1, 5e-5, 0.0, 0.0)
        with patch.object(ConvergenceRecord, "calculate", return_value=record):
            result = run(numerical, *rest, factor=0., power_ramp_mode=True, Npin=1)
        self.assertEqual(result.nn, 6)
        self.assertEqual(result.control.reason, "iteration_limit")
        self.assertEqual(result.control.iterationsuc, 0)
        np.testing.assert_array_equal(result.skipped_iterations, np.arange(2, 7))

    def test_optimization_exit_and_closed_window(self):
        track = BeamDynamics.track
        def missing_exit(state, *args, **kwargs):
            total_fields = track(state, *args, **kwargs)
            state.Yflag -= 1
            return total_fields
        with patch.object(BeamDynamics, "track", new=missing_exit):
            result = run(*inputs(), factor=0., optimization_mode=True)
        self.assertEqual(result.control.reason, "electron_loss")
        self.assertEqual(result.nn, 1)
        plotter = SimpleNamespace(is_open=lambda: False)
        result = run(*inputs(), factor=0., plotter=plotter)
        self.assertEqual(result.control.reason, "user_abort")
        self.assertEqual(result.control.iterationsuc, 0)

    def test_independent_runs_and_unsupported_modes(self):
        args = inputs(Nlimit=1)
        first = run(*args, factor=0.)
        second = run(*args, factor=0.)
        for name in ["beam_state", "cavity_state", "space_charge_state", "dynamics", "control", "history"]:
            self.assertIsNot(getattr(first, name), getattr(second, name))
        first.beam_state.Iz[:] = 999
        self.assertFalse(np.any(second.beam_state.Iz == 999))
        with self.assertRaises(NotImplementedError):
            run(*args, mono=0)
        numerical, beam, cavities, *rest = args
        with self.assertRaises(NotImplementedError):
            run(numerical, beam, replace(cavities, type=np.array([0, 2])), *rest)


if __name__ == "__main__":
    unittest.main()
