"""Checks for MATLAB 532--534, 601--622 and iteration diagnostic records.

Run from pyKlyC: python -B -m unittest discover -s test -p test_convergence.py
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.settings import NumericalSettings
from solver.convergence import ConvergenceRecord, ConvergenceHistory, IterationControl
from solver.cavity_state import CavityState


def inputs():
    numerical = NumericalSettings(
        f=1.0, Nc=2, Ns=1, Nz=128, Nlimit=100, Ntimes=10,
        Rlimit=1e-4, iterfactor=0.5, evolutionmethod=np.array([0.005, 2.0]),
        upstream_discard_distance_mm=0.0, efficiency_end_offset_mm=15.0,
    )
    grid = SimpleNamespace(zi=np.array([0., 10., 20., 30.]))
    beam_state = SimpleNamespace(effe=np.array([0., 0.1, 0.3, 0.4]))
    cavity_state = SimpleNamespace(Vnc=np.array([1 + 1j, 2 + 0j]),
                                   flagVnc=np.array([1 + 0j, 1 + 0j]))
    dynamics = SimpleNamespace(Y=np.array([0, 2]), Yflag=2,
                               Zn=np.full((5, 3), np.inf))
    return numerical, grid, beam_state, cavity_state, dynamics


class ConvergenceTests(unittest.TestCase):
    def test_interpolation_and_complex_voltage_residual(self):
        args = inputs()
        voltage_before = args[3].flagVnc.copy()
        record = ConvergenceRecord.calculate(7, *args, effendflag=0.15)
        self.assertAlmostEqual(record.effend, 0.2)
        self.assertAlmostEqual(record.residential, 1 / np.sqrt(2))
        self.assertAlmostEqual(record.residentialel, 0.25)
        self.assertEqual(record.iteration, 7)
        self.assertEqual(record.electron_status, "normal")
        np.testing.assert_array_equal(args[3].flagVnc, voltage_before)

    def test_phase_only_change_is_detected(self):
        args = inputs()
        args[3].Vnc[:] = 1j
        args[3].flagVnc[:] = 1
        record = ConvergenceRecord.calculate(2, *args, effendflag=0.2)
        self.assertAlmostEqual(record.residential, np.sqrt(2))

    def test_endpoints_and_outside_grid(self):
        numerical, *rest = inputs()
        for offset, expected in [(0., 0.4), (30., 0.), (-1., np.nan), (31., np.nan)]:
            with self.subTest(offset=offset), np.errstate(divide="ignore", invalid="ignore"):
                record = ConvergenceRecord.calculate(
                    1, replace(numerical, efficiency_end_offset_mm=offset), *rest, effendflag=0.0,
                )
                np.testing.assert_allclose(record.effend, expected, equal_nan=True)

    def test_zero_denominators_and_matlab_max_nan_behavior(self):
        args = inputs()
        for current, previous, expected in [
            ([0., 2.], [1., 1.], np.inf),
            ([0., 2.], [0., 1.], 0.5),
            ([0., 0.], [0., 0.], np.nan),
        ]:
            with self.subTest(current=current, previous=previous):
                args[3].Vnc = np.array(current, dtype=complex)
                args[3].flagVnc = np.array(previous, dtype=complex)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    record = ConvergenceRecord.calculate(1, *args, effendflag=0.0)
                np.testing.assert_allclose(record.residential, expected, equal_nan=True)
        args = inputs()
        args[2].effe[:] = 0.0
        for previous, expected in [(0.0, np.nan), (0.1, np.inf)]:
            with np.errstate(divide="ignore", invalid="ignore"):
                record = ConvergenceRecord.calculate(1, *args, effendflag=previous)
            np.testing.assert_allclose(record.residentialel, expected, equal_nan=True)

    def test_diagnostic_status_uses_active_columns(self):
        args = inputs()
        args[4].Yflag = 1
        args[4].Zn[-1, 1] = 10.0  # Inactive electron must not affect status.
        record = ConvergenceRecord.calculate(2, *args, effendflag=0.2)
        self.assertEqual(record.electron_status, "reflected")
        args[4].Zn[-1, 2] = 10.0
        record = ConvergenceRecord.calculate(2, *args, effendflag=0.2)
        self.assertEqual(record.electron_status, "trapped")

    def test_history_and_explicit_reset_reference(self):
        args = inputs()
        history = ConvergenceHistory()
        self.assertIsNone(history.last)
        first = ConvergenceRecord.calculate(1, *args, effendflag=0.0)
        history.append(first)
        second = ConvergenceRecord.calculate(2, *args, effendflag=first.effend)
        history.append(second)
        self.assertEqual(second.residentialel, 0.0)
        self.assertIs(history.last, second)
        reset = ConvergenceRecord.calculate(3, *args, effendflag=0.0)
        self.assertEqual(reset.residentialel, 1.0)
        self.assertEqual(len(history.records), 2)
        # Existing plotting properties still work; they do not alter residuals.
        self.assertAlmostEqual(first.log_residential, np.log10(first.residential))
        self.assertEqual(second.log_residentialel, -300.0)


class IterationControlTests(unittest.TestCase):
    def test_decision_branches(self):
        # name, SCmethod, residual, local threshold, drive, optimize, all exit,
        # iteration, expected reason. Rlimit=1e-4 and Nlimit=100 in inputs().
        cases = [
            ("global converged", 1, 1e-5, 2., 1., False, True, 3, "converged"),
            ("global equal tolerance", 1, 1e-4, 2., 1., False, True, 3, "continue"),
            ("switch", 0, 0.5, 2., 1., False, True, 3, "switch_to_global"),
            ("switch despite tiny residual", 0, 1e-6, 2., 1., False, True, 3, "switch_to_global"),
            ("equal switch threshold", 0, 2., 2., 1., False, True, 3, "continue"),
            ("local only converged", 0, 4e-5, 5e-5, 1., False, True, 3, "converged"),
            ("local only tighter tolerance", 0, 6e-5, 5e-5, 1., False, True, 3, "continue"),
            ("equal local threshold", 0, 5e-5, 5e-5, 1., False, True, 3, "continue"),
            ("local threshold at Rlimit", 0, 5e-5, 1e-4, 1., False, True, 3, "converged"),
            ("global ramp not finished", 1, 1e-5, 2., 0.5, False, True, 3, "continue"),
            ("local ramp not finished", 0, 1e-5, 5e-5, 0.5, False, True, 3, "continue"),
            ("switch during ramp", 0, 0.5, 2., 0.5, False, True, 3, "switch_to_global"),
            ("optimization loss before convergence", 1, 1e-5, 2., 1., True, False, 3, "electron_loss"),
            ("optimization loss before switch", 0, 0.5, 2., 1., True, False, 3, "electron_loss"),
            ("optimization with exits", 1, 1e-5, 2., 1., True, True, 3, "converged"),
            ("ordinary run permits loss", 1, 1e-5, 2., 1., False, False, 3, "converged"),
            ("limit", 1, 1., 2., 1., False, True, 100, "iteration_limit"),
            ("converged at limit", 1, 1e-5, 2., 1., False, True, 100, "converged"),
            ("switch at limit", 0, 0.5, 2., 1., False, True, 100, "iteration_limit"),
            ("ramp at limit", 1, 1e-5, 2., 0.5, False, True, 100, "iteration_limit"),
            ("infinite residual", 1, np.inf, 2., 1., False, True, 3, "continue"),
            ("undefined residual", 0, np.nan, 2., 1., False, True, 3, "continue"),
        ]
        for name, mode, residual, threshold, drive, optimize, exits, nn, reason in cases:
            with self.subTest(case=name):
                numerical, grid, beam_state, cavity_state, dynamics = inputs()
                numerical = replace(numerical, evolutionmethod=np.array([0.005, threshold]))
                space_charge_state = SimpleNamespace(SCmethod=mode)
                dynamics.Yflag = len(dynamics.Y) if exits else 1
                # Deliberately bad efficiency residual: it is NOT a stop gate.
                record = ConvergenceRecord(nn, residual, np.nan, 0.25)
                control = IterationControl(effendflag=0.12)
                previous_flag = cavity_state.flagVnc.copy()
                current_voltage = cavity_state.Vnc.copy()
                control.update(record, numerical, cavity_state, space_charge_state,
                               dynamics, drivefactor=drive, optimization_mode=optimize)
                self.assertEqual(control.reason, reason)
                self.assertEqual(control.stop, reason in ["converged", "electron_loss", "iteration_limit"])
                self.assertEqual(control.iterationsuc, int(reason == "converged"))
                np.testing.assert_array_equal(cavity_state.Vnc, current_voltage)
                switched = mode == 0 and threshold > numerical.Rlimit and residual < threshold
                if reason in ["converged", "electron_loss"]:
                    self.assertEqual(space_charge_state.SCmethod, mode)
                    self.assertEqual(control.effendflag, 0.12)
                    np.testing.assert_array_equal(cavity_state.flagVnc, previous_flag)
                elif switched:
                    self.assertEqual(space_charge_state.SCmethod, 1)
                    self.assertEqual(control.effendflag, 0.0)
                    np.testing.assert_array_equal(cavity_state.flagVnc, np.zeros(2))
                else:
                    self.assertEqual(space_charge_state.SCmethod, mode)
                    self.assertEqual(control.effendflag, record.effend)
                    np.testing.assert_array_equal(cavity_state.flagVnc, current_voltage)
                    cavity_state.Vnc[:] = 100
                    np.testing.assert_array_equal(cavity_state.flagVnc, current_voltage)

    def test_switch_reset_survives_next_circuit_solve(self):
        numerical, grid, beam_state, _, dynamics = inputs()
        numerical = replace(numerical, Nc=1)
        cavity = CavityState(
            Pinflag=np.ones(1), Pin=np.ones(1), Vin=np.ones(1),
            flagVnc=np.array([2 + 0j]), Vnp=np.zeros(1, dtype=complex),
            Vnc=np.array([4 + 0j]), Mc=np.eye(2, dtype=complex),
            B=np.array([1 + 0j, 4 + 0j]),
        )
        space_charge = SimpleNamespace(SCmethod=0)
        control = IterationControl()
        history = ConvergenceHistory()
        first = ConvergenceRecord.calculate(1, numerical, grid, beam_state, cavity,
                                             dynamics, control.effendflag)
        history.append(first)
        control.update(first, numerical, cavity, space_charge, dynamics)
        self.assertEqual(control.reason, "switch_to_global")
        # Exercise the REAL solve(), replacing only assembly with a known matrix.
        with patch.object(cavity, "assemble_Mc_B"):
            cavity.solve(numerical, None, None, None, None, grid)
        np.testing.assert_array_equal(cavity.flagVnc, [0])
        second = ConvergenceRecord.calculate(2, numerical, grid, beam_state, cavity,
                                              dynamics, control.effendflag)
        self.assertEqual(second.residential, 1.0)
        self.assertEqual(second.residentialel, 1.0)
        history.append(second)
        control.update(second, numerical, cavity, space_charge, dynamics)
        self.assertFalse(control.stop)
        self.assertEqual(control.reason, "continue")
        with patch.object(cavity, "assemble_Mc_B"):
            cavity.solve(numerical, None, None, None, None, grid)
        third = ConvergenceRecord.calculate(3, numerical, grid, beam_state, cavity,
                                             dynamics, control.effendflag)
        history.append(third)
        self.assertEqual(third.residential, 0.0)
        self.assertEqual(third.residentialel, 0.0)
        control.update(third, numerical, cavity, space_charge, dynamics)
        self.assertEqual(control.reason, "converged")
        self.assertTrue(control.stop)
        self.assertEqual(control.iterationsuc, 1)
        self.assertEqual(len(history.records), 3)


if __name__ == "__main__":
    unittest.main()
