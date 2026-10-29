"""Optional original-MATLAB loop comparison; see test/README.md for commands.

No argument: generate inputs and Python outputs in a fresh temporary directory.
One directory argument: compare with the outputs of nonlinear_reference.m.
"""

from pathlib import Path
from dataclasses import fields, replace
import sys
import tempfile
import warnings
import numpy as np
from scipy.io import savemat, loadmat

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'test'))
from test_large_signal_1d import inputs, run
from solver.beam_state import BeamState
from solver.cavity_state import CavityState
from solver.space_charge import SpaceChargeState
from constants import c, etom, esign, eta0


def prepare():
    folder = Path(tempfile.mkdtemp(prefix='pyklyc_nonlinear_'))
    cases = [
        ('ordinary', 40, 0., False, False, False, False),
        ('local', 8, 0.01, True, False, False, False),
        ('global', 8, 0.01, False, False, False, False),
        ('ramp', 80, 0., False, True, False, False),
        ('time', 3, 0., False, False, True, False),
        ('optimization', 3, 0., False, False, True, True),
    ]
    for name, limit, factor, local, ramp, timed, opt in cases:
        numerical, beam, cavities, grid, reference, maps = inputs(0.15, limit)
        numerical = replace(numerical,
            evolutionmethod=np.array([-1. if timed else 0.005, numerical.Rlimit if local else 2.]),
            Ntimes=1 if opt else numerical.Ntimes)
        state = BeamState.initialize(numerical, beam, grid, reference)
        cavity = CavityState.initialize(numerical, cavities, beam, reference)
        sc = SpaceChargeState.initialize(beam, grid, reference, state, factor, 1., 1.)
        data = {}
        for obj in [numerical, beam, cavities, grid, reference, maps, state, cavity, sc]:
            data.update({f.name: getattr(obj, f.name) for f in fields(obj)})
        data.update(c=c, etom=etom, esign=esign, eta0=eta0, factor=factor,
                    fillfactor=1., DCSPfactor=1., mono=1., Npin=4 if ramp else 1,
                    moviedisp=np.array([0, 0, 0, -2 if ramp and opt else 2 if ramp else -1 if opt else 0, 1]),
                    GIFpara=np.array([0., 0., 0., numerical.upstream_discard_distance_mm,
                                      numerical.efficiency_end_offset_mm, 0.]),
                    effendflag=0., iterationsuc=0.)
        data['Y0'] = beam.Y0 + 1
        data['Nesz'] = grid.Nesz + np.array([1, 0])
        for p in range(5):
            data['Eszr' + str(p+1)] = sc.Eszr[:, p].reshape(-1, 1)
        for key in ['z', 'zi', 'rcvs', 'IzDC', 'QQDC', 'effe', 'vez', 'gamaz', 'Esz0']:
            data[key] = data[key].reshape(-1, 1)
        for key, value in data.items():
            if isinstance(value, (int, np.integer)):
                data[key] = float(value)
            elif isinstance(value, np.ndarray) and value.dtype.kind in 'iu':
                data[key] = value.astype(float)
        savemat(folder / (name + '_input.mat'), data, oned_as='row')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = run(numerical, beam, cavities, grid, reference, maps,
                         factor=factor, power_ramp_mode=ramp, Npin=4 if ramp else 1,
                         optimization_mode=opt)
        output = {}
        for obj in [result.beam_state, result.cavity_state, result.dynamics]:
            output.update({f.name: getattr(obj, f.name) for f in fields(obj)})
        for name_field in ['Emz', 'Emz0', 'Esmz', 'Esmz0', 'nn', 'Vnc_history', 'Vnp_history',
                           'drivefactors', 'SCmethods', 'skipped_iterations']:
            output[name_field] = getattr(result, name_field)
        output.update(SCmethod=result.space_charge_state.SCmethod,
                      iterationsuc=result.control.iterationsuc, effendflag=result.control.effendflag,
                      records=np.array([[r.iteration, r.effend, r.residential, r.residentialel]
                                        for r in result.history.records]))
        savemat(folder / (name + '_python.mat'), output)
        print(f'Python {name}: {result.nn} attempted, {len(result.history.records)} evaluated, {result.control.reason}')
    print('RESULT_FOLDER=' + str(folder))


def compare(folder):
    for path in sorted(folder.glob('*_python.mat')):
        expected = loadmat(path)
        actual = loadmat(str(path).replace('_python.mat', '_matlab.mat'))
        for key, value in expected.items():
            if key.startswith('__'):
                continue
            # MATLAB reuses Esz0 for the total DC field. Python reserves that
            # name for space charge and names the total Esmz0.
            matlab_key = 'Esz0' if key == 'Esmz0' else key
            other = actual[matlab_key]
            if key == 'Y':
                other = other - 1
            if key == 'records':
                # Efficiency itself is ~roundoff without cavity acceleration.
                # Compare all other diagnostics directly; validate the ill-
                # conditioned ratio against each engine's own stored effend.
                np.testing.assert_allclose(other[:, :3], value[:, :3], rtol=1e-9, atol=1e-10,
                                           err_msg=f'{path.stem}: record iteration/effend/voltage residual')
                for records, modes in [(value, expected['SCmethods']), (other, actual['SCmethods'])]:
                    modes = modes.ravel()
                    previous = np.concatenate(([0.], records[:-1, 1]))
                    previous[1:][modes[1:] != modes[:-1]] = 0.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        residual = np.abs((previous - records[:, 1]) / records[:, 1])
                    np.testing.assert_allclose(records[:, 3], residual, rtol=1e-12, atol=1e-12)
                print(f'  Efficiency residual formula verified in both engines; max |effend|={np.max(np.abs(value[:, 1])):.3e}')
                continue
            np.testing.assert_allclose(other.reshape(value.shape), value, rtol=1e-9, atol=1e-10,
                                       err_msg=f'{path.stem}: {key}')
        print(f'PASS MATLAB comparison: {path.stem}, {len(expected)-3} outputs')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        prepare()
    else:
        compare(Path(sys.argv[1]))
