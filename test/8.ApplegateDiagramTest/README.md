# 8. Applegate diagram and current-harmonic plots

## Fresh solve with the manual's parameters

Run [manual_case.py](manual_case.py) to build and solve the five-cavity example
from scratch, independently of test 7:

```text
python test/8.ApplegateDiagramTest/manual_case.py
```

The script uses the manual's 1 GHz, 180 kV, 16 A beam, 60 W RF drive, and
five-cavity table. The entrance current is constant: there is no imposed
modulation. Accuracy settings are Nt=128, Nz=256, Ns=8, Nlimit=100,
Rlimit=1e-4 and iterfactor=0.5 (manual p. 8). Space charge is fully enabled
(`factor=1`, `DCSPfactor=1`), and the manual's `Image C=-1` is passed as
`fillfactor=-1`, following the GUI's `ConB` mapping.

It uses the 1D Gaussian option, with GUI defaults for N=200 local map intervals,
Ntimes=10, factorregion=10, extension/rc=2, evolutionmethod=[0.005,2], no power
ramp and no upstream discard cut. The domain follows the GUI's field-map
extent rule: entrance at min(map positions)-2*rc; exit at
max(map positions)+(4+extensionrc)*rc. Efficiency and the right plot edge are
sampled 3*rc before the exit. Ntp=64 is only a display choice.

The actual nonlinear solver is called, followed by the diagnostics and both
plotting functions. Outputs go into **manual_case_results/**: input and result
MAT files, PNG/PDF versions of both plots, and a text summary with termination
status. Existing paired-test inputs and results are not overwritten.

**Current model limitation:** the solver leaves cavity tracking fields Emz and
Emz0 zero. Its circuit voltages do not accelerate the beam. Thus this is a fresh
solve at the manual's parameters in the currently translated 1D/GS model, not
a validated reproduction of the manual's bunching, output power, efficiency,
or later EM/2D example. The script reports this limitation and saves the actual
fields and convergence status; it does not insert synthetic bunching or an
unreviewed cavity-field formula.

The script was executed with these settings: 3,699 planes, dz=0.7884472 mm,
domain [-74.7333, 2840.9444] mm. The solver reported convergence at iteration
71 (voltage residual 5.49e-5), taking about 153 s. All particles were counted
as exiting downstream. Maximum fundamental modulation was 2.23e-13 and the
cavity tracking field was exactly zero. Thus numerical convergence here does
not establish physical agreement with the manual. Saved results include the
actual voltage/efficiency histories and the separate field contributions.

## Paired plotting comparison

This test continues from test 7's **saved final solutions**. Python reads test
7's `python_results.mat`; MATLAB reads its own `matlab_results.mat`. Both read
the raw `inputs.mat` for cavity and entrance parameters. No Python trajectory
is supplied to MATLAB, and the nonlinear solve is not repeated here.

## Workflow

1. Run both producers in `../7.nonlinear_iteration_test/` if results are missing
   or you want to change the physical case.
2. Run **applegate_diagram_test.ipynb**. Adjust `Ntp` and `end_offset_mm`, inspect
   the plots, and export this folder's `inputs.mat` and `python_results.mat`.
3. Run **applegate_diagram_test.mlx** in MATLAB. The matching sectioned `.m`
   is its readable source. MATLAB selects its own reference electron and draws
   the original figures, then saves `matlab_results.mat`.
4. Run **compare_results.ipynb** to compare and overlay the plotted data.

Each producer saves two PNGs; the comparison saves `comparison.png`. Plots use
dashed gray grids. Run IDs reject stale outputs, including test 7 results from
a different run. Changing test 7 requires rerunning both test 8 producers.

## Meaning of the plots

- **Applegate:** physical position `Zn/betae` versus `Tn-Zn`, wrapped into one
  RF period around `tn1[ncenter]-z[0]`. Dots avoid false connecting segments
  across phase wrapping. Full trajectories include any time tracking;
  unwritten infinite entries are omitted by Python.
- **Right axis:** `abs(sum(ez * sign(nh)))`, the normalized RF cavity profiles.
  It excludes static gaps and is neither scaled by `Vnc` nor an instantaneous
  tracking field. Python creates this background without changing `ez`.
- **Harmonics:** `abs(Iz)` for every RF harmonic, already normalized by `Iz0`.
  DC is not plotted. Red dashed cavity markers span 0 to 2, as in MATLAB.

The default saved case has a modulated entrance beam and zero cavity tracking
fields, as in the available MATLAB solver. Nearly horizontal trajectories are
expected. This test demonstrates plot translation, not cavity-induced gain.

## MATLAB fidelity and comparison

The `.m` contains original lines 708-712 and 729-773 with visible arithmetic.
Adapters load the saved state, capture plot handles, clear the two test figures,
apply gray grids, omit unreadable comments, and save outputs. The reference-point
initialization and original plotting expressions remain. Python uses clearer
axis labels and Matplotlib in place of `plotyy`.
Regenerate the Live Script after editing the `.m`:

```matlab
matlab.internal.liveeditor.openAndSave('applegate_diagram_test.m', ...
    'applegate_diagram_test.mlx');
```

The comparator reads **actual plotting-artist coordinates**. It checks `Izmax`,
`ntemp`, `ncenter`, selected particle columns and stride; trajectory positions
and wrapped phases; background, harmonic amplitudes and line widths; cavity
markers and axis limits. Only declared vector orientations and MATLAB's
one-based indices are adapted. Omitted history positions have matching NaN
masks. Finite values use `rtol=1e-8, atol=1e-10`; indices and stride are exact.

This is a data comparison, not a pixel comparison. The default background is
nonconstant. Python also handles a constant/zero background with automatic
limits, where MATLAB's explicit equal limits would fail.

Focused tests in `../test_beam_plotting.py` check reference-speed drift,
noncontiguous active columns, MATLAB rounding, phase wrapping, unwritten rows,
DC suppression, complex harmonic magnitudes, and unchanged input arrays.

## Verification

The Python notebook, MATLAB script, and comparison notebook were executed on
the saved test 7 case. All 17 exported numeric quantities passed. Particle
selection and reference indices agreed exactly; maximum differences were
8.53e-14 rad in wrapped phase, 4.55e-13 mm in trajectory position, and 1.20e-14
in harmonic magnitude. All three focused Python plotting tests passed.
