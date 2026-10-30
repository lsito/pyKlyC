# pyKlyC
Python computation tool for 1/1.5D klystron simulations.

At the moment, the idea would be:

```
pyklyc/src
  constants.py
  units.py

  model/ --> Dataclasses
    beam.py
    cavity.py
    tube.py
    field_map.py
    settings.py

  solver/
    grid.py
    cavity_solver.py
    space_charge.py
    beam_dynamics.py
    large_signal_1d.py
    large_signal_15d.py
    convergence.py

  io/
    klc.py
    field_import.py
    matlab_compat.py

  diagnostics/
    power_balance.py
    efficiency.py
    bunching.py

  plotting/
    applegate.py
    modulation.py
    phase_space.py
    power_balance.py
    convergence.py

  tests/
    test_reference_cases.py
```

## 1. We start with a 1D klystron code with Gaussian cavity fields.
We mean no GUI, no EM solver, no Optimizer, no CGUN, no magnetostatic solver.
The idea is to start porting:
    - KLJCmainsv22.m --> 1D solver
    - Klyc_v6.mlapp --> GUI and IO mapping (we focus on the IO mapping)

### KLJCmainsv22.m inputs:
We need first to understand how we want to build the datamodels.

#### Beam Parameters (in src/model/beam.py --> BeamParameters dataclass): 
| MATLAB parameter | Meaning                                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `Nt`             | Number of emitted disks per RF period.                                                                |
| `Nbeam`                | Number of identical beams. For multi-beam klystron scaling.                                      |
| `rm2`                  | Outer beam radius, mm.                                                                           |
| `rm1`                  | Inner beam radius, mm. Zero for solid cylindrical beam.                                          |
| `rc`                   | Drift-tube / beam tunnel radius, mm. If `rc ~= 0`, the code treats the case as cylindrical beam. |
| `xb`, `yb`, `xd`, `yd` | Sheet-beam dimensions. In this 1D solver, sheet-beam branches are mostly empty or inactive.      |
| `t0`             | Emission phase samples over one RF period, usually `0:2π/Nt:2π`.                                                                        |
| `Ut`             | Beam voltage for each emitted disk, kV. Usually constant for klystron mode.                                                             |
| `Izt`            | Beam current carried by each emitted disk/sample, A. Usually negative for electron current.                                             |
| `tn1`            | Initial normalized arrival time/phase. Usually same as `t0`.                                                                            |
| `Y0`             | Active emitted disk indices. In normal klystron mode this is `1:Nt`; in bunched-beam/IOT mode only part of the RF period may be active. |

#### RF/cavity parameters (in src/model/cavity.py --> CavityParameters dataclass):
| MATLAB parameter | Meaning                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`           | Cavity type: `0` input, `1` bunching/intermediate, `-1` output. The code also has partial branches for `2`, probably traveling-wave structures. |
| `nh`             | Harmonic number used by each cavity. Fundamental is `1`; harmonic cavities may be `2`, `3`, etc.                                                |
| `f0`             | Cold resonant frequency of each cavity, GHz.                                                                                                    |
| `Qext`           | External Q. Mainly used for input/output coupling.                                                                                              |
| `Qin`            | Intrinsic/unloaded Q, used for ohmic loss.                                                                                                      |
| `zshift`         | Axial cavity center/field-map shift, mm.                                                                                                        |
| `Pin`            | Input RF power, W. Used for input cavity excitation and power ramp.                                                                             |
| `gaman`          | Complex reflection coefficient at output load.                                                                                                  |
| `cavityname`     | Space-separated list of field-map file names, for example `"KlyCcavity1 KlyCcavity2 ..."`.                                                      |
| `kai`             | Cavity-cavity coupling matrix.                                                                                                                      |

#### Tube Geometry (in src/model/tube.py --> TubeParameters dataclass):
| MATLAB parameter       | Meaning                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `z0`                   | Start position of simulation domain, mm.                                                         |
| `L`                    | Total simulation length, mm.                                                                     |
| `rcv`             | Axial tunnel-radius profile. First row is z, second row is radius normalized to `rc`.                 |


#### Numerical controls (in src/model/settings.py --> NumericalSettings dataclass):
| MATLAB parameter | Meaning                                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------------------- |
| `Nc`             | Number of cavities / RF structures.                                                                   |
| `Ns`             | Number of harmonics used for current, charge, and space-charge fields.                                |
| `Nz`             | Number of spatial divisions per electronic wavelength. Used to choose `dz`.                           |
| `Nlimit`         | Maximum number of nonlinear iterations.                                                               |
| `Ntimes`         | Extra trajectory-storage/evolution factor. The arrays `Zn`, `Vn`, `Tn` have length `Ntimes * len(z)`. |
| `Rlimit`         | Convergence tolerance for cavity voltage residual.                                                    |
| `evolutionmethod` | Two-element array in `NumericalSettings`: `[0]` limits the relative increase in inverse velocity per spatial step; `[1]` controls the local/global space-charge transition and local-mode convergence. MATLAB's ordinary GUI setting is `[0.005, 2.0]`. |
| `upstream_discard_distance_mm` | Reflected-electron cutoff distance from the simulation entrance, in mm. Use `0.0` to track up to upstream escape. Replaces MATLAB's `GIFpara(4)` in tracking; unlike the GUI's `zcut (mm)`, this is a distance, not an absolute coordinate. |
| `efficiency_end_offset_mm` | Efficiency observation plane is `max(grid.zi) - efficiency_end_offset_mm`. Replaces MATLAB's `GIFpara(5)`. The example notebooks use `3 * rc`; zero samples the last plane. |
| `f`              | Operating frequency, GHz.                                                                                                                       |
| `mono`           | Mode flag. `1` is normal klystron simulation; `0` activates monotron-frequency logic.                                                           |

#### Plot/Output controls (in src/model/settings.py --> DisplaySettings dataclass):
| MATLAB parameter | Meaning                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `figuredisp`     | `'on'` or `'off'`; controls static plots.                                                                        |
| `moviedisp`      | Vector of flags controlling GIFs, power ramp, trajectory saving, and convergence display. Eg. [0 0 0 0 1] display convergence history.                       |
| `GIFpara`        | Plot/animation parameters: number of plotted trajectories, min/max velocity axis, z-cut, end cut, velocity type. |
| `Npin`           | Number of input-power ramp points.                                                                    |

`track_t()` reads the reflected-electron cutoff from `NumericalSettings`, not
`DisplaySettings`. To compare tracking with the original MATLAB solver, export
`numerical.upstream_discard_distance_mm` as `GIFpara(4)` (Python index `3`).
MATLAB couples that plot setting to tracking; Python keeps the two independent.
Likewise, export `numerical.efficiency_end_offset_mm` as `GIFpara(5)` (Python
index `4`) when comparing efficiency and convergence with MATLAB.

Create `control = IterationControl()` and `history = ConvergenceHistory()` once
per nonlinear solve, alongside a fresh `CavityState`. After each
`beam_state.update(numerical, beam, reference, dynamics)`, calculate and store
the iteration diagnostics, then decide whether to continue:

```python
record = ConvergenceRecord.calculate(
    nn, numerical, grid, beam_state, cavity_state, dynamics, control.effendflag,
)
history.append(record)
control.update(
    record, numerical, cavity_state, space_charge_state, dynamics,
    drivefactor=drivefactor, optimization_mode=optimization_mode,
)
# The outer loop breaks when control.stop is True.
```

`record` contains `effend`, `residential`, `residentialel`, and the electron-status
label. `control.update()` implements MATLAB's end-of-iteration decisions: voltage
convergence, local-to-global space-charge switching, optimization-mode electron
loss, and iteration-limit failure. Its `reason` describes the decision;
`iterationsuc` is 1 only for convergence, and `stop` requests termination.

Use the same voltage-amplitude `drivefactor` as in the cavity solve (1.0 without
ramping). `optimization_mode` corresponds to MATLAB `moviedisp(4) < 0`. These are
explicit control inputs, not an optimizer. The outer solver below supplies the
ramp schedule and skips; final power accounting remains separate post-processing.

`IterationControl` owns the efficiency comparison value and updates or resets
`cavity_state.flagVnc`. `CavityState.solve()` no longer overwrites that reference,
so the reset when switching to global space charge survives the next solve.
Residual calculation and history append alone do not change comparison values.
The efficiency residual is diagnostic; it is not an extra convergence gate.

#### Nonlinear 1D solve

See [the orchestration schematic](docs/large_signal_1d.md) for the flowchart,
UML-style sequence diagram, state lifetimes, iteration decisions and result
interpretation for `large_signal_1d.solve()`.

After constructing `numerical`, `beam`, `cavities`, `reference`, `grid`, and
`cavity_fields` as in the test notebooks, run:

```python
from solver.large_signal_1d import solve

result = solve(
    numerical, beam, cavities, grid, reference, cavity_fields,
    factor=1.0, fillfactor=1.0, DCSPfactor=1.0,
    power_ramp_mode=False, Npin=1, optimization_mode=False,
)
print(result.control.reason, result.control.iterationsuc)
print(result.history.last)
```

For iteration snapshots, pass `on_iteration=callback`. The callback receives
`(record, beam_state, cavity_state, space_charge_state, dynamics)` after the
beam update and before the convergence decision. Treat the states as read-only
and copy arrays that must survive the next iteration. It is not called for
ramp-skipped iterations. The paired Python/MATLAB notebook test is in
[test 7](test/7.nonlinear_iteration_test/README.md).

The function creates fresh beam/cavity/space-charge states per run. It solves
the circuit, updates space charge, tracks a fresh entrance beam, relaxes the
beam estimate, and checks convergence at every evaluated iteration. It also
implements MATLAB's staircase power ramp, settled-step skips, and optimization
exit. An optional existing `ConvergencePlotter` can be passed as `plotter`;
closing it requests `user_abort`. No plots or files are created by default.

`BeamDynamics.track()` receives separate `Esz`, `Esz0`, `Emz`, and `Emz0`
contributions. It assembles the RF/DC totals internally once and returns them
for diagnostics. Only the optional current-based space-charge correction
updates RF rows during the spatial march; the ordinary sum is not repeated.
The returned totals are `result.Esmz` and `result.Esmz0`. Separate space-charge
fields are `result.space_charge_state.Esz` and `.Esz0`; cavity contributions
are `result.Emz` and `result.Emz0`. The MATLAB comparison adapter maps Python's
`Esmz0` to MATLAB's reused total-field name `Esz0`.

`result` retains the final states and trajectory, field buffers, convergence
history, per-pass voltage histories/drive factors/space-charge modes, skipped
iteration numbers, and the last attempted iteration `nn`. History length can
be smaller than `nn` during ramping. Input parameter objects are not modified.

**Source limitation:** the available MATLAB file leaves `Emz` and `Emz0` zero.
The solver deliberately does the same and emits a warning: cavity voltages are
computed, but they do not accelerate the beam. Convergence is not evidence of
physical klystron gain. Adding the missing cavity-to-beam field connection is
deferred, not silently inferred. Monotron, sheet-beam, DC-gap and traveling-wave
branches are explicitly unsupported by this entry point. Final cavity power/loss
accounting and space-charge energy accounting are available separately below.
Corrected ramp-efficiency tables and animations are not included.

#### First post-processing step: final trajectory diagnostics

The [LaTeX variable reference](docs/postprocessing_variables.tex) collects the
post-processing and plotting quantities, including all six MATLAB `GIFpara`
entries and their Python counterparts.

MATLAB lines 640-662 inspect the completed particle histories. Run this after
`solve()`; it does not change the solution or its convergence decision:

```python
from diagnostics.beam import check_boundary_reaching, check_negative_velocity

boundary = check_boundary_reaching(result.dynamics)
velocity = check_negative_velocity(result.dynamics, reference)
print(boundary)
print(velocity)
```

The boundary result contains `n` (MATLAB's one-based last written history row)
and `status`. The velocity result contains `minimum_v_over_c` and
`negative_velocity`. The original `Storage limit` branch
takes priority over the downstream exit count. Its `Reflected electrons` classification
means missing downstream exits; recorded backward motion is checked separately.
These final checks differ from the per-iteration plotting label in
`ConvergenceRecord`.

#### Cavity gap voltages and power accounting

MATLAB lines 664-706 are available as a separate diagnostic after `solve()`:

```python
from diagnostics.cavity import calculate_cavity_power

cavity_power = calculate_cavity_power(
    cavities, cavity_fields, result.cavity_state, beam, grid, reference,
    drivefactor=result.drivefactors[-1],
)
```

`Vng` contains complex gap voltages in kV. `effloss` contains each cavity's
wall loss divided by total reference beam power; `lossacc` accumulates these
fractions along `grid.zi`, with half a cavity's loss at its center. `RFinput`
is the sum of net RF input powers in W. As in MATLAB, `refcavity` is the
complex reflection coefficient of the last input cavity, and `effcavity` is
the load efficiency of the last output cavity. Both default to zero when
there is no corresponding cavity.

Use the saved drive factor from the final evaluated circuit solve: it scales
incident voltage, while its square scales power. The function changes no
solver state. Traveling-wave accounting is not implemented. The static-gap
voltage formula is included for supplied circuit states; the nonlinear solver
still does not support DC gaps.

#### Peak fundamental modulation and plot reference electron

MATLAB lines 708-712 locate the strongest fundamental current modulation and
select an active electron nearest the fundamental peak phase at that plane:

```python
from diagnostics.beam import find_peak_modulation

modulation = find_peak_modulation(result.beam_state, result.dynamics)
zpeak = grid.zi[modulation["ntemp"]]  # Position in mm.
phase_reference = beam.tn1[modulation["ncenter"]]  # Entrance phase in radians.
```

The dictionary contains `Izmax`, `ntemp`, and `ncenter`. Both indices are
zero-based: `ntemp` indexes the longitudinal grid, while `ncenter` indexes
particle columns directly, after mapping through `dynamics.Y`. The selected
electron provides a reference for centering the trajectory plot. Higher
harmonics can shift the actual current peak away from the fundamental peak.

#### Space-charge energy exchange and final power balance

MATLAB lines 715-724 are available as two separate diagnostics:

```python
from diagnostics.space_charge import calculate_space_charge_power
from diagnostics.power_balance import calculate_power_balance

space_charge_power = calculate_space_charge_power(
    result.beam_state, result.space_charge_state, grid,
)

power_balance = calculate_power_balance(
    result.beam_state, space_charge_power, cavity_power,
    numerical, beam, grid, reference,
)
```

`space_charge_power` contains the longitudinal arrays `effSCDC`, `effSCRF`,
and their sum `effSC`. These are cumulative power fractions relative to the
entrance, normalized to `Nbeam * (U0 * 1000) * I0`. The integration uses the
left endpoint of every interval, as in MATLAB. RF power uses
`0.5 * real(conj(Iz) * Esz)`, summed over harmonics. Negative contributions
are retained; this is energy exchange rather than purely dissipative loss.

`power_balance` contains `effend`, `effSCend`, and `effendmodified`. The first
two are interpolated at `max(grid.zi) - numerical.efficiency_end_offset_mm`.
The final balance subtracts the space-charge contribution and all cavity
wall losses from the beam efficiency, then adds net RF input divided by
reference beam power. It is an efficiency inferred from energy balance,
distinct from the output-port efficiency `cavity_power["effcavity"]`.
An observation point outside the grid gives NaN, matching MATLAB `interp1`.

Both functions read the returned states without updating them. They use the
separate space-charge fields, not the total tracking fields. MATLAB reuses
`Esz0` for its total DC field; these agree in the supported solver because
`Emz0` is zero. Future DC cavity support will need explicit power accounting.
#### Applegate diagram and current harmonics

The first two MATLAB figures (lines 729-773) are available as ordinary plotting
functions. Both return the figure and axes; the caller can display or save them.

```python
from plotting.beam import plot_applegate, plot_current_harmonics

modulation = find_peak_modulation(result.beam_state, result.dynamics)
zmax = grid.zi.max() - numerical.efficiency_end_offset_mm
fig, axes = plot_applegate(
    result.dynamics, beam, grid, cavities, cavity_fields,
    ncenter=modulation["ncenter"], Ntp=100, zmax=zmax,
)
fig, ax = plot_current_harmonics(result.beam_state, grid, cavities, zmax=zmax)
```

The Applegate plot uses full trajectory histories, wraps arrival-phase delay
into one RF period, and excludes unwritten history entries. `Ntp=None` displays
all active samples; a positive `Ntp` selects a subset using MATLAB's rounding.
The second axis displays normalized RF cavity profiles, not the instantaneous
force field. The current plot shows already-normalized RF harmonic amplitudes
and cavity-center markers. Neither function modifies the stored data.
Both use dashed gray grids. See
[test 8](test/8.ApplegateDiagramTest/README.md) for notebooks and MATLAB comparison.

#### Space-charge and iteration controls:
| MATLAB parameter  | Meaning                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factor`          | Space-charge multiplier. `0` disables space charge; `1` nominally enables it.                                                                       |
| `factorregion`    | Local space-charge interaction window size, used as `factorregion * rc / 2.40483`.                                                                  |
| `fillfactor`      | Adds pre-filled upstream DC beam contribution to space charge.                                                                                      |
| `DCSPfactor`      | Scales DC space-charge component.                                                                                                                   |
| `iterfactor`      | Iteration relaxation coefficient for updating `Iz`, `QQ`, `IzDC`, `QQDC`.                                                                           |



### KLJmainsv22.m reference parameter calculations:
| MATLAB parameter  | Meaning                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Iz0`          | Signed average emission current, in amperes. Given by the average in the RF period of the sum of the emission current(Izt) associated with all (Nt) emitted samples. $$I_{z0}= \frac{1}{N_t} \sum_{j \in Y_0} I_{zt,j}$$  |
|`I0`| $$I_0 = \|I_{z0}\|$$ |
|`U0`|  Current-weighted average emission voltage, in kilovolts. $$U_0 = \frac{\sum_{j \in Y_0} I_{zt,j} U_{t,j}}{I_{z0}} $$ |
