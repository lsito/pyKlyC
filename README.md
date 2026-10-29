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
branches are explicitly unsupported by this entry point. Final RF power/loss
accounting, corrected ramp-efficiency tables and animations are not included.

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
