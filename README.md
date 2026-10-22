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
| `f`              | Operating frequency, GHz.                                                                                                                       |
| `mono`           | Mode flag. `1` is normal klystron simulation; `0` activates monotron-frequency logic.                                                           |

#### Plot/Output controls (in src/model/settings.py --> DisplaySettings dataclass):
| MATLAB parameter | Meaning                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `figuredisp`     | `'on'` or `'off'`; controls static plots.                                                                        |
| `moviedisp`      | Vector of flags controlling GIFs, power ramp, trajectory saving, and convergence display. Eg. [0 0 0 0 1] display convergence history.                       |
| `GIFpara`        | Plot/animation parameters: number of plotted trajectories, min/max velocity axis, z-cut, end cut, velocity type. |
| `Npin`           | Number of input-power ramp points.                                                                    |

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
