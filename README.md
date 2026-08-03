# pyKlyC
Python computation tool for 1/1.5D klystron simulations.

At the moment, the idea would be:

pyklyc/
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

  tests/
    test_reference_cases.py

## 1. We start with a 1D klystron code with Gaussian cavity fields.
We mean no GUI, no EM solver, no Optimizer, no CGUN, no magnetostatic solver.
The idea is to start porting:
    - KLJCmainsv22.m --> 1D solver
    - Klyc_v6.mlapp --> GUI and IO mapping (we focus on the IO mapping)



### KLJCmainsv22.m inputs:
We need first to understand how we want to build the datamodels.

#### Numerical controls
| MATLAB parameter | Meaning                                                                                               | Python                                              |
| ---------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `Nc`             | Number of cavities / RF structures.                                                                   | `Circuit.n_cavities`                                    |
| `Ns`             | Number of harmonics used for current, charge, and space-charge fields.                                | `SolverSettings.n_harmonics`                            |
| `Nz`             | Number of spatial divisions per electronic wavelength. Used to choose `dz`.                           | `GridSettings.points_per_electronic_wavelength`         |
| `Nlimit`         | Maximum number of nonlinear iterations.                                                               | `SolverSettings.max_iterations`                         |
| `Ntimes`         | Extra trajectory-storage/evolution factor. The arrays `Zn`, `Vn`, `Tn` have length `Ntimes * len(z)`. | `TrajectorySettings.max_rf_periods` or remove from core |
| `Rlimit`         | Convergence tolerance for cavity voltage residual.                                                    | `SolverSettings.residual_tol`                           |
| `Nt`             | Number of emitted disks per RF period.                                                                | `BeamDiscretization.n_phase_particles`                  |
| `Npin`           | Number of input-power ramp points.                                                                    | `PowerRampSettings.n_points`                            |

#### Space-charge and iteration controls:
| MATLAB parameter  | Meaning                                                                                                                                             | Python home                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `factor`          | Space-charge multiplier. `0` disables space charge; `1` nominally enables it.                                                                       | `SpaceChargeSettings.scale`                |
| `factorregion`    | Local space-charge interaction window size, used as `factorregion * rc / 2.40483`.                                                                  | `SpaceChargeSettings.local_window_factor`  |
| `fillfactor`      | Adds pre-filled upstream DC beam contribution to space charge.                                                                                      | `SpaceChargeSettings.upstream_fill_factor` |
| `DCSPfactor`      | Scales DC space-charge component.                                                                                                                   | `SpaceChargeSettings.dc_scale`             |
| `iterfactor`      | Iteration relaxation coefficient for updating `Iz`, `QQ`, `IzDC`, `QQDC`.                                                                           | `SolverSettings.relaxation`                |
| `evolutionmethod` | Two-element control vector. First element limits step-size change in fast pass; second controls switching from local to global space-charge method. | Split into named fields                    |
| `kai`             | Cavity-cavity coupling matrix.                                                                                                                      | `Circuit.coupling_matrix`                  |
| `rcv`             | Axial tunnel-radius profile. First row is z, second row is radius normalized to `rc`.                                                               | `Tube.aperture_profile`                    |


#### Geometry:
| MATLAB parameter       | Meaning                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `z0`                   | Start position of simulation domain, mm.                                                         |
| `L`                    | Total simulation length, mm.                                                                     |
| `Nbeam`                | Number of identical beams. For multi-beam klystron scaling.                                      |
| `rm2`                  | Outer beam radius, mm.                                                                           |
| `rm1`                  | Inner beam radius, mm. Zero for solid cylindrical beam.                                          |
| `rc`                   | Drift-tube / beam tunnel radius, mm. If `rc ~= 0`, the code treats the case as cylindrical beam. |
| `xb`, `yb`, `xd`, `yd` | Sheet-beam dimensions. In this 1D solver, sheet-beam branches are mostly empty or inactive.      |

#### Beam Emission Parameters:
| MATLAB parameter | Meaning                                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `t0`             | Emission phase samples over one RF period, usually `0:2π/Nt:2π`.                                                                        |
| `Ut`             | Beam voltage for each emitted disk, kV. Usually constant for klystron mode.                                                             |
| `Izt`            | Beam current carried by each emitted disk/sample, A. Usually negative for electron current.                                             |
| `tn1`            | Initial normalized arrival time/phase. Usually same as `t0`.                                                                            |
| `Y0`             | Active emitted disk indices. In normal klystron mode this is `1:Nt`; in bunched-beam/IOT mode only part of the RF period may be active. |

#### RF/cavity parameters:
| MATLAB parameter | Meaning                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `f`              | Operating frequency, GHz.                                                                                                                       |
| `type`           | Cavity type: `0` input, `1` bunching/intermediate, `-1` output. The code also has partial branches for `2`, probably traveling-wave structures. |
| `nh`             | Harmonic number used by each cavity. Fundamental is `1`; harmonic cavities may be `2`, `3`, etc.                                                |
| `f0`             | Cold resonant frequency of each cavity, GHz.                                                                                                    |
| `Qext`           | External Q. Mainly used for input/output coupling.                                                                                              |
| `Qin`            | Intrinsic/unloaded Q, used for ohmic loss.                                                                                                      |
| `zshift`         | Axial cavity center/field-map shift, mm.                                                                                                        |
| `Pin`            | Input RF power, W. Used for input cavity excitation and power ramp.                                                                             |
| `gaman`          | Complex reflection coefficient at output load.                                                                                                  |
| `cavityname`     | Space-separated list of field-map file names, for example `"KlyCcavity1 KlyCcavity2 ..."`.                                                      |
| `mono`           | Mode flag. `1` is normal klystron simulation; `0` activates monotron-frequency logic.                                                           |

#### Plot/Output controls:
| MATLAB parameter | Meaning                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- |
| `figuredisp`     | `'on'` or `'off'`; controls static plots.                                                                        |
| `moviedisp`      | Vector of flags controlling GIFs, power ramp, trajectory saving, and convergence display.                        |
| `GIFpara`        | Plot/animation parameters: number of plotted trajectories, min/max velocity axis, z-cut, end cut, velocity type. |
