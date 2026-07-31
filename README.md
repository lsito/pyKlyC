# pyKlyC
Python computation tool for 1/1.5D klystron simulations.

At the moment, the idea would be:

pyklyc/
  constants.py
  units.py

  model/
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