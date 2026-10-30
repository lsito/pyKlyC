"""Build and solve the manual's five-cavity example with the current 1D/GS model.

Run from any directory: python path/to/8.ApplegateDiagramTest/manual_case.py
No test 7 checkpoint is loaded. Outputs go to manual_case_results/ here.

Sources: UserManual.pdf pp. 3-4 (beam/cavities), p. 8 (resolution), p. 9
(Gaussian option and default extension). KlyC_v6.mlapp supplies the entrance
current, domain construction, Ntimes, and Image C -> fillfactor mapping.

This runs the nonlinear solver with RF cavity feedback reconstructed from Vnc.
The reconstruction extends the supplied MATLAB source's empty mode-field block
using its gap-voltage normalization. The manual also demonstrates EM/2D
operation; this script uses the supported 1D Gaussian option. Numerical
convergence alone does not validate agreement with the manual's EM/2D results.
"""

from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save figures without opening GUI windows.
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

project_directory = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_directory / "src"))

from model.beam import BeamParameters
from model.cavity import CavityParameters
from model.cavity_field_map import CavityFieldMap
from model.settings import NumericalSettings
from model.tube import TubeParameters
from solver.reference import ReferenceParameters
from solver.grid import GridParameters
from solver.cavity_fields import CavityFields
from solver.large_signal_1d import solve
from diagnostics.beam import find_peak_modulation, check_boundary_reaching, check_negative_velocity
from diagnostics.cavity import calculate_cavity_power
from diagnostics.space_charge import calculate_space_charge_power
from diagnostics.power_balance import calculate_power_balance
from plotting.beam import plot_applegate, plot_current_harmonics


def main():
    # 1. Manual pp. 3-4: one 180 kV, 16 A solid beam; 1 GHz RF input at 60 W.
    Nbeam = 1
    Nt = 128
    rm1, rm2, rc = 0.0, 6.0, 10.0
    t0 = 2 * np.pi * np.arange(Nt) / Nt
    tn1 = t0.copy()
    Y0 = np.arange(Nt)
    Ut = np.full(Nt, 180.0)
    Izt = np.full(Nt, -16.0)  # GUI's Pin-enabled case: no imposed bunching.

    Nc = 5
    f = 1.0  # GHz in Python; manual GUI uses 1000 MHz.
    f0 = np.array([1.000, 1.006, 1.016, 1.030, 1.000])
    RtoQ = np.array([182.8959, 116.9999, 102.0570, 123.6318, 180.3864])
    M = np.array([0.9225, 0.8930, 0.8192, 0.9518, 0.9325])
    type = np.array([0, 1, 1, 1, -1])
    nh = np.ones(Nc, dtype=int)
    Qext = np.array([334., 95000., 95000., 95000., 50.])
    Qin = np.array([16428., 19768., 19816., 16970., 14787.])
    zshift = np.array([0., 480., 1630., 2400., 2730.])
    Pin = np.array([60., 0., 0., 0., 0.])
    gaman = np.zeros(Nc, dtype=complex)  # Matched output.
    kai = np.zeros((Nc, Nc), dtype=complex)
    Lsb = np.pi * np.ones(Nc)
    cavityname = "C1 C2 C3 C4 C5"

    # 2. Manual p. 8: recommended accuracy; remaining controls from the GUI.
    Nz, Ns = 256, 8
    Nlimit, Ntimes = 100, 10
    Rlimit, iterfactor = 1e-4, 0.5
    evolutionmethod = np.array([0.005, 2.0])
    factor = 1.0  # Full plasma-frequency / space-charge strength.
    factorregion = 10.0
    fillfactor = -1.0  # Manual p. 3: Image C = -1; GUI passes ConB unchanged.
    DCSPfactor = 1.0
    upstream_discard_distance_mm = 0.0  # GUI default zcut lies outside domain.
    efficiency_end_offset_mm = 3 * rc  # GUI constructs GIFpara(5) this way.
    extensionrc = 2.0  # Default extension setting stated in the manual.
    N = 200  # Gaussian map sampling used by the GUI, independent of Nz.
    Ntp = 64  # Plotting choice only; does not change the computed solution.

    beam = BeamParameters(
        Nbeam=Nbeam, rm1=rm1, rm2=rm2, rc=rc, xb=0., yb=0., xd=0., yd=0.,
        Nt=Nt, t0=t0, tn1=tn1, Y0=Y0, Ut=Ut, Izt=Izt,
    )
    numerical = NumericalSettings(
        f=f, Nc=Nc, Ns=Ns, Nz=Nz, Nlimit=Nlimit, Ntimes=Ntimes,
        Rlimit=Rlimit, iterfactor=iterfactor, evolutionmethod=evolutionmethod,
        upstream_discard_distance_mm=upstream_discard_distance_mm,
        efficiency_end_offset_mm=efficiency_end_offset_mm,
    )
    cavities = CavityParameters(
        type=type, nh=nh, f0=f0, Qext=Qext, Qin=Qin, zshift=zshift, Pin=Pin,
        gaman=gaman, Lsb=Lsb, kai=kai, cavityname=cavityname,
    )
    reference = ReferenceParameters.calculate_reference_parameters(beam)
    maps = CavityFieldMap.from_circuit_parameters(
        cavities.names, f0, RtoQ, M, reference.s, N,
    )

    # 3. GUI's GS-domain rule: include every map, then add entrance/exit margins.
    zminarray = np.array([maps[name].z_mm.min() for name in cavities.names]) + zshift
    zmaxarray = np.array([maps[name].z_mm.max() for name in cavities.names]) + zshift
    z0 = np.min(zminarray) - 2 * rc
    L = np.max(zmaxarray) - z0 + (4 + extensionrc) * rc
    rcv = np.array([[z0, z0 + L], [1., 1.]])
    tube = TubeParameters(z0=z0, L=L, rcv=rcv)
    grid = GridParameters.calculate(numerical, cavities, tube, beam, reference, factorregion)
    cavity_fields = CavityFields.calculate(numerical, cavities, grid, reference, maps)

    output_directory = Path(__file__).resolve().parent / "manual_case_results"
    output_directory.mkdir(exist_ok=True)
    model_note = (
        "Manual parameters, 1D/GS solver with RF cavity feedback: "
        "Emz=sum(Vnc*ez/(betae*sqrt(k0))) by harmonic. This reconstruction "
        "extends the supplied MATLAB source. Emz0=0 for RF-only cavities. "
        "An EM/2D manual-result comparison remains separate validation."
    )
    print(model_note, flush=True)
    print(f"Grid: {len(grid.z)} planes, dz={grid.dz:.6f} mm; "
          f"domain [{z0:.6f}, {z0 + L:.6f}] mm", flush=True)
    print(f"Nt={Nt}, Nz={Nz}, Ns={Ns}, Nlimit={Nlimit}; "
          f"factor={factor}, Image C / fillfactor={fillfactor}", flush=True)

    # Save a complete input snapshot under separate names from the paired test.
    savemat(output_directory / "manual_case_inputs.mat", dict(
        Nbeam=Nbeam, Nt=Nt, rm1=rm1, rm2=rm2, rc=rc, t0=t0, tn1=tn1,
        Y0=Y0, Ut=Ut, Izt=Izt, Nc=Nc, f=f, f0=f0, RtoQ=RtoQ, M=M,
        type=type, nh=nh, Qext=Qext, Qin=Qin, zshift=zshift, Pin=Pin,
        gaman=gaman, Lsb=Lsb, kai=kai, cavityname=cavityname,
        Nz=Nz, Ns=Ns, Nlimit=Nlimit, Ntimes=Ntimes, Rlimit=Rlimit,
        iterfactor=iterfactor, evolutionmethod=evolutionmethod,
        factor=factor, factorregion=factorregion, fillfactor=fillfactor,
        DCSPfactor=DCSPfactor, upstream_discard_distance_mm=upstream_discard_distance_mm,
        efficiency_end_offset_mm=efficiency_end_offset_mm,
        N=N, Ntp=Ntp, extensionrc=extensionrc, z0=z0, L=L, rcv=rcv,
        power_ramp_mode=False, mono=1, index_base=0, model_note=model_note,
    ), do_compression=True)

    # 4. Actually execute initialization and every nonlinear pass.
    def report_iteration(record, beam_state, cavity_state, space_charge_state, dynamics):
        print(f"nn={record.iteration:3d}  "
              f"SCmethod={space_charge_state.SCmethod}  "
              f"voltage residual={record.residential:.3e}  "
              f"beam efficiency={record.effend:.6g}", flush=True)

    start = perf_counter()
    result = solve(
        numerical, beam, cavities, grid, reference, cavity_fields,
        factor=factor, fillfactor=fillfactor, DCSPfactor=DCSPfactor,
        on_iteration=report_iteration,
    )
    elapsed = perf_counter() - start

    # 5. Read-only post-processing of this newly computed solution.
    modulation = find_peak_modulation(result.beam_state, result.dynamics)
    boundary = check_boundary_reaching(result.dynamics)
    velocity = check_negative_velocity(result.dynamics, reference)
    cavity_power = calculate_cavity_power(
        cavities, cavity_fields, result.cavity_state, beam, grid, reference,
        drivefactor=result.drivefactors[-1],
    )
    space_charge_power = calculate_space_charge_power(result.beam_state, result.space_charge_state, grid)
    power_balance = calculate_power_balance(
        result.beam_state, space_charge_power, cavity_power,
        numerical, beam, grid, reference,
    )

    # 6. Plot full trajectories and final current harmonics, saving PNG and PDF.
    zmax = grid.zi.max() - efficiency_end_offset_mm
    fig, axes = plot_applegate(
        result.dynamics, beam, grid, cavities, cavity_fields,
        ncenter=modulation["ncenter"], Ntp=Ntp, zmax=zmax,
    )
    axes[0].set_title("Manual parameters - Applegate diagram (current 1D/GS solver)")
    fig.savefig(output_directory / "applegate.png", dpi=180)
    fig.savefig(output_directory / "applegate.pdf")
    plt.close(fig)
    fig, ax = plot_current_harmonics(result.beam_state, grid, cavities, zmax=zmax)
    ax.set_title("Manual parameters - current harmonics (current 1D/GS solver)")
    fig.savefig(output_directory / "current_harmonics.png", dpi=180)
    fig.savefig(output_directory / "current_harmonics.pdf")
    plt.close(fig)

    output = dict(
        model_note=model_note, index_base=0, runtime_seconds=elapsed,
        stop_reason=result.control.reason, iterationsuc=result.control.iterationsuc,
        nn=result.nn, zi=grid.zi, z=grid.z, betae=grid.betae, ez=cavity_fields.ez,
        Vnc=result.cavity_state.Vnc, Vnp=result.cavity_state.Vnp,
        Vnc_history=result.Vnc_history, Vnp_history=result.Vnp_history,
        SCmethods=result.SCmethods, drivefactors=result.drivefactors,
        Emz=result.Emz, Emz0=result.Emz0,
        Esz=result.space_charge_state.Esz, Esz0=result.space_charge_state.Esz0,
        Esmz=result.Esmz, Esmz0=result.Esmz0,
        modulation=modulation, boundary=boundary, velocity=velocity,
        cavity_power=cavity_power, space_charge_power=space_charge_power,
        power_balance=power_balance,
        iterations=np.array([record.iteration for record in result.history.records]),
        residential=np.array([record.residential for record in result.history.records]),
        effend_history=np.array([record.effend for record in result.history.records]),
    )
    for name in ["Iz", "IzDC", "QQ", "QQDC", "effe", "vez", "gamaz"]:
        output[name] = getattr(result.beam_state, name)
    for name in ["Zn", "Tn", "Vn", "Tnz", "Vnz", "Y", "Yflag", "Utn", "ttn"]:
        output[name] = getattr(result.dynamics, name)
    savemat(output_directory / "manual_case_results.mat", output, do_compression=True)

    summary = (
        f"{model_note}\n\n"
        f"Stop reason: {result.control.reason}\n"
        f"Converged flag: {result.control.iterationsuc}\n"
        f"Evaluated iterations: {len(result.history.records)}\n"
        f"Runtime: {elapsed:.2f} s\n"
        f"Grid planes: {len(grid.z)}; dz: {grid.dz:.8g} mm\n"
        f"Domain: [{z0:.8g}, {z0 + L:.8g}] mm\n"
        f"Trajectory diagnostic: {boundary}\n"
        f"Maximum fundamental modulation: {modulation['Izmax']:.8g}\n"
        f"Output-port efficiency: {cavity_power['effcavity']:.8g}\n"
        f"Power-balance efficiency: {power_balance['effendmodified']:.8g}\n"
        f"Maximum cavity tracking field: {np.max(np.abs(result.Emz)):.8g}\n"
    )
    (output_directory / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print("Saved results and plots in", output_directory, flush=True)


if __name__ == "__main__":
    main()
