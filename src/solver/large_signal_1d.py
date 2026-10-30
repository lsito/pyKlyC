"""Orchestrate circuit, space-charge and beam-tracking passes for the 1D solve.

Each evaluated iteration uses the previous BeamState to solve the circuit and
fields, tracks a fresh entrance beam, relaxes its new sources, then records
diagnostics and updates iteration control. BeamDynamics is recreated per pass;
the other working states persist for the duration of one solve() call.

Flowchart, sequence diagram and state/control tables:
    docs/large_signal_1d.md (relative to the repository root).
"""

import logging
from dataclasses import dataclass

import numpy as np

from model.beam import BeamParameters
from solver.beam_dynamics import BeamDynamics
from solver.beam_state import BeamState
from solver.cavity_state import CavityState
from solver.space_charge import SpaceChargeState
from solver.convergence import ConvergenceHistory, ConvergenceRecord, IterationControl


logger = logging.getLogger(__name__)


@dataclass
class LargeSignal1DResult:
    """Final working states and diagnostics from one fresh nonlinear solve.

    history contains only evaluated iterations; ramp-skipped iteration numbers
    are in skipped_iterations. Vnc_history, Vnp_history, drivefactors and
    SCmethods have the SAME row order as history.records. SCmethods records the
    method USED in each pass, before a possible end-of-pass switch.

    nn is the last attempted MATLAB iteration number, not the history length.
    control.stop/reason/iterationsuc describe termination. The RF cavity field
    is reconstructed from Vnc on every evaluated pass; Emz0 remains zero because
    this entry point supports RF cavities only.

    Fields and trajectories belong to the last evaluated pass. BeamState has
    already been relaxed using that pass; no extra field solve is performed
    after stopping. All returned mutable states are independent of other runs.

    Emz/Emz0 are the cavity RF/DC contributions; Esmz/Esmz0 are the tracking
    RF/DC totals. The separate space-charge fields remain available as
    space_charge_state.Esz and space_charge_state.Esz0.
    """

    beam_state: BeamState
    cavity_state: CavityState
    space_charge_state: SpaceChargeState
    dynamics: BeamDynamics
    history: ConvergenceHistory
    control: IterationControl
    nn: int
    Vnc_history: np.ndarray
    Vnp_history: np.ndarray
    drivefactors: np.ndarray
    SCmethods: np.ndarray
    skipped_iterations: np.ndarray
    Emz: np.ndarray
    Emz0: np.ndarray
    Esmz: np.ndarray
    Esmz0: np.ndarray


def calculate_drivefactor(nn, numerical, Npin, power_ramp_mode):
    """MATLAB's staircase input-voltage multiplier for iteration nn.

    Each step sets a POWER fraction; circuit excitation needs its square root.
    The schedule reaches full power approximately halfway through Nlimit.
    In ordinary operation the input drive remains at its prescribed full value.
    """

    if not power_ramp_mode:
        return 1.0
    return np.sqrt(min((np.floor(2 * nn / numerical.Nlimit * Npin) + 1) / Npin, 1.0))


def solve(
    numerical,
    beam,
    cavities,
    grid,
    reference,
    cavity_fields,
    *,
    factor=1.0,
    fillfactor=1.0,
    DCSPfactor=1.0,
    power_ramp_mode=False,
    Npin=1,
    optimization_mode=False,
    plotter=None,
    mono=1,
    on_iteration=None,
):
    """Run the 1D nonlinear iteration using the currently translated physics.

    Supply the already constructed grid, reference beam and normalized cavity
    profiles. This function initializes its OWN beam, cavity, prefill/space-
    charge and convergence states, then executes MATLAB's nn=1:Nlimit loop.
    factor scales the plasma-frequency calculation; fillfactor and DCSPfactor
    have their existing SpaceChargeState meanings. Nlimit and Npin must be
    positive; Npin matters only when power_ramp_mode is True.

    Power-ramp and optimization controls correspond to abs(moviedisp(4))==2
    and moviedisp(4)<0, respectively. They are explicit inputs, independent of
    plotting. An optional ConvergencePlotter receives each record; closing its
    window requests termination at MATLAB's post-diagnostic display check.
    The solver itself opens no window and writes no files; a supplied plotter
    or diagnostic callback may provide those side effects explicitly.

    on_iteration is an optional diagnostic callback, called once per evaluated
    pass after BeamState.update and diagnostic recording, but BEFORE the control
    decision. Its arguments are (record, beam_state, cavity_state,
    space_charge_state, dynamics). The space-charge mode is the one USED in
    this pass. Treat these live states as read-only; copy arrays to retain a
    snapshot. Skipped ramp iterations do not call it. Callback errors propagate.

    The supplied KLJCmainsv22.m leaves its RF-field assignment empty. This loop
    reconstructs Emz from Vnc * ez / (betae * sqrt(k0)), consistent with the
    source's gap-voltage normalization. This closes the cavity-to-beam coupling
    and is an explicit extension to that source. Emz0 remains zero for RF gaps.

    Supported scope: cylindrical beams, ordinary klystron frequency (mono=1),
    RF input/bunching/output cavities. Monotron frequency iteration, DC gaps,
    traveling-wave and sheet-beam branches are not implemented. Final RF-power
    accounting, corrected ramp-efficiency tables and animations are separate
    post-processing, not calculated by this loop.
    """

    _validate_beam_type(beam)
    if mono != 1:
        raise NotImplementedError("Monotron frequency iteration is not implemented.")
    if np.any(~np.isin(cavities.type, [-1, 0, 1])) or np.any(cavities.nh < 1):
        raise NotImplementedError("Only standing-wave RF input, bunching and output cavities are supported.")
    if numerical.Nlimit < 1 or (power_ramp_mode and Npin < 1):
        raise ValueError("Nlimit and the number of power-ramp steps must be positive.")

    beam_state = BeamState.initialize(numerical, beam, grid, reference)
    cavity_state = CavityState.initialize(numerical, cavities, beam, reference)
    space_charge_state = SpaceChargeState.initialize(
        beam, grid, reference, beam_state, factor, fillfactor, DCSPfactor,
    )
    history = ConvergenceHistory()
    control = IterationControl()
    Vnc_history = []
    Vnp_history = []
    drivefactors = []
    SCmethods = []
    skipped_iterations = []
    for nn in range(1, numerical.Nlimit + 1):
        drivefactor = calculate_drivefactor(nn, numerical, Npin, power_ramp_mode)

        # MATLAB 214--216: if a ramp step has settled, wait for the next power
        # step without another circuit or beam pass. nn still advances, but
        # the last residual, states and comparison references do not change.
        if (
            nn > 1
            and power_ramp_mode
            and history.last.residential < numerical.Rlimit
            and drivefactor == calculate_drivefactor(nn - 1, numerical, Npin, True)
        ):
            skipped_iterations.append(nn)
            continue

        # 1. Solve the circuit using the PREVIOUS beam estimate. Do not update
        # flagVnc here: IterationControl owns its retention/reset semantics.
        cavity_state.solve(
            numerical, cavities, cavity_fields, beam, beam_state, grid,
            drivefactor=drivefactor,
        )

        # 2. Reconstruct the current cavity fields from the solved amplitudes.
        # Explicit extension of MATLAB's empty mode-field block (310-313),
        # using its gap-voltage normalization. Includes input drive on nn=1.
        Emz, Emz0 = cavity_fields.calculate_mode_fields(
            numerical, cavities, grid, cavity_state,
        )

        # MATLAB 314--347: update the separate space-charge fields.
        # Space charge is zero on nn=1; later it uses the selected local/global
        # model and the beam estimate from the previous nonlinear iteration.
        SCmethods.append(space_charge_state.SCmethod)
        space_charge_state.update(
            nn, beam, grid, reference, beam_state, fillfactor, DCSPfactor,
        )

        # 3. A nonlinear iteration re-tracks the SAME entrance beam in the new
        # prescribed fields; it does not continue the last iteration's particle
        # histories. The spatial-to-time handoff within this pass is different:
        # track() preserves those histories and crossing sums when it switches.
        dynamics = BeamDynamics.initialize(numerical, beam, grid, reference)
        Esmz, Esmz0 = dynamics.track(
            numerical, beam, grid, reference,
            Esz=space_charge_state.Esz, Esz0=space_charge_state.Esz0,
            Emz=Emz, Emz0=Emz0, nn=nn, Iz=beam_state.Iz,
        )

        # 4. Mix new beam sources, then refresh the energy/speed profiles. These
        # become the beam estimate for the NEXT circuit and space-charge solve.
        beam_state.update(numerical, beam, reference, dynamics)
        record = ConvergenceRecord.calculate(
            nn, numerical, grid, beam_state, cavity_state, dynamics, control.effendflag,
        )
        history.append(record)
        Vnc_history.append(cavity_state.Vnc.copy())
        Vnp_history.append(cavity_state.Vnp.copy())
        drivefactors.append(drivefactor)
        logger.info("Iteration %s: effend=%s, voltage residual=%s, efficiency residual=%s",
                    nn, record.effend, record.residential, record.residentialel)

        if on_iteration is not None:
            on_iteration(record, beam_state, cavity_state, space_charge_state, dynamics)

        # MATLAB checks the display before the convergence/mode decisions.
        # A closed window is an abort, never a successful convergence result.
        if plotter is not None:
            if not plotter.is_open():
                control.stop = True
                control.reason = "user_abort"
                break
            plotter.update(record)

        # 5. Stop, switch the space-charge model for the next pass, or retain
        # comparison values. Local-to-global switching does not reset the beam.
        control.update(
            record, numerical, cavity_state, space_charge_state, dynamics,
            drivefactor=drivefactor, optimization_mode=optimization_mode,
        )
        if control.stop:
            break

    # Ramp skips can reach Nlimit without calling control.update on that last
    # iteration. MATLAB then exits its for-loop with iterationsuc still zero.
    if not control.stop:
        control.stop = True
        control.reason = "iteration_limit"

    return LargeSignal1DResult(
        beam_state=beam_state, cavity_state=cavity_state,
        space_charge_state=space_charge_state, dynamics=dynamics,
        history=history, control=control, nn=nn,
        Vnc_history=np.array(Vnc_history), Vnp_history=np.array(Vnp_history),
        drivefactors=np.array(drivefactors), SCmethods=np.array(SCmethods),
        skipped_iterations=np.array(skipped_iterations, dtype=int),
        Emz=Emz, Emz0=Emz0, Esmz=Esmz, Esmz0=Esmz0,
    )


def _validate_beam_type(beam: BeamParameters) -> None:
    """
    Validate the beam geometry selected for the one-dimensional solver.

    The current implementation supports cylindrical beams. Sheet-beam
    parameters are recognized but their numerical branches are not yet
    implemented.
    """

    if beam.is_cylindrical:
        logger.info(
            "Cylindrical beam simulation will be started shortly."
        )
        return

    if beam.is_sheet:
        raise NotImplementedError(
            "The sheet-beam branches in KLJCmainsv22 are incomplete."
        )

    raise ValueError(
        "The beam type is inconsistent: select either a nonzero rc with "
        "yd equal to zero, or rc equal to zero with nonzero yd."
    )
