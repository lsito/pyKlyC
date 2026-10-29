from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class ConvergenceRecord:
    """
    Residual information from one nonlinear iteration.

    Parameters
    ----------
    iteration:
        Iteration number.

    residential:
        Residual associated with the RF gap voltage convergence.

    residentialel:
        Residual associated with the electron-efficiency convergence.

    effend:
        Beam energy-extraction efficiency sampled at the observation plane.
        This is not necessarily the efficiency at the last grid point, nor
        the useful RF output efficiency.

    electron_status:
        Status of the particle motion during this iteration.
        Typical values are ``"normal"``, ``"reflected"``, or ``"trapped"``.
    """

    iteration: int
    residential: float
    residentialel: float
    effend: float
    electron_status: str = "normal"

    @classmethod
    def calculate(cls, nn, numerical, grid, beam_state, cavity_state, dynamics, effendflag):
        """Evaluate one iteration without deciding convergence (MATLAB 532--534).

        Call after the circuit solve, tracking and BeamState.update. The
        current voltage is cavity_state.Vnc; flagVnc is the comparison voltage
        managed by IterationControl. effendflag is the previous comparison
        efficiency from that control object (initially zero in MATLAB).

        Comparison values are NOT changed here. The outer loop must decide
        whether to retain the current values or reset them after switching
        space-charge methods. History.last is not automatically the comparison
        reference, because a mode switch can require a reset.

        Voltage residuals use complex differences, detecting phase as well as
        amplitude changes, divided by CURRENT voltage magnitudes. The efficiency
        residual also uses the CURRENT efficiency in its denominator. Neither
        residual is a particle-integration error or an energy-balance error.

        No denominator floors are inserted: nonzero/zero gives inf and 0/0
        gives nan. MATLAB max omits nan entries, so nanmax matches its cavity
        reduction (all-nan stays nan). No convergence decision is made here.
        """

        # Exclude the prescribed downstream length before sampling efficiency.
        # interp1 returns nan out of range; np.interp would otherwise clamp
        # silently to an endpoint, so specify both out-of-range values.
        zi = grid.zi
        effe = beam_state.effe
        effend = np.interp(
            np.max(zi) - numerical.efficiency_end_offset_mm,
            zi, effe, left=np.nan, right=np.nan,
        )

        Vnc = cavity_state.Vnc
        flagVnc = cavity_state.flagVnc
        residential = np.nanmax(np.abs(flagVnc - Vnc) / np.abs(Vnc))
        residentialel = np.abs((effendflag - effend) / effend)

        # Match MATLAB's plotting labels (547--552). "reflected" includes
        # discarded trajectories; "trapped" means a finite final stored
        # position, not proof of permanent physical trapping. This is a
        # diagnostic label only, not an additional convergence criterion.
        if dynamics.Yflag == len(dynamics.Y):
            electron_status = "normal"
        elif np.sum(np.abs(1.0 / dynamics.Zn[-1, dynamics.Y])) == 0:
            electron_status = "reflected"
        else:
            electron_status = "trapped"

        return cls(
            iteration=nn,
            residential=residential,
            residentialel=residentialel,
            effend=effend,
            electron_status=electron_status,
        )

    @property
    def log_residential(self) -> float:
        """
        Base-10 logarithm of the RF gap-voltage residual.
        """

        return math.log10(max(self.residential, 1e-300))

    @property
    def log_residentialel(self) -> float:
        """
        Base-10 logarithm of the electron-efficiency residual.
        """

        return math.log10(max(self.residentialel, 1e-300))


@dataclass
class ConvergenceHistory:
    """
    Sequence of convergence records accumulated during the nonlinear solve.
    """

    records: list[ConvergenceRecord] = field(default_factory=list)

    def append(self, record: ConvergenceRecord) -> None:
        """
        Store one nonlinear iteration record.
        """

        self.records.append(record)

    @property
    def last(self) -> ConvergenceRecord | None:
        """
        Most recent convergence record.
        """

        if not self.records:
            return None
        return self.records[-1]


@dataclass
class IterationControl:
    """End-of-iteration decisions and comparison efficiency (MATLAB 601--622).

    Create once per nonlinear solve, alongside a fresh CavityState whose
    flagVnc starts at zero. Unlike ConvergenceHistory, these values are working
    control state, not a record of every calculated result.

    effendflag is the efficiency reference for the NEXT residual calculation.
    The matching voltage reference lives in cavity_state.flagVnc. Only update
    below changes these references: a circuit solve must not overwrite them.

    iterationsuc follows MATLAB: 1 for successful convergence, otherwise 0.
    stop tells the future outer loop to break. reason is one of "continue",
    "switch_to_global", "converged", "electron_loss", or "iteration_limit".
    The outer solver can also set stop=True and reason="user_abort" if its
    optional convergence window is closed before the decision step.
    Switching space-charge methods does not mean convergence or termination.
    """

    effendflag: float = 0.0
    iterationsuc: int = 0
    stop: bool = False
    reason: str = "continue"

    def update(
        self,
        record,
        numerical,
        cavity_state,
        space_charge_state,
        dynamics,
        *,
        drivefactor=1.0,
        optimization_mode=False,
    ):
        """Decide after calculating/storing this iteration's residuals.

        drivefactor is the SAME voltage-amplitude multiplier used by the
        cavity solve, not its power fraction. It is 1 for ordinary operation;
        during power ramping convergence is permitted only at full drive.
        The ramp schedule and skipped iterations are not implemented here.

        optimization_mode corresponds to MATLAB moviedisp(4) < 0. In that
        mode, any electron not counted as exiting downstream terminates the
        run before the convergence test. This includes exhausted trajectories,
        not just reflected electrons. Outside optimization mode MATLAB does
        not impose this electron-exit requirement on voltage convergence.

        Call once per evaluated iteration and stop calling when stop is True.
        This method neither calculates new fields nor appends history records.
        It preserves MATLAB's strict inequalities and uses the voltage residual
        only; residentialel remains diagnostic, not an extra stop criterion.
        """

        residential = record.residential
        Rlimit = numerical.Rlimit
        evolutionmethod = numerical.evolutionmethod
        SCmethod = space_charge_state.SCmethod
        self.reason = "continue"

        # 601--603: optimization rejects a pass with missing downstream exits,
        # even if its cavity voltages appear converged. Keep references intact.
        if optimization_mode and dynamics.Yflag != len(dynamics.Y):
            self.stop = True
            self.reason = "electron_loss"
            return

        # 604--614: a settled GLOBAL solution uses Rlimit. A deliberately
        # LOCAL-only solution is allowed when its threshold is <= Rlimit,
        # and must meet that (possibly tighter) threshold. In either case,
        # convergence at an intermediate ramp power is not final convergence.
        global_converged = SCmethod == 1 and residential < Rlimit
        local_converged = (
            SCmethod == 0
            and evolutionmethod[1] <= Rlimit
            and residential < evolutionmethod[1]
        )
        if (global_converged or local_converged) and drivefactor == 1.0:
            self.iterationsuc = 1
            self.stop = True
            self.reason = "converged"
            return

        # 615--618: the local approximation has settled enough to start the
        # global space-charge model. It takes effect on the NEXT field update.
        # Reset both references, not the actual cavity voltages or beam state:
        # do not mistake agreement with the old model for global convergence.
        # MATLAB allows this transition before the power ramp reaches full drive.
        if (
            SCmethod == 0
            and evolutionmethod[1] > Rlimit
            and residential < evolutionmethod[1]
        ):
            space_charge_state.SCmethod = 1
            cavity_state.flagVnc[:] = 0.0
            self.effendflag = 0.0
            self.reason = "switch_to_global"
        else:
            # 619--621: copy the current comparison values for the next pass.
            # Slice assignment avoids aliasing flagVnc to the mutable Vnc array.
            cavity_state.flagVnc[:] = cavity_state.Vnc
            self.effendflag = record.effend

        # MATLAB reaches the end of for nn=1:Nlimit without iterationsuc=1.
        # Check AFTER the normal reference/mode updates, matching that order.
        # Convergence at the last permitted iteration was already handled above.
        if record.iteration >= numerical.Nlimit:
            self.stop = True
            self.reason = "iteration_limit"
