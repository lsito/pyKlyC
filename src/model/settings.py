from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NumericalSettings:
    """
    Numerical resolution and nonlinear iteration controls.

    Parameters
    ----------
    f:
        Operating frequency in gigahertz.

    Nc:
        Number of RF cavities or RF interaction structures.

    Ns:
        Number of RF harmonics used in the beam current, charge-density,
        RF field, and space-charge calculations.

    Nz:
        Number of longitudinal mesh divisions per electronic wavelength.

    Nlimit:
        Maximum number of nonlinear self-consistent iterations.

    Ntimes:
        Maximum time-extension factor used when electrons slow down,
        reverse direction, or remain inside the simulated region longer
        than one RF period.

    Rlimit:
        Convergence threshold for the nonlinear iteration residual.

    iterfactor:
        Its absolute value is the relaxation weight used to mix newly computed
        beam currents/charges with the previous iteration values. Its sign
        separately controls the optional tracking space-charge correction:
        a nonpositive value allows that correction after iteration 5.

    evolutionmethod:
        Two numerical controls, in the same order as MATLAB's vector.
        [0] limits the relative increase in inverse velocity Vnz over one
        spatial step. Exceeding it requests continuation by time tracking.
        [1] is the cavity-voltage residual threshold used by the outer
        iteration to switch from local to global space charge; if it is
        no greater than Rlimit, MATLAB can instead finish in local mode.
        The ordinary GUI setting is np.array([0.005, 2.0]): a 0.5% inverse-
        velocity increase threshold and a residual threshold of 2.0.

    upstream_discard_distance_mm:
        Distance from the simulation entrance to the reflected-electron
        cutoff, in mm. A backward crossing with its lower grid plane below
        min(grid.zi) + upstream_discard_distance_mm ends that trajectory.
        Use 0.0 to retain tracking up to upstream escape. This is a distance,
        not the absolute coordinate entered in MATLAB's GUI "zcut (mm)".
        It replaces GIFpara(4) for tracking; plotting remains independent.

    efficiency_end_offset_mm:
        Distance upstream from max(grid.zi) at which the beam efficiency is
        sampled for convergence. The observation coordinate is therefore
        max(grid.zi) - efficiency_end_offset_mm, in mm. Zero samples the last
        grid plane. This replaces MATLAB's GIFpara(5); the GUI normally uses
        3*beam.rc. An observation outside the grid gives nan, as with interp1.
    """

    f: float
    Nc: int
    Ns: int
    Nz: int
    Nlimit: int
    Ntimes: int
    Rlimit: float
    iterfactor: float
    evolutionmethod: np.ndarray
    upstream_discard_distance_mm: float
    efficiency_end_offset_mm: float


@dataclass(frozen=True)
class DisplaySettings:
    """
    Plotting, movie, and diagnostic-output controls.

    Parameters
    ----------
    figuredisp:
        Controls which diagnostic figures are generated after the simulation.

    moviedisp:
        Vector of runtime display and movie-generation flags.

    GIFpara:
        Parameters used when generating animated output. GIFpara[3] sets the
        plot's upstream cut distance in mm. Python tracking instead uses
        NumericalSettings.upstream_discard_distance_mm. Original MATLAB uses
        GIFpara(4) for both; when exporting tracking inputs to MATLAB, put the
        numerical discard distance in that entry to match Python's behavior.
        Likewise, export NumericalSettings.efficiency_end_offset_mm as
        GIFpara(5) to match the efficiency observation plane. In Python the
        plot's end cut GIFpara[4] does not control convergence calculations.

    Npin:
        Number of input-power sweep points when the power-ramp mode is active.
    """

    figuredisp: np.ndarray
    moviedisp: np.ndarray
    GIFpara: np.ndarray
    Npin: int

    @property
    def show_convergence_window(self) -> bool:
        """
        Whether the interactive convergence-history window should be shown.
        """

        return int(self.moviedisp[4]) != 1

    @property
    def power_ramp_mode(self) -> bool:
        """
        Whether the y-axis of the convergence plot should allow efficiency-like
        values above zero.
        """

        return abs(float(self.moviedisp[3])) == 2
