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
        Relaxation factor used to mix the newly computed beam harmonics
        with the previous iteration values.

    evolutionmethod:
        Two numerical controls, in the same order as MATLAB's vector.
        [0] limits the relative increase in inverse velocity Vnz over one
        spatial step. Exceeding it requests continuation by time tracking.
        [1] is the cavity-voltage residual threshold used by the outer
        iteration to switch from local to global space charge; if it is
        no greater than Rlimit, MATLAB can instead finish in local mode.
        The ordinary GUI setting is np.array([0.005, 2.0]): a 0.5% inverse-
        velocity increase threshold and a residual threshold of 2.0.
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
        Parameters used when generating animated output. MATLAB also uses
        GIFpara[3] as an upstream discard distance in mm during time tracking:
        a backward crossing below min(grid.zi) + GIFpara[3] ends that electron's
        trajectory. It is therefore not solely a display control.

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
