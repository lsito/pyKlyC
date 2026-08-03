from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NumericalSettings:
    """
    Numerical resolution and nonlinear iteration controls.

    Parameters
    ----------
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
    """

    Nc: int
    Ns: int
    Nz: int
    Nlimit: int
    Ntimes: int
    Rlimit: float
    iterfactor: float


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
        Parameters used when generating animated output.

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