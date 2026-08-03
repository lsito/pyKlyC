from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TubeParameters:
    """
    Longitudinal simulation region and drift-tube aperture profile.

    Parameters
    ----------
    z0:
        Longitudinal coordinate of the simulation entrance, in millimetres.

    L:
        Total simulated longitudinal length, in millimetres.

    rcv:
        Two-row aperture profile. The first row contains longitudinal
        coordinates in millimetres. The second row contains tunnel-radius
        values normalized to rc.
    """

    z0: float
    L: float
    rcv: FloatArray

    def __post_init__(self) -> None:
        if self.L <= 0.0:
            raise ValueError("L must be positive.")

        if self.rcv.ndim != 2 or self.rcv.shape[0] != 2:
            raise ValueError("rcv must have shape (2, N).")

        if self.rcv.shape[1] < 2:
            raise ValueError("rcv must contain at least two profile points.")

        if np.any(np.diff(self.rcv[0]) <= 0.0):
            raise ValueError(
                "The first row of rcv must be strictly increasing."
            )