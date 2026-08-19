from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constants import c, eta0

@dataclass(frozen=True)
class CavityFieldMap:
    """
    Local normalized longitudinal RF cavity mode profile.

    Parameters
    ----------
    z_mm:
        Local longitudinal coordinates in millimetres. Here, the origin is at 
        the cavity center.

    e_z:
        Energy-normalized longitudinal electric field in mm**(-3/2).
    """

    z_mm: np.ndarray
    e_z: np.ndarray

    # Just placeholder for the moment.
    @classmethod
    def from_txt(cls, filename: str | Path) -> "CavityFieldMap":
        data = np.loadtxt(filename, dtype=np.float64)
        data = np.atleast_2d(data)

        return cls(
            z_mm=data[:, 0],
            e_z=data[:, 1],
        )

    @classmethod
    def from_circuit_parameters(
        cls,
        cavityname: tuple[str, ...],
        f0: np.ndarray,
        RtoQ: np.ndarray,
        M: np.ndarray,
        s: float,
        N: int = 200,
    ) -> dict[str, "CavityFieldMap"]:
        
        """Generate the Gaussian approximation for the longitudinal electric 
        field profile of a cavity mode from its circuit parameters."""

        if np.any(M <= 0):
            raise ValueError(
                "The cavity mode coupling coefficient M must be positive."
            )

        # Compute kappa from Eq. 5.2b
        w0 = 2.0 * np.pi * f0 * 1.0e9 # rad/s
        k0 = w0 / c # 1/m
        k = 0.5 * w0 / s / c / np.sqrt(-np.log(M)) # 1/m

        # What in Eq. 43 we called |A| (amplitude of the gaussian)
        ezm = k * np.sqrt(RtoQ / eta0 / np.pi * k0) * 1000.0 ** (-3/2) # mm**(-3/2)

        # Local longitudinal coordinates in millimetres. Here, the origin is at 
        # the cavity center. It will be src/solbver/cavity_fields.py that will 
        # place the cavity in the correct position along the longitudinal grid.
        z = np.outer(np.linspace(-3.0, 3.0, N + 1), 1.0 / k) * 1000.0 # mm

        # Energy normalized longitudinal electric field in mm**(-3/2).
        ez = ezm * np.exp(-(k * 1.0e-3 * z) ** 2)

        # We return a dictionary of CavityFieldMap objects, one for each cavity 
        # name.
        return {
            cavityname[kk]: cls(
                z_mm=z[:, kk],
                e_z=ez[:, kk],
            )
            for kk in range(len(cavityname))
        }
