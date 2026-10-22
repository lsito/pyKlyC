from dataclasses import dataclass
import numpy as np

from constants import c, etom
from model.beam import BeamParameters


@dataclass(frozen=True)
class ReferenceParameters:
    """
    Reference beam quantities and normalized entrance conditions.

    Parameters
    ----------
    Iz0:
        Signed average emission current, in amperes. Given by the average in the
        RF period of the sum of the emission currents (Izt) associated with all 
        (Nt) emitted samples. 

    I0:
        Magnitude of the average emission current (Iz0), in amperes.

    U0:
        Current-weighted average emission voltage, in kilovolts.

    s:
        Reference beam velocity divided by the speed of light.

    vetoc:
        Velocity divided by the speed of light for every emitted sample.

    vn1:
        Reference velocity divided by each sample velocity at the entrance.

    gama:
        Lorentz factor associated with U0.
    """

    Iz0: float
    I0: float
    U0: float
    s: float
    vetoc: np.ndarray
    vn1: np.ndarray
    gama: float

    @classmethod
    def calculate_reference_parameters(
        cls,
        beam: BeamParameters,
    ) -> "ReferenceParameters":
        """
        Calculate current-weighted beam reference quantities.

        Parameters
        ----------
        beam:
            Beam emission and geometry parameters.
        """

        Y0 = beam.Y0
        Nt = beam.Nt
        Izt = beam.Izt
        Ut = beam.Ut

        Iz0 = np.sum(Izt[Y0]) / Nt

        if Iz0 == 0.0:
            raise ValueError(
                "The average signed emission current Iz0 is zero."
            )

        I0 = abs(Iz0)
        U0 = np.sum(Izt[Y0] * Ut[Y0]) / Nt / Iz0

        s = np.sqrt(
            1.0 - (1.0 + etom * U0 * 1.0e3 / c**2) ** -2
        )

        vetoc = np.sqrt(
            1.0 - (1.0 + etom * Ut * 1.0e3 / c**2) ** -2
        )

        vn1 = np.zeros_like(vetoc)
        vn1[Y0] = s / vetoc[Y0]

        gama = 1.0 + etom * U0 * 1.0e3 / c**2

        return cls(
            Iz0=Iz0,
            I0=I0,
            U0=U0,
            s=s,
            vetoc=vetoc,
            vn1=vn1,
            gama=gama,
        )
