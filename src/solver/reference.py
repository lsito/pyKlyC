from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

from model.beam import BeamParameters


FloatArray = npt.NDArray[np.float64]


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
    vetoc: FloatArray
    vn1: FloatArray
    gama: float