from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BeamParameters:
    """
    Electron-beam geometry, emission sampling, and entrance conditions.

    Parameters
    ----------
    Nbeam:
        Number of identical electron beams represented by the simulation.

    rm2:
        Outer beam radius in millimetres.

    rm1:
        Inner beam radius in millimetres. Zero represents a solid beam.

    rc:
        Cylindrical drift-tube radius in millimetres. A nonzero value
        selects cylindrical-beam calculations.

    xb:
        Sheet-beam width parameter.

    yb:
        Sheet-beam height parameter.

    xd:
        Sheet drift-region width parameter.

    yd:
        Sheet drift-region height parameter. A nonzero value with rc equal
        to zero selects the sheet-beam branch.

    Nt:
        Number of emitted beam samples during one RF period.

    t0:
        Departure phases of all emitted samples, in radians.

    Ut:
        Emission voltage of each sample, in kilovolts.

    Izt:
        Emission current associated with each sample, in amperes.

    tn1:
        Initial normalized time or phase assigned to each sample at the
        simulation entrance.

    Y0:
        Active emitted disk indices. In normal klystron mode this is [0:Nt-1];
        in bunched-beam mode, only part of the RF period may be active. 
        These indices must be zero-based in Python.
    """

    Nbeam: int
    rm2: float
    rm1: float
    rc: float
    xb: float
    yb: float
    xd: float
    yd: float
    Nt: int
    t0: np.ndarray
    Ut: np.ndarray
    Izt: np.ndarray
    tn1: np.ndarray
    Y0: np.ndarray

    def __post_init__(self) -> None:
        expected_shape = (self.Nt,)

        for name, value in (
            ("t0", self.t0),
            ("Ut", self.Ut),
            ("Izt", self.Izt),
            ("tn1", self.tn1),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"received {value.shape}."
                )

        if self.Nbeam < 1:
            raise ValueError("Nbeam must be at least one.")

        if np.any(self.Y0 < 0) or np.any(self.Y0 >= self.Nt):
            raise ValueError("Y0 contains indices outside the emitted samples.")

    @property
    def is_cylindrical(self) -> bool:
        """
        Whether the cylindrical-beam model is selected.
        """

        return self.yd == 0.0 and self.rc != 0.0

    @property
    def is_sheet(self) -> bool:
        """
        Whether the sheet-beam model is selected.
        """

        return self.rc == 0.0 and self.yd != 0.0
