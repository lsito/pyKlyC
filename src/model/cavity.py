from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CavityParameters:
    """
    RF cavity and interaction-structure parameters.

    Parameters
    ----------
    type:
        Structure classification. Values used by this solver include
        -1 for output cavities, 0 for input cavities, 1 for bunching
        cavities, and 2 for traveling-wave structures.

    nh:
        Harmonic number associated with each structure. Zero denotes a
        static gap field.

    f0:
        Resonant frequency of each structure, in gigahertz.

    Qext:
        External quality factor of each structure.

    Qin:
        Intrinsic quality factor of each structure.

    zshift:
        Longitudinal field-map shift or cavity-center coordinate,
        in millimetres.

    Pin:
        Incident RF power assigned to each input structure, in watts.

    gaman:
        Complex reflection coefficient assigned to each output structure.

    Lsb:
        Electrical phase length beta0 * Ls of each cavity-port connection,
        in radians. The MATLAB initialization uses pi for every cavity.

    cavityname:
        Field-map base names, one for each structure.

    kai:
        Complex or real coupling matrix between cavity modes.
    """

    type: np.ndarray
    nh: np.ndarray
    f0: np.ndarray
    Qext: np.ndarray
    Qin: np.ndarray
    zshift: np.ndarray
    Pin: np.ndarray
    gaman: np.ndarray
    Lsb: np.ndarray
    cavityname: str
    kai: np.ndarray

    @property
    def names(self) -> tuple[str, ...]:
        """Return the space-separated MATLAB field-map names."""

        return tuple(self.cavityname.split())

    def validate(self, Nc: int) -> None:
        """
        Validate all cavity arrays against Nc.
        """

        for name, value in (
            ("type", self.type),
            ("nh", self.nh),
            ("f0", self.f0),
            ("Qext", self.Qext),
            ("Qin", self.Qin),
            ("zshift", self.zshift),
            ("Pin", self.Pin),
            ("gaman", self.gaman),
            ("Lsb", self.Lsb),
        ):
            if value.shape != (Nc,):
                raise ValueError(
                    f"{name} must have shape ({Nc},); received {value.shape}."
                )

        if len(self.names) != Nc:
            raise ValueError(f"cavityname must contain {Nc} names.")

        if self.kai.shape != (Nc, Nc):
            raise ValueError(f"kai must have shape ({Nc}, {Nc}).")
