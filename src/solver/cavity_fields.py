from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.interpolate import interp1d

from constants import eta0
from model.cavity import CavityParameters
from model.settings import NumericalSettings
from solver.grid import GridParameters
from solver.reference import ReferenceParameters


@dataclass(frozen=True)
class CavityFields:
    """Cavity fields sampled and normalized on the longitudinal grid."""

    ez: np.ndarray # Longitudinal electric field per cavity (interpolated on the longitudinal grid).
    C3: np.ndarray # Normalized longitudinal electric field per cavity (interpolated on the longitudinal grid).
    vpz: np.ndarray
    alfaz: np.ndarray #TWT

    @classmethod
    def calculate(
        cls,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        grid: GridParameters,
        reference: ReferenceParameters,
        field_map_directory: str | Path = ".",
    ) -> "CavityFields":
      
        cavities.validate(numerical.Nc)

        shape = (len(grid.zi), numerical.Nc)
        ez = np.zeros(shape)
        C3 = np.zeros(shape)
        vpz = np.zeros(shape)
        alfaz = np.zeros(shape)

        field_map_directory = Path(field_map_directory)

        # Check that for the specific cavity we are looking at, the field
        # harmonic number is less than or equal to the number of harmonics used 
        # in the simulation settings.
        for k, name in enumerate(cavities.names):
            if cavities.nh[k] > numerical.Ns:
                continue

            # Check that we are not in the TWT case
            if cavities.type[k] == 2:
                raise NotImplementedError(
                    "The traveling-wave field branch is empty."
                )

            # Importing the field map made as a 2d array: first column is the
            # longitudinal position and the second column is the longitudinal 
            # electric field values.
            field_map_path = field_map_directory / f"{name}.txt"
            field_map = np.atleast_2d(np.loadtxt(field_map_path))

            if field_map.shape[1] != 2:
                raise ValueError(
                    f"Field map {field_map_path} must contain two columns."
                )

            # Placing the cavities in the correct position along the 
            # longitudinal grid
            positions = field_map[:, 0] + cavities.zshift[k]
            field_values = field_map[:, 1]

            if (
                np.min(positions) <= grid.zi[0] + grid.dz / 10.0
                or np.max(positions) >= grid.zi[-1] - grid.dz / 10.0
            ):
                warnings.warn(
                    "The emission plate is located in the field margin of "
                    f"cavity number {k + 1}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Padding the field map with two points on each side (both of value 
            # 0, we assume the field totally decayied) to avoid extrapolation 
            # errors when interpolating the field values onto the longitudinal 
            # grid. The padding is set to 10% of the grid spacing (dz).
            padding = grid.dz / 10.0
            padded_positions = np.concatenate(
                (
                    [positions[0] - 2.0 * padding, positions[0] - padding],
                    positions,
                    [positions[-1] + padding, positions[-1] + 2.0 * padding],
                )
            )
            padded_fields = np.concatenate(
                ([0.0, 0.0], field_values, [0.0, 0.0])
            )

            # Finally, we interpolate the field values onto the longitudinal 
            # grid using linear interpolation.
            field_interpolator = interp1d(
                padded_positions,
                padded_fields,
                kind="linear",
                fill_value="extrapolate",
            )
            # Now the impoerted ez is in [kV/mm] if DC and in [(mm)^(-3/2)] if RF
            ez[:, k] = field_interpolator(grid.zi)

            # We normalize the field values according to the harmonic number of 
            # the cavity. For static gaps (nh=0), we use a different 
            # normalization than for cavities with a non-zero harmonic number.
            # So C3 is in [kV/mm]*[mm/kV] which is dimensionless.
            if cavities.nh[k] == 0:
                C3[:, k] = ez[:, k] / grid.betae / reference.U0
            else:
                C3[:, k] = (
                    ez[:, k]
                    * eta0
                    * reference.I0
                    / np.sqrt(grid.k0[k])
                    / grid.betae
                    / reference.U0
                    / 1.0e3
                )

        return cls(
            ez=ez,
            C3=C3,
            vpz=vpz,
            alfaz=alfaz,
        )
