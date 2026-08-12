from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from constants import c, mu01
from model.beam import BeamParameters
from model.cavity import CavityParameters
from model.settings import NumericalSettings
from model.tube import TubeParameters
from solver.reference import ReferenceParameters


@dataclass(frozen=True)
class GridParameters:
    """Frequency quantities, longitudinal grid, and sampled tunnel radius."""

    w: float
    w0: np.ndarray
    k0: np.ndarray
    Lsb: np.ndarray
    betae: float
    dz: float
    zi: np.ndarray
    z: np.ndarray
    dt: float
    rcvs: np.ndarray
    dN: int
    Nesz: np.ndarray

    @classmethod
    def calculate(
        cls,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        tube: TubeParameters,
        beam: BeamParameters,
        reference: ReferenceParameters,
        factorregion: float,
    ) -> "GridParameters":
        """Calculate the grid quantities in lines 64--86 and 121--122."""

        w = 2.0 * np.pi * numerical.f * 1.0e9
        w0 = 2.0 * np.pi * cavities.f0 * 1.0e9
        k0 = w0 / c / 1.0e3
        Lsb = np.pi * np.ones(numerical.Nc)

        betae = w / c / reference.s / 1.0e3
        initial_dz = 2.0 * np.pi / betae / numerical.Nz
        number_of_intervals = int(np.floor(tube.L / initial_dz))
        dz = tube.L / number_of_intervals

        zi = np.linspace(
            tube.z0,
            tube.z0 + tube.L,
            number_of_intervals + 1,
        )

        radius_interpolator = interp1d(
            tube.rcv[0],
            tube.rcv[1] * beam.rc,
            kind="linear",
            fill_value="extrapolate",
        )
        rcvs = radius_interpolator(zi)

        z = betae * zi
        dt = betae * dz

        if beam.rc == 0.0:
            raise NotImplementedError(
                "The sheet-beam branch not yet implemented."
            )
        else: 
            dN = int(np.floor(factorregion * beam.rc / mu01 / dz))

            # Array of start and stop indices for the longitudinal region of each sample.
            # Notice that these indices should be dN after and dN before the sample index.
            # However, no index before 0 and after len(zi)-1 is allowed, so the start and stop indices are clipped to the valid range.
            Nesz = np.zeros((len(zi), 2), dtype=int)

            for k in range(len(zi)):
                Nesz[k, 0] = max(0, k - dN)
                # Python slice stops are exclusive, unlike MATLAB's upper index.
                Nesz[k, 1] = min(len(zi), k + dN + 1)

        return cls(
            w=w,
            w0=w0,
            k0=k0,
            Lsb=Lsb,
            betae=betae,
            dz=dz,
            zi=zi,
            z=z,
            dt=dt,
            rcvs=rcvs,
            dN=dN,
            Nesz=Nesz,
        )
