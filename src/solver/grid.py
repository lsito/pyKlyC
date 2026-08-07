from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from constants import c
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
    rcvs: np.ndarray

    @classmethod
    def calculate(
        cls,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        tube: TubeParameters,
        beam: BeamParameters,
        reference: ReferenceParameters,
    ) -> "GridParameters":
        """Calculate the quantities defined in KLJCmainsv22 lines 64--75."""

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

        return cls(
            w=w,
            w0=w0,
            k0=k0,
            Lsb=Lsb,
            betae=betae,
            dz=dz,
            zi=zi,
            rcvs=rcvs,
        )
