from dataclasses import dataclass

import numpy as np

from model.beam import BeamParameters
from model.settings import NumericalSettings
from solver.grid import GridParameters
from solver.reference import ReferenceParameters

@dataclass(frozen=True)
class BeamState:
    """Initial current and charge-density harmonics along the beam."""

    Iz: np.ndarray
    QQ: np.ndarray
    IzDC: np.ndarray
    QQDC: np.ndarray
    effe: np.ndarray
    vez: np.ndarray
    gamaz: np.ndarray

    @classmethod
    def initialize(
        cls,
        numerical: NumericalSettings,
        beam: BeamParameters,
        grid: GridParameters,
        reference: ReferenceParameters,
    ) -> "BeamState":

        active = beam.Y0
        normalized_current = beam.Izt[active] / reference.Iz0
        number_of_grid_points = len(grid.z)

        IzDC_value = np.sum(normalized_current) / beam.Nt
        IzDC = np.full(number_of_grid_points, IzDC_value)

        Iz = np.zeros(
            (number_of_grid_points, numerical.Ns),
            dtype=np.complex128,
        )

        for harmonic in range(1, numerical.Ns + 1):
            harmonic_current = (
                2.0j
                / beam.Nt
                * np.sum(
                    normalized_current
                    * np.exp(-harmonic * 1.0j * beam.tn1[active])
                )
            )
            Iz[:, harmonic - 1] = harmonic_current

        QQDC_value = np.sum(
            normalized_current * reference.vn1[active]
        ) / beam.Nt
        QQDC = np.full(number_of_grid_points, QQDC_value)

        QQ = np.zeros(
            (number_of_grid_points, numerical.Ns),
            dtype=np.complex128,
        )

        for harmonic in range(1, numerical.Ns + 1):
            harmonic_charge = (
                2.0j
                / beam.Nt
                * np.sum(
                    normalized_current
                    * np.exp(-harmonic * 1.0j * beam.tn1[active])
                    * reference.vn1[active]
                )
            )
            QQ[:, harmonic - 1] = harmonic_charge

        effe = np.zeros(number_of_grid_points)
        vez = np.ones(number_of_grid_points)
        gamaz = (1.0 - (vez * reference.s) ** 2) ** -0.5

        return cls(
            Iz=Iz,
            QQ=QQ,
            IzDC=IzDC,
            QQDC=QQDC,
            effe=effe,
            vez=vez,
            gamaz=gamaz,
        )
