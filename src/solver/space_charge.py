from dataclasses import dataclass

import numpy as np
from scipy.special import j1

from constants import esign, eta0, etom, mu0
from model.beam import BeamParameters
from solver.beam_state import BeamState
from solver.grid import GridParameters
from solver.reference import ReferenceParameters


@dataclass
class SpaceChargeState:
    """Space-charge fields stored during the beam evolution."""

    wp: float
    zp: np.ndarray
    gamazp: np.ndarray
    QQfill: np.ndarray
    Esz00: np.ndarray
    Eszr: np.ndarray
    Esztotal: np.ndarray
    Esz0: np.ndarray
    Esz: np.ndarray
    SCmethod: int

    @classmethod
    def calculate(
        cls,
        beam: BeamParameters,
        grid: GridParameters,
        reference: ReferenceParameters,
        beam_state: BeamState,
        factor: float,
        fillfactor: float,
        DCSPfactor: float,
    ) -> "SpaceChargeState":
        wp = cls.calculate_plasma_frequency(beam, reference, factor)
        Esz00, zp, gamazp, QQfill = cls.calculate_prefill_space_charge(
            beam,
            grid,
            reference,
            beam_state,
            wp,
            fillfactor,
            DCSPfactor,
        )
        Eszr = cls.calculate_Eszr(beam, grid)

        return cls(
            wp=wp,
            zp=zp,
            gamazp=gamazp,
            QQfill=QQfill,
            Esz00=Esz00,
            Eszr=Eszr,
            Esztotal=np.zeros_like(Esz00),
            Esz0=np.zeros(len(grid.z), dtype=complex),
            Esz=np.zeros_like(beam_state.QQ),
            SCmethod=0,
        )

    @staticmethod
    def calculate_Eszr(
        beam: BeamParameters,
        grid: GridParameters,
    ) -> np.ndarray:
        roots = mu0.reshape(1, -1)
        rcvs = grid.rcvs.reshape(-1, 1)

        Eszr = (
            2.0
            / roots
            * (
                beam.rm2 * j1(roots * beam.rm2 / rcvs)
                - beam.rm1 * j1(roots * beam.rm1 / rcvs)
            )
            / j1(roots)
        ) ** 2

        return Eszr

    @staticmethod
    def calculate_plasma_frequency(
        beam: BeamParameters,
        reference: ReferenceParameters,
        factor: float,
    ) -> float:
        beam_area = np.pi * (
            (beam.rm2 * 1.0e-3) ** 2
            - (beam.rm1 * 1.0e-3) ** 2
        )

        wp = factor * np.sqrt(
            reference.I0
            * etom
            * eta0
            / (beam_area * reference.s)
        )

        return wp

    def update_local(
        self,
        beam: BeamParameters,
        grid: GridParameters,
        reference: ReferenceParameters,
        beam_state: BeamState,
    ):
        harmonics = np.arange(1, beam_state.Iz.shape[1] + 1)

        modal_denominator = 1.0 + (
            mu0[np.newaxis, np.newaxis, :]
            / harmonics[np.newaxis, :, np.newaxis]
            / beam.rc
            / grid.betae
            * beam_state.vez[:, np.newaxis, np.newaxis]
            * beam_state.gamaz[:, np.newaxis, np.newaxis]
        ) ** 2

        cylindrical_modes = np.sum(
            self.Eszr[:, np.newaxis, :] / modal_denominator,
            axis=2,
        )

        self.Esz[:] = (
            -esign
            * 1.0j
            * beam_state.Iz
            / harmonics
            * self.wp**2
            / grid.w**2
            * (reference.gama + 1.0)
            / reference.gama**2
            / (beam.rm2**2 - beam.rm1**2)
            * cylindrical_modes
        )
        self.Esz0.fill(0.0)

    @staticmethod
    def calculate_prefill_space_charge(
        beam: BeamParameters,
        grid: GridParameters,
        reference: ReferenceParameters,
        beam_state: BeamState,
        wp: float,
        fillfactor: float,
        DCSPfactor: float,
    ):
        Ns = beam_state.QQ.shape[1]
        Esz00 = np.zeros((len(grid.z), Ns + 1), dtype=complex)
        zp = grid.z[0] - grid.dt * np.arange(1, grid.dN + 1)
        zp = np.repeat(zp[:, np.newaxis], Ns + 1, axis=1)

        gamazp = np.ones((len(zp), Ns + 1))
        gamazp[:, 1:] = beam_state.gamaz[0]

        QQfill = np.zeros((len(zp), Ns + 1), dtype=complex)
        QQfill[:, 0] = beam_state.QQDC[0]

        harmonics = np.arange(1, Ns + 1)
        QQfill[:, 1:] = beam_state.QQ[0] * np.exp(
            -1.0j
            * harmonics
            * (grid.z[0] - zp[:, 1:])
        )

        if wp == 0.0 or fillfactor == 0.0 or DCSPfactor == 0.0:
            return Esz00, zp, gamazp, QQfill

        if beam.rc == 0.0:
            raise NotImplementedError(
                "The sheet-beam branch is not implemented."
            )

        if len(zp) == 0:
            return Esz00, zp, gamazp, QQfill

        space_charge_factor = (
            wp**2
            / grid.w**2
            * (reference.gama + 1.0)
            / reference.gama**2
        )

        for k, z_value in enumerate(grid.z):
            distance = np.abs(zp - z_value)
            direction = np.sign(z_value - zp)
            cylindrical_modes = np.zeros_like(gamazp)

            for root in mu0:
                radial_coupling = (
                    2.0
                    / root
                    * (
                        beam.rm2 * j1(root * beam.rm2 / beam.rc)
                        - beam.rm1 * j1(root * beam.rm1 / beam.rc)
                    )
                    / j1(root)
                ) ** 2 / (beam.rm2**2 - beam.rm1**2)

                cylindrical_modes += radial_coupling * np.exp(
                    -root
                    * gamazp
                    * distance
                    / grid.betae
                    / beam.rc
                )

            Esz00[k] = (
                esign
                * grid.dz
                * grid.betae
                * np.sum(
                    -QQfill
                    * direction
                    / 2.0
                    * space_charge_factor
                    * cylindrical_modes,
                    axis=0,
                )
            )

        return Esz00, zp, gamazp, QQfill
