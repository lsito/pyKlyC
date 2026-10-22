from dataclasses import dataclass

import numpy as np

from constants import esign, eta0
from model.beam import BeamParameters
from model.cavity import CavityParameters
from model.settings import NumericalSettings
from solver.beam_state import BeamState
from solver.cavity_fields import CavityFields
from solver.grid import GridParameters
from solver.reference import ReferenceParameters


@dataclass
class CavityState:
    """Cavity voltages and equivalent-circuit state during evolution."""

    Pinflag: np.ndarray
    Pin: np.ndarray
    Vin: np.ndarray
    flagVnc: np.ndarray
    Vnp: np.ndarray
    Vnc: np.ndarray
    Mc: np.ndarray
    B: np.ndarray

    @classmethod
    def initialize(
        cls,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        beam: BeamParameters,
        reference: ReferenceParameters,
    ) -> "CavityState":
        """Initialize the input excitation and allocate the circuit arrays."""
        Pinflag = np.sign(cavities.Pin)
        Pin = np.abs(cavities.Pin)

        Vin = (
            np.sqrt(
                Pin
                / reference.U0
                / 1.0e3
                / beam.Nbeam
                / reference.I0
            )
            * np.sqrt(
                2.0
                * beam.Nbeam
                * reference.I0
                / reference.U0
                / 1.0e3
                * eta0
            )
        )

        return cls(
            Pinflag=Pinflag,
            Pin=Pin,
            Vin=Vin,
            flagVnc=np.zeros(numerical.Nc, dtype=complex),
            Vnp=np.zeros(numerical.Nc, dtype=complex),
            Vnc=np.zeros(numerical.Nc, dtype=complex),
            Mc=np.zeros(
                (2 * numerical.Nc, 2 * numerical.Nc),
                dtype=complex,
            ),
            B=np.zeros(2 * numerical.Nc, dtype=complex),
        )

    def assemble_Mc_B(
        self,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        cavity_fields: CavityFields,
        beam: BeamParameters,
        beam_state: BeamState,
        grid: GridParameters,
        drivefactor: float = 1.0,
    ):
        self.Mc.fill(0.0)
        self.B.fill(0.0)

        for k in range(numerical.Nc):
            port = 2 * k
            mode = port + 1
            nh = cavities.nh[k]

            if nh < 1 or nh > numerical.Ns:
                self.Mc[port, port] = 1.0
                self.Mc[mode, mode] = 1.0
                continue

            beam_drive = (
                -esign
                * beam.Nbeam
                * np.trapezoid(
                    cavity_fields.C3[:, k]
                    * beam_state.Iz[:, nh - 1],
                    x=grid.z,
                )
            )

            D0k = (
                1.0 / cavities.Qin[k]
                + 2.0j
                * (nh * grid.w - grid.w0[k])
                / grid.w0[k]
            )

            self.Mc[mode, mode] = (
                -D0k - 2.0j * cavities.kai[k, k]
            )

            for j in range(numerical.Nc):
                if j != k:
                    self.Mc[mode, 2 * j + 1] = (
                        -1.0j * cavities.kai[k, j]
                    )

            exp_positive = np.exp(1.0j * cavities.Lsb[k])
            exp_negative = np.exp(-1.0j * cavities.Lsb[k])
            sqrt_Qext = np.sqrt(cavities.Qext[k])
            drive = self.Vin[k] * drivefactor

            if cavities.type[k] == -1:
                self.Mc[port, port] = sqrt_Qext * (
                    cavities.gaman[k] * exp_positive
                    + exp_negative
                )
                self.Mc[port, mode] = -1.0
                self.Mc[mode, port] = (
                    cavities.gaman[k] * exp_positive
                    - exp_negative
                ) / sqrt_Qext
                self.B[mode] = beam_drive

            elif cavities.type[k] == 0:
                self.Mc[port, port] = sqrt_Qext * exp_negative
                self.Mc[port, mode] = -1.0
                self.Mc[mode, port] = -exp_negative / sqrt_Qext
                self.B[port] = -sqrt_Qext * exp_positive * drive
                self.B[mode] = (
                    beam_drive
                    - exp_positive * drive / sqrt_Qext
                )

            elif cavities.type[k] == 1:
                self.Mc[port, port] = 1.0
                self.B[mode] = beam_drive

            else:
                self.Mc[port, port] = 1.0
                self.Mc[mode, :] = 0.0
                self.Mc[mode, mode] = 1.0

        return self.Mc, self.B

    def solve(
        self,
        numerical: NumericalSettings,
        cavities: CavityParameters,
        cavity_fields: CavityFields,
        beam: BeamParameters,
        beam_state: BeamState,
        grid: GridParameters,
        drivefactor: float = 1.0,
    ):
        self.flagVnc[:] = self.Vnc

        self.assemble_Mc_B(
            numerical,
            cavities,
            cavity_fields,
            beam,
            beam_state,
            grid,
            drivefactor,
        )

        solution = np.linalg.solve(self.Mc, self.B)
        self.Vnp[:] = solution[0::2]
        self.Vnc[:] = solution[1::2]
