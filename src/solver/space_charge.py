import numpy as np
from scipy.special import j1

from constants import esign, eta0, etom, mu0
from model.beam import BeamParameters
from solver.beam_state import BeamState
from solver.grid import GridParameters
from solver.reference import ReferenceParameters


def calculate_plasma_frequency(
    beam: BeamParameters,
    reference: ReferenceParameters,
    factor: float,
) -> float:
    """Calculate the cylindrical-beam plasma angular frequency in rad/s."""

    beam_area = np.pi * (
        (beam.rm2 * 1.0e-3) ** 2
        - (beam.rm1 * 1.0e-3) ** 2
    )

    physical_wp = np.sqrt(
        reference.I0
        * etom
        * eta0
        / (beam_area * reference.s)
    )

    return factor * physical_wp


def calculate_prefill_space_charge(
    beam: BeamParameters,
    grid: GridParameters,
    reference: ReferenceParameters,
    state: BeamState,
    wp: float,
    fillfactor: float,
    DCSPfactor: float,
) -> np.ndarray:
    Ns = state.QQ.shape[1]
    Esz00 = np.zeros((len(grid.z), Ns + 1), dtype=complex)

    if wp == 0.0 or fillfactor == 0.0 or DCSPfactor == 0.0:
        return Esz00

    if beam.rc == 0.0:
        raise NotImplementedError(
            "The sheet-beam branch is not implemented."
        )

    zp = grid.z[0] - grid.dt * np.arange(1, grid.dN + 1)
    if len(zp) == 0:
        return Esz00

    gamazp = np.ones((len(zp), Ns + 1))
    gamazp[:, 1:] = state.gamaz[0]

    QQfill = np.zeros((len(zp), Ns + 1), dtype=complex)
    QQfill[:, 0] = state.QQDC[0]

    harmonics = np.arange(1, Ns + 1)
    QQfill[:, 1:] = state.QQ[0] * np.exp(
        -1.0j
        * harmonics
        * (grid.z[0] - zp[:, np.newaxis])
    )

    space_charge_factor = (
        wp**2
        / grid.w**2
        * (reference.gama + 1.0)
        / reference.gama**2
    )

    for k, z_value in enumerate(grid.z):
        distance = np.abs(zp - z_value)[:, np.newaxis]
        direction = np.sign(z_value - zp)[:, np.newaxis]
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

    return Esz00
