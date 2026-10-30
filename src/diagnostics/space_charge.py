"""Accumulated beam energy exchange with the space-charge field."""

import numpy as np

from constants import esign


def calculate_space_charge_power(beam_state, space_charge_state, grid):
    """Accumulate the DC and RF contributions in MATLAB lines 715-722.

    effSCDC, effSCRF and their sum effSC are power fractions normalized to
    Nbeam * (U0 * 1000) * I0. They start at zero at the entrance and record
    accumulated energy exchange, not absolute stored electrostatic energy.
    Their sign is retained; these contributions need not be positive losses.

    Use the final beam currents and the separate space-charge fields. MATLAB
    reuses Esz0 for its total DC field; the two agree in the currently
    supported solver because the cavity DC contribution Emz0 is zero.
    This diagnostic does not include work from any future DC cavity field.

    Each interval uses its LEFT endpoint, exactly as in MATLAB. betae * dz
    is the normalized longitudinal step. No fields or beam state are changed.
    """
    IzDC = beam_state.IzDC
    Iz = beam_state.Iz
    Esz0 = space_charge_state.Esz0
    Esz = space_charge_state.Esz

    # Keep the DC field's complex dtype; do not silently discard any residual
    # imaginary component from the space-charge calculation.
    effSCDC = np.zeros_like(Esz0)
    effSCRF = np.zeros(len(grid.z))

    for kz in range(1, len(grid.z)):
        # DC current times DC field gives the steady power-transfer term.
        effSCDC[kz] = (
            effSCDC[kz - 1]
            + esign * grid.betae * grid.dz
            * IzDC[kz - 1] * Esz0[kz - 1]
        )

        # Period-averaged RF power: one half of Re(I* E) per harmonic.
        # Conjugation preserves the relative phase of current and field.
        effSCRF[kz] = (
            effSCRF[kz - 1]
            + esign * grid.betae * grid.dz
            * 0.5
            * np.sum(np.real(np.conj(Iz[kz - 1, :]) * Esz[kz - 1, :]))
        )

    effSC = effSCDC + effSCRF

    return {
        "effSCDC": effSCDC,
        "effSCRF": effSCRF,
        "effSC": effSC,
    }
