"""Gap voltages and RF power accounting from the final circuit solution."""

import numpy as np

from constants import eta0


def calculate_cavity_power(
    cavities, cavity_fields, cavity_state, beam, grid, reference,
    drivefactor=1.0,
):
    """Evaluate MATLAB lines 664-706 for input, output and bunching cavities.

    Pass the drivefactor used for the circuit solution, normally
    result.drivefactors[-1]. It multiplies incident voltage; its square
    multiplies incident power. This function does not solve the circuit or
    change its state.

    Returned quantities keep the MATLAB names:
      Vng: complex gap voltage per cavity, in kV (not its magnitude).
      effloss: wall loss per cavity divided by total reference beam power.
      lossacc: accumulated wall-loss fraction along grid.zi.
      RFinput: total net input RF power after reflection, in W.
      refcavity: complex reflection coefficient of the LAST input cavity.
      effcavity: load power / beam power of the LAST output cavity.

    The last two scalars remain zero when no corresponding cavity exists.
    Total reference beam power is Nbeam * (U0 * 1000) * I0, in W.
    A static intermediate gap uses Vng = U0 * Vnc and contributes no RF wall
    loss here. This diagnostic does not add DC-gap support to the solver.
    Traveling-wave power accounting requires the separate, unimplemented
    traveling-wave model and its Pmzloss.
    """
    if np.any(cavities.type == 2):
        raise NotImplementedError("Traveling-wave power accounting is not implemented.")

    Vnc = cavity_state.Vnc
    Vnp = cavity_state.Vnp
    Vin = cavity_state.Vin
    Pin = cavity_state.Pin
    U0 = reference.U0
    I0 = reference.I0
    Nbeam = beam.Nbeam
    zi = grid.zi
    k0 = grid.k0
    ez = cavity_fields.ez
    Qin = cavities.Qin
    nh = cavities.nh
    gaman = cavities.gaman
    zshift = cavities.zshift

    # MATLAB 664-669: mode amplitudes give wall loss through the intrinsic Q.
    refcavity = 0.0
    effcavity = 0.0
    Vng = np.zeros_like(Vnc, dtype=complex)
    effloss = (
        np.abs(Vnc)**2 * (U0 * 1000) / Nbeam / I0 / 2 / eta0 / Qin
        * np.sign(nh) * np.sign(np.abs(cavities.type - 2))
    )
    lossacc = np.zeros_like(zi, dtype=float)
    RFinput = 0.0

    for k in range(len(Vnc)):
        # Integrate the RF mode shape in mm, retaining the voltage phase.
        # This is the gap voltage, without an electron transit-phase factor.
        if cavities.type[k] == 1 and nh[k] == 0:
            Vng[k] = U0 * Vnc[k]
        else:
            Vng[k] = U0 * Vnc[k] / np.sqrt(k0[k]) * np.trapezoid(ez[:, k], x=zi)

        if cavities.type[k] == 0:
            # Incident minus reflected power; drivefactor scales voltage.
            refcavity = Vnp[k] / (Vin[k] * drivefactor)
            RFinput += Pin[k] * (1 - np.abs(refcavity)**2) * drivefactor**2
        elif cavities.type[k] == -1:
            # The load absorbs the fraction 1 - |gaman|^2 of outgoing power.
            effcavity = (
                0.5 * (1 - np.abs(gaman[k])**2) * np.abs(Vnp[k])**2
                * U0 * 1000 / Nbeam / I0 / eta0
            )

        # MATLAB 705: assign each cavity's loss at its center, giving a step
        # profile with half the loss included exactly at zshift[k].
        lossacc += 0.5 * effloss[k] * (1 + np.sign(zi - zshift[k]))

    return {
        "Vng": Vng,
        "effloss": effloss,
        "lossacc": lossacc,
        "RFinput": RFinput,
        "refcavity": refcavity,
        "effcavity": effcavity,
    }
