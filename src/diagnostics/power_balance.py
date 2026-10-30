"""RF efficiency inferred from the final beam and field energy balance."""

import numpy as np


def calculate_power_balance(
    beam_state, space_charge_power, cavity_power,
    numerical, beam, grid, reference,
):
    """Combine beam, space-charge and cavity power terms (MATLAB 724-726).

    effend is the beam kinetic-power reduction at the observation position.
    effSCend is the accumulated space-charge contribution at that same point.
    Subtract it and cavity wall losses, then add net external RF input, to
    obtain effendmodified: the RF efficiency inferred from energy balance.
    This is distinct from effcavity, obtained from the output-port amplitude.

    All returned quantities are fractions of the total reference beam power,
    Nbeam * (U0 * 1000) * I0 in W. RFinput is already a net power in W.
    The observation offset matches the convergence diagnostic (GIFpara(5)
    in MATLAB). Out-of-domain interpolation gives NaN, like MATLAB interp1.
    Read the supplied final states without another solve or any mutation.
    """
    zi = grid.zi
    z_eval = np.max(zi) - numerical.efficiency_end_offset_mm

    effend = np.interp(
        z_eval, zi, beam_state.effe,
        left=np.nan, right=np.nan,
    )

    effSC = space_charge_power["effSC"]
    effSCend = np.interp(
        z_eval, zi, effSC,
        left=np.nan, right=np.nan,
    )

    effloss = cavity_power["effloss"]
    RFinput = cavity_power["RFinput"]

    # Lost beam kinetic power is not all available as output RF: account for
    # space charge and wall losses, and include the external source's input.
    effendmodified = (
        effend
        - effSCend
        - np.sum(effloss)
        + RFinput / beam.Nbeam / (reference.U0 * 1000) / reference.I0
    )

    return {
        "effend": effend,
        "effSCend": effSCend,
        "effendmodified": effendmodified,
    }
