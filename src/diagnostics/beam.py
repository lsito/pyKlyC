"""Post-processing of a completed beam-tracking pass."""

import numpy as np


def find_peak_modulation(beam_state, dynamics):
    """Find the strongest fundamental modulation (MATLAB 708-712).

    Izmax is the maximum magnitude of the normalized first current harmonic.
    ntemp is its longitudinal grid index; grid.zi[ntemp] gives its position
    in mm. ncenter is the particle-column index nearest the fundamental
    current peak phase at that plane. Both indices are zero-based in Python.

    The first harmonic varies as Im(Iz1 * exp(1j * phase)), so its maximum
    occurs at pi/2 - angle(Iz1). Compare phases on the unit circle to handle
    the 2*pi wrap. Higher harmonics can shift the actual current peak; this
    reference electron is used to center the trajectory plot, not to define
    the exact center of the full bunch. As in MATLAB, ties select the first
    entry. The calculation reads the final states without changing them.
    """
    Iz = beam_state.Iz
    Tnz = dynamics.Tnz
    Y = dynamics.Y

    # MATLAB 708-710: locate the strongest fundamental current modulation.
    ntemp = np.argmax(np.abs(Iz[:, 0]))
    Izmax = np.abs(Iz[ntemp, 0])

    # MATLAB 711: closest active electron to the fundamental peak phase.
    ncenter = np.argmin(
        np.abs(
            np.exp(1j * Tnz[ntemp, Y])
            - np.exp(1j * (np.pi / 2 - np.angle(Iz[ntemp, 0])))
        )
    )

    # MATLAB 712: map the position within Y to the actual particle column.
    ncenter = Y[ncenter]

    return {
        "Izmax": Izmax,
        "ntemp": ntemp,
        "ncenter": ncenter,
    }


def check_boundary_reaching(dynamics):
    """Check trajectory storage and downstream exits (MATLAB 640-655).

    Zn has shape (Ntimes * len(z), Nt). Each column is an emitted
    sample; Y contains the active column indices. Unwritten history entries
    are positive infinity. Every active column has an initialized entrance
    entry, so its last written row can always be found.

    The returned n retains MATLAB's ONE-BASED row numbering. It is the largest
    last-written row among active trajectories, not a particle count, spatial
    coordinate or common arrival time. In Python its array index is n - 1.

    status follows the original branch order:
      Storage limit: at least one active trajectory wrote the final row;
      Reflected electrons: fewer downstream exits than active samples;
      Normal: all active samples counted as downstream exits.

    These are MATLAB's bookkeeping diagnostics. A last-row write does not
    prove trapping: an exit may coincide with the storage limit. Likewise,
    the reflected label includes trajectories discarded before a downstream
    exit. Use check_negative_velocity separately to detect recorded backward
    motion, even if that particle eventually exited downstream.

    Return a dictionary for inspection/reporting. This function changes no
    solver state and neither prints a warning nor changes convergence status.
    """
    Zn = dynamics.Zn
    Y = dynamics.Y
    Yflag = dynamics.Yflag

    # MATLAB 641-647: largest last-written row across active electrons.
    n = 0
    for kkk in Y:
        ntemp = np.flatnonzero(Zn[:, kkk] != np.inf)[-1] + 1
        n = max(n, ntemp)

    # MATLAB 649-655: storage exhaustion takes priority over the exit count.
    # The allocated row count is Ntimes * len(z); it already exists in Zn.
    if n == Zn.shape[0]:
        status = "Storage limit" # i.e. The last written row correspond also to the last plane available for computation
    elif Yflag != len(Y):
        status = "Reflected electrons"
    else:
        status = "Normal"

    return {
        "n": n,
        "status": status,
    }


def check_negative_velocity(dynamics, reference):
    """Detect recorded backward motion independently of exits (MATLAB 659-662).

    Vn stores signed v/v0 and reference.s is v0/c, so multiplying their
    minimum by s gives the minimum velocity in units of c. As in MATLAB,
    inspect the whole Vn array. Positive-infinity placeholders do not lower
    the minimum; nanmin retains MATLAB's omission of NaNs.

    Return the minimum and a flag without changing the solver state.
    """
    Vn = dynamics.Vn
    s = reference.s
    minimum_v_over_c = np.nanmin(Vn) * s

    return {
        "minimum_v_over_c": minimum_v_over_c,
        "negative_velocity": minimum_v_over_c < 0,
    }
