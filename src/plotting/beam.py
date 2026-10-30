"""Final beam plots corresponding to MATLAB figures 1 and 2."""

import numpy as np
import matplotlib.pyplot as plt


def plot_applegate(
    dynamics, beam, grid, cavities, cavity_fields,
    ncenter, Ntp=None, zmax=None,
):
    """Plot particle arrival-phase delay and normalized RF cavity profiles.

    MATLAB 729-752: x = Zn/betae in mm and psi = Tn-Zn = omega*(t-x/v0).
    Wrap psi into one RF period centered on tn1[ncenter]-z[0]. ncenter is
    the zero-based particle column returned by find_peak_modulation().
    Use the full Zn/Tn histories so time-tracked and reflected motion remain
    visible. Unwritten history entries are omitted before phase arithmetic.

    Ntp is a positive display sampling parameter, not a tracking resolution.
    None displays all active samples. Otherwise use MATLAB's positive-number
    rounding to select every round(Nt/Ntp)-th emitted sample, then intersect
    with Y. The selected count need not equal Ntp.

    The right axis shows abs(sum(ez * sign(nh))), excluding static profiles.
    It is NOT the instantaneous field or a profile scaled by Vnc.
    zmax defaults to the final grid position. Return fig, (ax, ax_field);
    the caller decides whether to display or save. Input states are unchanged.
    """
    Nt = beam.Nt
    tn1 = beam.tn1
    z = grid.z
    zi = grid.zi
    betae = grid.betae
    Y = dynamics.Y

    if Ntp is None:
        Ntp = Nt
    if zmax is None:
        zmax = np.max(zi)

    # MATLAB rounds positive half-integers upward; Python round() does not.
    stride = max(int(np.floor(Nt / Ntp + 0.5)), 1)
    selected = np.intersect1d(Y, np.arange(0, Nt, stride))
    psi_center = tn1[ncenter] - z[0]

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax_field = ax.twinx()

    for kp in selected:
        valid = np.isfinite(dynamics.Zn[:, kp]) & np.isfinite(dynamics.Tn[:, kp])
        Zn = dynamics.Zn[valid, kp]
        Tn = dynamics.Tn[valid, kp]
        psi = Tn - Zn
        psi_plot = psi - np.floor((psi - psi_center + np.pi) / (2 * np.pi)) * 2 * np.pi
        # Dots avoid artificial lines across the 2*pi phase-wrap boundary.
        ax.plot(Zn / betae, psi_plot, "b.", markersize=2, label=f"particle {kp}")

    # Form a separate array: do not clear DC columns in the original ez.
    ez_background = np.abs(np.sum(cavity_fields.ez * np.sign(cavities.nh), axis=1))
    ax_field.plot(zi, ez_background, color="tab:orange", label="Normalized RF profiles")
    ax_field.set_ylabel(r"Normalized RF profile [mm$^{-3/2}$]")
    if np.min(ez_background) < np.max(ez_background):
        ax_field.set_ylim(np.min(ez_background), np.max(ez_background))
    # For an identically zero/constant background, retain automatic limits.
    ax_field.grid(False)

    ax.set(
        title="Applegate diagram", xlabel="z position [mm]",
        ylabel=r"$\omega(t-x/v_0)$ [rad]",
        xlim=(np.min(zi), zmax), ylim=(psi_center - np.pi, psi_center + np.pi),
    )
    ax.set_yticks([psi_center - np.pi, psi_center + np.pi])
    ax.grid(True, color="gray", linestyle="--", alpha=0.45)
    return fig, (ax, ax_field)


def plot_current_harmonics(beam_state, grid, cavities, zmax=None):
    """Plot normalized current-harmonic magnitudes (MATLAB 757-773).

    Iz is already normalized by Iz0. Column k contains harmonic k+1; DC is
    stored separately and is not plotted here. Cavity markers span [0, 2],
    as in MATLAB. Return fig, ax without showing or saving the figure.
    """
    zi = grid.zi
    Iz = beam_state.Iz
    if zmax is None:
        zmax = np.max(zi)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for k in range(Iz.shape[1]):
        ax.plot(zi, np.abs(Iz[:, k]), linewidth=2.0**(2 - k), label=f"Harmonic {k + 1}")

    for zshift in cavities.zshift:
        ax.plot([zshift, zshift], [0, 2], "r--", linewidth=1, label="_nolegend_")

    ax.set(
        title="Current-harmonic modulation depth", xlabel="z position [mm]",
        ylabel=r"$|I_m/I_{z0}|$", xlim=(np.min(zi), zmax),
    )
    ax.legend(loc="upper left")
    ax.grid(True, color="gray", linestyle="--", alpha=0.45)
    return fig, ax
