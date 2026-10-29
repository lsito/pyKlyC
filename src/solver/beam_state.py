from dataclasses import dataclass

import numpy as np

from constants import c, etom
from model.beam import BeamParameters
from model.settings import NumericalSettings
from solver.grid import GridParameters
from solver.reference import ReferenceParameters

@dataclass
class BeamState:
    """Mutable beam estimate used by the circuit and space-charge solvers.

    initialize supplies the entrance-based first guess. After each tracking
    pass, update mixes the new current/charge distributions into this estimate.
    The individual electron trajectories remain in BeamDynamics, not here.
    """

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

    def update(self, numerical, beam, reference, dynamics):
        """Update beam sources and derived profiles (MATLAB 523--530).

        Call after dynamics.track(), which also reconstructs QQr. Arrays with
        suffix r are the NEW distributions produced by that tracking pass;
        this object's arrays are the PREVIOUS nonlinear iteration's estimate.

        Mixing damps changes in the beam sources seen by the next circuit and
        space-charge solves. It is numerical relaxation between iterations,
        not an extra physical time step or a change to particle trajectories.
        A weight of 1 replaces the estimate, 0 retains it, and a weight between
        them blends old and new values. Complex harmonics are mixed directly,
        preserving phase information rather than mixing their magnitudes.

        MATLAB takes abs(iterfactor) before these lines. Its sign separately
        controls the tracking field correction and must not reverse mixing.
        Assigning new arrays leaves the raw dynamics accumulators unchanged
        and does not alias them, even when the relaxation weight is 1.

        Then recompute effe, vez and gamaz from the current pass's UNRELAXED Pe
        and the ALREADY RELAXED IzDC, exactly in MATLAB's order. These derived
        profiles are replaced directly, even if iterfactor is zero. beam supplies
        the active entrance samples and voltages; reference supplies Iz0 and s.

        No clipping or artificial small current is introduced. Zero net current
        or zero injected energy can produce inf/nan; signed counterstreaming
        fluxes can give a non-real effective velocity. Such results do not
        describe a physical single forward beam. Complex square roots retain
        MATLAB's behavior for negative radicands rather than hiding them.
        """

        iterfactor = abs(numerical.iterfactor)
        self.IzDC = (1.0 - iterfactor) * self.IzDC + iterfactor * dynamics.IzDCr
        self.Iz = (1.0 - iterfactor) * self.Iz + iterfactor * dynamics.Izr
        self.QQDC = (1.0 - iterfactor) * self.QQDC + iterfactor * dynamics.QQDCr
        self.QQ = (1.0 - iterfactor) * self.QQ + iterfactor * dynamics.QQr

        Nt = beam.Nt
        Izt = beam.Izt
        Ut = beam.Ut
        Y0 = beam.Y0
        Iz0 = reference.Iz0
        s = reference.s
        Pe = dynamics.Pe

        # 528: Fraction of injected kinetic-energy flux removed from the beam.
        # Pe is the signed, current-weighted sum of gamma-1. c^2/etom converts
        # it to an equivalent voltage, and 1e-3 converts V to kV to match Ut.
        # Only active entrance samples enter the denominator. This efficiency
        # includes net beam-energy loss, not only useful RF output power.
        self.effe = 1.0 - c * c / etom * 1.0e-3 * Pe / np.sum(
            (2.0 * np.pi / Nt) * Izt[Y0] / Iz0 * Ut[Y0]
        )

        # 529: Infer one representative speed from the kinetic energy per net
        # current: gamma_eff = 1 + Pe/(2*pi*IzDC). This is NOT the arithmetic
        # mean of particle velocities, nor a signed reflected-particle speed.
        # The positive root gives v_eff/c; dividing by s gives vez = v_eff/v0.
        # Keep Pe unrelaxed and use the updated IzDC, as in the original code.
        self.vez = 1.0 / s * np.emath.sqrt(
            1.0 - (1.0 + (1.0 / (2.0 * np.pi) * Pe / self.IzDC)) ** -2
        )

        # 530: Relativistic factor corresponding to that representative speed.
        # These profiles are inputs to subsequent space-charge calculations.
        self.gamaz = 1.0 / np.emath.sqrt(1.0 - (self.vez * s) ** 2)
