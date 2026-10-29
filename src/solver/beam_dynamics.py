"""Electron trajectories and newly accumulated beam quantities for one pass.
"""

from dataclasses import dataclass

import numpy as np

from constants import c, esign, etom


@dataclass
class BeamDynamics:
    """Mutable working arrays for one beam-tracking pass.

    This is different from BeamState. BeamState holds the current estimate of
    the beam used by the circuit and space-charge calculations. BeamDynamics
    holds individual trajectories and the NEW beam quantities accumulated
    during a pass. The MATLAB suffix ``r`` identifies those new quantities;
    they have not yet been mixed with the previous iteration's BeamState.

    Coordinates and array layout
    ----------------------------
    Let N = len(grid.z) and v0 = reference.s * c. The solver coordinates are
    normalized: z = grid.betae * zi, where zi is in mm, and time is represented
    by the RF phase omega * t, in radians. Thus z is not a distance in mm and
    the stored times are not in seconds. Arrival phases are not wrapped to
    one RF period: they must retain the accumulated flight time.

    Trajectory columns identify emitted samples, including inactive samples.
    Only the zero-based indices in Y are tracked. Harmonic-array columns instead
    identify harmonics kk = 1, ..., Ns; column kk - 1 holds harmonic kk. DC is
    stored separately, not as an extra column in Izr or QQr.

    Attributes
    ----------
    Zn : array, shape (Ntimes * N, Nt)
        Normalized position history. The first row is the entrance position.
        Unwritten rows contain positive infinity, following MATLAB. Infinity
        is a storage marker, not an electron physically located at infinity.
    Vn : array, shape (Ntimes * N, Nt)
        Signed velocity v / v0 along the trajectory history. Unwritten rows
        contain positive infinity. Inactive samples also have infinite entrance
        Vn because their vn1 is zero; those columns must not be tracked.
    Tn : array, shape (Ntimes * N, Nt)
        Arrival-phase history associated with Zn and Vn. During the spatial
        march, rows correspond to successive grid planes, so time increments
        depend on electron velocity. A later time-stepping continuation uses
        fixed phase increments grid.dt. Unwritten entries start at zero.
        Ntimes extends the history capacity; it is not a number of RF periods.
    Tnz : array, shape (N, Nt)
        Arrival phase at each longitudinal grid plane. Only the entrance row
        is known initially; subsequent zeros are placeholders, not arrivals.
    Vnz : array, shape (N, Nt)
        Signed inverse velocity v0 / v at the grid-plane crossings. Notice
        that this is the reciprocal of Vn at the same event, NOT the same
        velocity normalization. Subsequent rows start at zero as in MATLAB.
        If an electron later crosses a plane repeatedly, the tracking code
        must accumulate every crossing's contribution to current and charge;
        a single Tnz/Vnz entry cannot describe all those crossings.
    Y : integer array
        Zero-based indices of the active emitted samples, initially a copy
        of beam.Y0. It contains indices, not the number of active samples.
    Yflag : int
        Number of electrons counted as exiting downstream, initially zero.
        Despite its name, this is a counter, not a Boolean reflection flag.
    Utn : array, shape (Nt,)
        Exit kinetic energy expressed as an equivalent beam voltage in kV.
        Starts at zero because no electron has reached the exit.
    ttn : array, shape (Nt,)
        Exit arrival phase, in radians. Its initial zeros are placeholders.
    Pe : array, shape (N,)
        Dimensionless current-weighted kinetic-energy quantity. At the entrance
        Pe[0] = (2*pi/Nt) * sum((Izt[Y0]/Iz0) * (gamma[Y0] - 1)).
        It is not a voltage in kV. For a forward beam, dividing it by
        2*pi*IzDCr gives the current-weighted mean of gamma - 1.
    IzDCr : array, shape (N,)
        Newly accumulated DC current, normalized by the signed reference
        current Iz0. Only the entrance value is calculated initially.
    Izr : complex array, shape (N, Ns)
        Newly accumulated current harmonics, normalized by Iz0. The MATLAB
        convention reconstructs current as IzDCr + imag(sum(Izr_k*exp(i*k*t))).
        Only the entrance row is calculated initially.
    QQDCr : array, shape (N,)
        Newly accumulated normalized DC charge density. The entrance sum has
        the additional factor vn1 = v0/v: slower samples contribute more
        charge per unit length for the same current.
    QQr : complex array, shape (N, Ns)
        Newly accumulated normalized charge-density harmonics. ALL entries,
        including the entrance row, remain zero during initialization. MATLAB
        later obtains QQr = -(1/(i*k)) * d(Izr_k)/dz after tracking; its direct
        entrance charge-harmonic sum is commented out in lines 390--392.

    Each call to initialize creates independent arrays for a fresh pass from
    the prescribed entrance beam. It does not continue old trajectories or
    overwrite BeamState. This class is mutable because tracking will fill its
    arrays and change its counters.
    """

    Zn: np.ndarray
    Vn: np.ndarray
    Tn: np.ndarray
    Tnz: np.ndarray
    Vnz: np.ndarray
    Y: np.ndarray
    Yflag: int
    Utn: np.ndarray
    ttn: np.ndarray
    Pe: np.ndarray
    IzDCr: np.ndarray
    Izr: np.ndarray
    QQDCr: np.ndarray
    QQr: np.ndarray

    @classmethod
    def initialize(cls, numerical, beam, grid, reference):
        """Allocate a fresh tracking pass and calculate its entrance values.

        Parameters
        ----------
        numerical : NumericalSettings
            Supplies Ns (RF harmonics) and Ntimes (history extension factor).
        beam : BeamParameters
            Supplies Nt, t0, tn1, Ut, Izt and the active sample indices Y0.
            Nt counts ALL samples in one RF period, even when only a subset
            is active. t0 identifies the departure-phase sampling; tn1 is the
            phase at the simulation entrance and initializes the trajectories.
        grid : GridParameters
            Supplies the normalized longitudinal coordinates z.
        reference : ReferenceParameters
            Supplies Iz0 (signed mean current in A), s = v0/c and the entrance
            inverse velocities vn1 = v0/v. The inputs are assumed consistent
            with the beam and grid already initialized by the existing code.

        Returns
        -------
        BeamDynamics
            Independent trajectory and accumulator arrays. Nothing beyond the
            entrance has been calculated; no fields or BeamState are modified.
        """

        # Keep the MATLAB variable names in the formulas below.
        z = grid.z
        Nt = beam.Nt
        t0 = beam.t0
        tn1 = beam.tn1
        Izt = beam.Izt
        Y0 = beam.Y0
        Ns = numerical.Ns
        Ntimes = numerical.Ntimes
        Iz0 = reference.Iz0
        s = reference.s
        vn1 = reference.vn1

        # 1. Initialize Trajectory-history storage on extended grid
        Zn = np.full((Ntimes * len(z), len(t0)), np.inf)
        Vn = np.full((Ntimes * len(z), len(t0)), np.inf)
        Tn = np.zeros((Ntimes * len(z), len(t0)))

        # 1.1 Entrance values
        Zn[0, :] = z[0]
        # Evaluate 1/vn1 where it is nonzero. Inactive columns retain infinity,
        np.divide(1.0, vn1, out=Vn[0, :], where=vn1 != 0.0)
        Tn[0, :] = tn1

        # 2. Grid-plane storage
        # These rows always represent grid planes, unlike the extended history.
        Tnz = np.zeros((len(z), len(t0)))
        Vnz = np.zeros((len(z), len(t0)))
        Tnz[0, :] = tn1
        Vnz[0, :] = vn1

        # 3. Active samples and exit information
        Y = Y0.copy()
        Yflag = 0
        Utn = np.zeros_like(beam.Ut, dtype=float)
        ttn = np.zeros_like(t0, dtype=float)

        # 4. Fresh accumulators
        # Do not reuse BeamState's arrays: those hold the previous beam estimate
        Pe = np.zeros(len(z))
        IzDCr = np.zeros(len(z))
        Izr = np.zeros((len(z), Ns), dtype=complex)
        QQDCr = np.zeros(len(z))
        QQr = np.zeros((len(z), Ns), dtype=complex)

        # 5. Entrance quantities (kz=0) first DC, then all the harmonics
        kz = 0

        IzDCr[kz] = np.sum(Izt[Y0] / Iz0) / Nt

        for kk in range(1, Ns + 1):
            Izr[kz, kk - 1] = (
                2.0j / Nt
                * np.sum(Izt[Y0] / Iz0 * np.exp(-kk * 1.0j * tn1[Y0]))
            )

        QQDCr[kz] = np.sum(Izt[Y0] / Iz0 * vn1[Y0]) / Nt

        # Here s/vn1 = v/c, so the last factor is the Lorentz factor minus one.
        # Retain the velocity-based expression used in the MATLAB calculation.
        Pe[kz] = np.sum(
            (2.0 * np.pi / Nt) * (Izt[Y0] / Iz0)
            * ((1.0 - (s / vn1[Y0]) ** 2) ** -0.5 - 1.0)
        )

        # Building these for kz > 0 is the tracking code's responsibility.
        
        # QQr intentionally remains zero, including its entrance row. It will
        # be reconstructed from the longitudinal derivative of Izr AFTER tracking.

        return cls(
            Zn=Zn,
            Vn=Vn,
            Tn=Tn,
            Tnz=Tnz,
            Vnz=Vnz,
            Y=Y,
            Yflag=Yflag,
            Utn=Utn,
            ttn=ttn,
            Pe=Pe,
            IzDCr=IzDCr,
            Izr=Izr,
            QQDCr=QQDCr,
            QQr=QQr,
        )

    def track(
        self,
        numerical,
        beam,
        grid,
        reference,
        *,
        Esz,
        Esz0,
        Emz,
        Emz0,
        nn=1,
        Iz=None,
    ):
        """Run one beam pass in prescribed fields and reconstruct its charge.

        Call on a freshly initialized BeamDynamics. The spatial march is the
        fast forward-only stage. If strong deceleration requests a switch,
        continue in time on THIS object's existing trajectories and sums;
        reinitializing would erase the beam contributions already recorded.

        Supply separate RF and DC space-charge fields (Esz, Esz0) and cavity
        fields (Emz, Emz0). All are read-only inputs normalized by betae*U0.
        RF arrays have shape (len(grid.z), Ns); DC arrays have shape (len(grid.z),).
        This method assembles Esmz = Esz + Emz and Esmz0 = Esz0 + Emz0 once.
        The spatial and time trackers share these working totals. Only the
        optional spatial correction subsequently changes selected RF rows.
        Iz is the PREVIOUS iteration's current harmonics, needed for the
        optional space-charge correction. nn starts at 1 as in MATLAB.
        MATLAB separates the sign of iterfactor (the correction control) from
        its magnitude (the relaxation weight used later by BeamState.update).

        After either tracking route, reconstruct QQr from the spatial current
        gradient using charge continuity. This does not update BeamState or
        solve new fields: the outer iteration must call BeamState.update next.

        Returns (Esmz, Esmz0) for diagnostics and updates this object in place.
        The returned RF total includes any rowwise tracking correction.
        Like MATLAB, charge
        reconstruction also runs when time tracking ends through upstream
        loss, discard or exhausted storage. Finishing this method does not
        imply that every electron exited downstream: inspect Yflag and the
        histories. It does not imply nonlinear convergence either.
        """

        # Assemble the complete fields once, including planes beyond a possible
        # handoff. Array addition creates independent buffers: source fields
        # must not be overwritten by the optional RF correction in track_z.
        Esmz = Esz + Emz
        Esmz0 = Esz0 + Emz0

        # MATLAB lines 210--211: sign controls the field correction, not mixing.
        iterf = np.sign(numerical.iterfactor)
        kzstart = self.track_z(
            numerical, beam, grid, reference, Esmz, Esmz0,
            Esz=Esz, Emz=Emz, nn=nn, iterf=iterf, Iz=Iz,
        )

        # None means the spatial march reached the last grid plane. An index
        # means time tracking must resume there without counting it twice.
        if kzstart is not None:
            self.track_t(numerical, beam, grid, reference, Esmz, Esmz0, kzstart)

        self.calculate_QQr(numerical, grid)
        return Esmz, Esmz0

    def calculate_QQr(self, numerical, grid):
        """Reconstruct charge harmonics by continuity (MATLAB 516--521).

        With exp(i*kk*tau) harmonics, normalized charge continuity gives
        i*kk*QQr[:, kk-1] + d(Izr[:, kk-1])/dz = 0. Thus a spatial change
        in harmonic current requires a corresponding oscillating charge.
        Differentiate the complex phasors, NOT their magnitudes.

        Izr and QQr have shape (len(grid.z), Ns), even for Ns == 1. Axis 0
        is position; columns are harmonic orders 1 through Ns, not electrons.
        grid.dt also equals the normalized spatial spacing in this uniform
        grid. Despite its name, this is a spatial derivative, not a time one.

        NumPy's default gradient uses centered differences inside and one-sided
        first differences at the ends, matching MATLAB gradient. Broadcasting
        divides each column by its harmonic order without repmat or an Ns==1
        branch. DC charge QQDCr remains the separately accumulated residence-
        time contribution; this formula cannot be applied at harmonic zero.
        """

        Fy = np.gradient(self.Izr, grid.dt, axis=0)
        kk = np.arange(1, numerical.Ns + 1)
        self.QQr = -1.0 / (1.0j * kk) * Fy

    def track_z(
        self,
        numerical,
        beam,
        grid,
        reference,
        Esmz,
        Esmz0,
        *,
        Esz,
        Emz,
        nn=1,
        iterf=1,
        Iz=None,
    ):
        """Track forward in space - The "forward spatial march""

        Normally called by track(), which assembles the total fields first.
        Call this once on a freshly initialized BeamDynamics. All active
        electrons advance together from one longitudinal plane to the next.
        Each electron has its own arrival phase; this is NOT a simultaneous
        snapshot of the beam at one common time.

        Parameters
        ----------
        numerical, beam, grid, reference
            The same inputs used by initialize. grid.z and grid.dt are
            normalized; beam.Y0 contains zero-based active sample indices.
            The grid must have at least two planes, in increasing order.
            numerical.evolutionmethod[0] is the maximum permitted relative
            INCREASE in inverse velocity over one spatial step. Exceeding it
            requests time tracking for all active electrons. This is not an
            absolute-value or reflection test. The second control belongs
            to the outer iteration and is not used by this method.
        Esmz : complex array, shape (len(grid.z), numerical.Ns)
            Working total RF array, already filled with Esz + Emz by track().
            Supply a separate array, not an alias of either source field.
            All planes must be populated, including those beyond a possible
            time-tracking handoff. This method updates only rows requiring the
            optional SC correction. Retain the same array for track_t().
        Esmz0 : array, shape (len(grid.z),)
            Total normalized DC field, already assembled as the space-charge
            DC field plus Emz0. Its physical value is real. np.real below
            accepts the complex storage used by SpaceChargeState for this
            real-valued DC component. This input is not modified.
            Both fields are dimensionless: E[kV/mm] / (grid.betae * reference.U0).
        nn : int
            Nonlinear iteration number, starting at 1 as in MATLAB. Defaults
            to 1 for prescribed-field tests of a single tracking pass.
        iterf : number
            MATLAB's sign flag for the original iterfactor, not the relaxation
            magnitude. A nonpositive value allows the optional space-charge
            correction after iteration 5. The default 1 disables it.
        Esz, Emz : complex arrays, shape (len(grid.z), numerical.Ns)
            Required RF space-charge and cavity harmonics, respectively.
            Both are read-only inputs. The ordinary field is Esz + Emz;
            only the space-charge part is rescaled when the correction is on.
            No circuit or space-charge solve is performed here.
        Iz : complex array, optional
            Previous-iteration current harmonics. Required when nn > 5 and
            iterf <= 0. At rows where abs(Iz[kz, 0]) > 0.01, the corrected
            field is Esz[kz] * Izr[kz] / Iz[kz] + Emz[kz]. Iz is only read.
            A zero divisor in this correction is reported, not replaced with
            an artificial small number.

        Returns
        -------
        None
            All active electrons reached the last grid plane. Utn, ttn and
            Yflag are filled. This completes spatial tracking, NOT the full
            nonlinear iteration or the later QQr reconstruction.
        int
            The zero-based kzstart plane at which time tracking must begin.
            Its trajectories and beam contributions have ALREADY been stored.
            Do not initialize again or count that plane's crossing twice.
            The remaining Tn rows have fixed phase increments, ready for the
            continuation. Exit data remain unset: the caller must pass this
            index to track_t to finish tracking. Returning an index does not
            mean that the beam pass is complete.

        Notes
        -----
        The active phase and inverse-velocity equations both use the CURRENT
        plane's state (explicit Euler). A commented alternative after the
        velocity update adds half the inverse-velocity difference to the
        phase advance. It is inactive to retain MATLAB agreement, and does
        not make the full coupled integrator second-order.

        QQr remains untouched. Its derivative-based reconstruction belongs
        after the complete tracking pass. BeamState relaxation and convergence
        checks also remain outside this method.
        """

        z = grid.z
        dt = grid.dt
        Nt = beam.Nt
        t0 = beam.t0
        Izt = beam.Izt
        Y0 = beam.Y0
        Y = self.Y
        Ns = numerical.Ns
        Ntimes = numerical.Ntimes
        Iz0 = reference.Iz0
        s = reference.s

        for kz in range(len(z) - 1):

            # 1. Predict the arrival phase at the NEXT plane
            # Vnz = v0/v, so a larger Vnz means a longer flight time.
            # Acceleration within this interval is neglected by this Euler step
            self.Tnz[kz + 1, Y] = (
                self.Tnz[kz, Y]
                + (z[kz + 1] - z[kz]) * self.Vnz[kz, Y]
            )

            # 2. Optionally correct the SC contribution using the NEW current
            # accumulated at this plane. Izr[kz] only becomes available as the
            # march advances, so this correction cannot be assembled up front.
            # Otherwise keep the ordinary sum already prepared by track().
            if nn > 5 and iterf <= 0 and abs(Iz[kz, 0]) > 0.01:
                with np.errstate(divide="raise", invalid="raise"): # This is just to raise an error in case of inf or NaN
                    Esmz[kz, :] = (
                        Esz[kz, :] * self.Izr[kz, :] / Iz[kz, :] + Emz[kz, :]
                    )

            # 3. Field at the CURRENT plane and CURRENT arrival phases
            # The new Tnz[kz+1] is NOT used here.
            # Each electron therefore experiences its own instantaneous RF field.
            Est = np.zeros_like(t0, dtype=float)
            for kk in range(1, Ns + 1):
                Est[Y] += np.imag(
                    Esmz[kz, kk - 1] * np.exp(kk * 1.0j * self.Tnz[kz, Y])
                )
            Est[Y] += np.real(Esmz0[kz])

            # 4. Relativistic inverse-velocity Euler update (line 405).
            # The reference gamma minus one is 1/sqrt(1-s**2) - 1.
            # The electron's local gamma**(-3) is (1-s**2/Vnz**2)**(3/2).
            # For electrons, esign=1: positive Est increases Vnz and SLOWS v.
            self.Vnz[kz + 1, Y] = (
                self.Vnz[kz, Y]
                + esign * (z[kz + 1] - z[kz]) / s**2
                * (1.0 / np.sqrt(1.0 - s**2) - 1.0)
                * (1.0 - s**2 / self.Vnz[kz, Y] ** 2) ** 1.5
                * self.Vnz[kz, Y] ** 3 * Est[Y]
            )

            # An excessively large Euler step can leave the physical forward
            # range 0 < v < c. Report it before calculating invalid energies;
            # do not clip the velocity or silently return a completed pass.
            if (
                np.any(~np.isfinite(self.Vnz[kz + 1, Y]))
                or np.any(self.Vnz[kz + 1, Y] <= s)
            ):
                raise ValueError(
                    "The spatial Euler step left the physical forward-velocity "
                    "range. Reduce the grid spacing before continuing."
                )

            # OPTIONAL PHASE UPDATE 
            # It must stay AFTER Vnz[kz+1] is computed and BEFORE storing Tn
            # or accumulating Izr. It uses the change in INVERSE velocity:
            # Delta Tnz = Delta z * (Vnz_old + (Vnz_new - Vnz_old)/2).
            # This is a trapezoidal estimate of the flight-time integral.
            # The force still uses the old phase and velocity, so changing
            # only this line does NOT make the whole integrator second-order.
            # self.Tnz[kz + 1, Y] = (
            #     self.Tnz[kz, Y]
            #     + (z[kz + 1] - z[kz]) * self.Vnz[kz, Y]
            #     + 0.5 * (z[kz + 1] - z[kz])
            #     * (self.Vnz[kz + 1, Y] - self.Vnz[kz, Y])
            # )

            # 5. Record the grid crossing in the trajectory history
            # (lines 407--409). Vn stores v/v0, the reciprocal of Vnz.
            self.Zn[kz + 1, Y] = z[kz + 1]
            self.Vn[kz + 1, Y] = 1.0 / self.Vnz[kz + 1, Y]
            self.Tn[kz + 1, Y] = self.Tnz[kz + 1, Y]

            # 6. Accumulate the new beam quantities at this plane
            # (lines 410--418). There have been no reflected crossings yet.
            # Normalize with Nt, even when only part of the samples is active.
            self.IzDCr[kz + 1] = np.sum(Izt[Y0] / Iz0) / Nt
            for kk in range(1, Ns + 1):
                self.Izr[kz + 1, kk - 1] = (
                    2.0j / Nt
                    * np.sum(
                        Izt[Y0] / Iz0
                        * np.exp(-kk * 1.0j * self.Tnz[kz + 1, Y0])
                    )
                )
            self.QQDCr[kz + 1] = np.sum(
                Izt[Y0] / Iz0 * self.Vnz[kz + 1, Y0]
            ) / Nt
            self.Pe[kz + 1] = np.sum(
                (2.0 * np.pi / Nt) * (Izt[Y0] / Iz0)
                * ((1.0 - (s / self.Vnz[kz + 1, Y0]) ** 2) ** -0.5 - 1.0)
            )

            # 7. Request time tracking when the inverse velocity increases
            # too much over this step (lines 420--423). The newly reached
            # plane is already stored and accumulated before this test.
            if np.max(
                (self.Vnz[kz + 1, Y] - self.Vnz[kz, Y]) / self.Vnz[kz, Y]
            ) > numerical.evolutionmethod[0]:
                break

        # 8. Successful forward exit (lines 426--431). Reaching
        # the last plane counts as completion even if its last step triggered
        # the threshold. Additional Tn values are bookkeeping, not further
        # trajectories: the corresponding Zn and Vn entries remain infinity.
        if kz == len(z) - 2:
            self.Tn[len(z):, Y] = (
                self.Tn[len(z) - 1, Y]
                + (dt * np.arange(1, (Ntimes - 1) * len(z) + 1)).reshape(-1, 1)
            )
            self.Utn[Y] = (
                1.0e-3 * c**2 / etom
                * ((1.0 - (s / self.Vnz[len(z) - 1, Y]) ** 2) ** -0.5 - 1.0)
            )
            self.ttn[Y] = self.Tnz[len(z) - 1, Y]
            self.Yflag = len(Y)
            return None

        # 9. Handoff to track_t preparation (lines 433--434). kzstart is zero-based.
        # Keep the first handoff phase unchanged, then extend time by dt.
        # No electron has been marked as exiting: track_t must continue here.
        kzstart = kz + 1
        self.Tn[kzstart:, Y] = (
            self.Tn[kzstart, Y]
            + (dt * np.arange(Ntimes * len(z) - kzstart)).reshape(-1, 1)
        )
        return kzstart

    def track_t(self, numerical, beam, grid, reference, Esmz, Esmz0, kzstart):
        """Continue a spatial pass with fixed time steps (MATLAB 435--513).

        Call only when track_z returns an integer kzstart, on the SAME object
        and with the SAME Esmz working array. Do not initialize again. The
        handoff plane's crossing contributions are already included, and Tn
        already contains the remaining phases separated by grid.dt.

        Each electron is followed separately, starting at its own handoff
        phase. The fields stay fixed during this pass; this method neither
        solves the cavity circuit nor recomputes the space-charge harmonics.

        Parameters
        ----------
        numerical, beam, grid, reference
            The inputs used by initialize and track_z. The grid is uniform,
            with normalized spatial spacing equal to grid.dt, as in MATLAB.
            numerical.upstream_discard_distance_mm sets the reflected-electron
            cutoff relative to min(zi), in mm (MATLAB's GIFpara(4)). A backward
            crossing with its lower plane below this cutoff ends the electron's
            trajectory. Zero leaves upstream escape as the stopping condition.
            Tracking does not depend on DisplaySettings.
        Esmz, Esmz0
            Total normalized RF harmonics and DC field from track_z. Read-only
            here. RF phasors are interpolated before evaluating their imaginary
            parts at the electron's phase; the DC field's real part is added.
        kzstart
            Zero-based handoff plane/history row returned by track_z.

        Indices and normalizations
        --------------------------
        kt identifies ONE electron (a column). k is its history/time row.
        kz is its current spatial cell's lower plane; kzp is the previous
        cell index. kzrest is the fractional position within the current cell.
        Thus k and kz cease to mean the same thing after the handoff.
        Zn uses normalized position, Tn uses RF phase, and Vn = v/v0 is signed.
        Vnz stores its reciprocal at crossings, not the time-stepped velocity.

        MATLAB boundary conventions are retained: the final field value is
        used throughout the cell beginning at the last grid plane. Downstream
        exit is detected at the NEXT cell, one grid spacing beyond that plane.
        ttn is then the detection phase, not an interpolated last-plane phase;
        Tnz[-1, kt] stores the latter. Upstream exit is detected below z[0].

        Results
        -------
        Updates histories, crossing arrays, current/DC-charge/energy sums and
        downstream exit data in place. Returns None. Yflag counts downstream
        exits only: reflected/discarded or time-limited electrons do not add
        to it. Reaching the storage limit is NOT reported as a successful exit.
        QQr is untouched; reconstruct it only after the complete tracking pass.
        """

        z = grid.z
        zi = grid.zi
        dt = grid.dt
        Nt = beam.Nt
        Izt = beam.Izt
        Y0 = beam.Y0
        Ns = numerical.Ns
        Ntimes = numerical.Ntimes
        Iz0 = reference.Iz0
        s = reference.s
        upstream_cutoff_mm = np.min(zi) + numerical.upstream_discard_distance_mm

        for kt in Y0:
            kz = kzstart
            kzrest = 0.0

            for k in range(kzstart, Ntimes * len(z) - 1):
                # 1. Locate this electron. At the first step it is exactly
                # on the handoff plane; retaining kzstart avoids roundoff
                # in floor() incorrectly counting that crossing again.
                kzp = kz
                if k != kzstart:
                    kz = int(np.floor((self.Zn[k, kt] - z[0]) / dt))
                    kzrest = (self.Zn[k, kt] - z[0]) / dt - kz
                tt = self.Tn[k, kt]

                # 2. Check boundaries BEFORE advancing. Clamp only the crossing
                # range, not the stored trajectory or its velocity.
                upstream_exit = kz < 0
                downstream_exit = kz >= len(z)
                if upstream_exit:
                    kz = -1
                elif downstream_exit:
                    kz = len(z) - 1
                else:
                    # 3. Explicit Euler position update with the OLD velocity.
                    self.Zn[k + 1, kt] = self.Zn[k, kt] + dt * self.Vn[k, kt]

                    # 4. Field at the OLD position and phase. The last cell
                    # uses the last plane's field, exactly as in MATLAB.
                    if kz == len(z) - 1:
                        Est = np.sum(np.imag(
                            Esmz[kz, :] * np.exp(np.arange(1, Ns + 1) * 1.0j * tt)
                        )) + np.real(Esmz0[kz])
                    else:
                        Est = np.sum(np.imag(
                            ((1.0 - kzrest) * Esmz[kz, :] + kzrest * Esmz[kz + 1, :])
                            * np.exp(np.arange(1, Ns + 1) * 1.0j * tt)
                        )) + np.real((1.0 - kzrest) * Esmz0[kz] + kzrest * Esmz0[kz + 1])

                    # 5. Relativistic velocity update. Unlike Vnz in track_z,
                    # Vn is v/v0 and can pass through zero into backward motion.
                    # For electrons, positive Est reduces Vn.
                    self.Vn[k + 1, kt] = (
                        self.Vn[k, kt] - esign * dt / s**2
                        * (1.0 / np.sqrt(1.0 - s**2) - 1.0)
                        * (1.0 - s**2 * self.Vn[k, kt] ** 2) ** 1.5 * Est
                    )

                # 6. Account for crossings during the PREVIOUS interval k-1
                # to k, not the new k -> k+1 step. Several planes may be crossed.
                # Forward: exclude the old lower plane, include the new one.
                # Backward: include the old lower plane, exclude the new one.
                # At the handoff kz == kzp: no crossing is counted twice.
                if kz > kzp:
                    crossed = np.arange(kzp + 1, kz + 1)
                elif kz < kzp:
                    crossed = np.arange(kzp, kz, -1)
                else:
                    crossed = np.array([], dtype=int)

                if len(crossed) > 0:
                    self.Tnz[crossed, kt] = (
                        tt - dt
                        + (z[crossed] - self.Zn[k - 1, kt]) / self.Vn[k - 1, kt]
                    )
                    self.Vnz[crossed, kt] = 1.0 / self.Vn[k - 1, kt]

                    # Signed current and energy flux; charge residence time
                    # uses |v|, so a backward crossing still adds DC charge.
                    self.IzDCr[crossed] += (
                        Izt[kt] / (Nt * Iz0) * np.sign(self.Vn[k - 1, kt])
                    )
                    self.QQDCr[crossed] += (
                        Izt[kt] / (Nt * Iz0 * abs(self.Vn[k - 1, kt]))
                    )
                    for kk in range(1, Ns + 1):
                        self.Izr[crossed, kk - 1] += (
                            2.0j / Nt * Izt[kt] / Iz0
                            * np.exp(-kk * 1.0j * self.Tnz[crossed, kt])
                            * np.sign(self.Vn[k - 1, kt])
                        )
                    self.Pe[crossed] += (
                        (2.0 * np.pi / Nt) * Izt[kt] / Iz0
                        * np.sign(self.Vn[k - 1, kt])
                        * ((1.0 - (s * self.Vn[k - 1, kt]) ** 2) ** -0.5 - 1.0)
                    )

                # 7. Finish only this electron; continue with the next sample.
                # Upstream losses do not increment the downstream exit count.
                if upstream_exit:
                    break
                if downstream_exit:
                    self.Yflag += 1
                    self.Utn[kt] = (
                        1.0e-3 * c**2 / etom
                        * ((1.0 - (s * self.Vn[k - 1, kt]) ** 2) ** -0.5 - 1.0)
                    )
                    self.ttn[kt] = tt
                    break
                if kz < kzp and zi[kz] < upstream_cutoff_mm:
                    break
