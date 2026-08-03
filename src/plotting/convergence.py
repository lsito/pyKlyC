import numpy as np
import matplotlib.pyplot as plt

from solver.convergence import ConvergenceRecord


class ConvergencePlotter:
    """
    Interactive convergence-history plot.

    The plot displays the logarithm of the RF gap-voltage residual and the
    logarithm of the electron-efficiency residual as functions of nonlinear
    iteration number.
    """

    def __init__(self, Rlimit: float, power_ramp_mode: bool = False):
        """
        Create and initialize the convergence-history figure.

        Parameters
        ----------
        Rlimit:
            Residual threshold used as the convergence target.

        power_ramp_mode:
            If enabled, the vertical axis is extended upward to allow
            efficiency-like values to be displayed together with residuals.
        """

        self.Rlimit = Rlimit
        self.power_ramp_mode = power_ramp_mode

        self.fig, self.ax = plt.subplots()

        try:
            self.fig.canvas.manager.set_window_title("Convergence history")
        except Exception:
            pass

        self._initialize_axes()

    def _initialize_axes(self) -> None:
        """
        Initialize title, labels, limits, grid, legend, and convergence line.
        """

        self.ax.plot([0], [0], "ko", label="gap voltage")
        self.ax.plot([0], [0], "rs", label="electron efficiency")

        self.ax.set_title("iteration process (abort by closing this window)")
        self.ax.set_xlabel("Iteration number")
        self.ax.set_ylabel("log10(Residual)")

        self.ax.set_xlim(0, 1)

        ymin = np.log10(self.Rlimit) - 1.0
        ymax = 1.0 if self.power_ramp_mode else 0.0
        self.ax.set_ylim(ymin, ymax)

        self.ax.set_xticks([0, 1])
        self.ax.grid(True)

        self.ax.axhline(np.log10(self.Rlimit), linestyle="--")
        self.ax.legend(loc="lower left")

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def update(self, record: ConvergenceRecord) -> None:
        """
        Add one nonlinear iteration point to the convergence plot.
        """

        marker_face = "none"

        if record.electron_status == "reflected":
            marker_face = "yellow"
        elif record.electron_status == "trapped":
            marker_face = "cyan"

        self.ax.plot(
            record.iteration,
            record.log_residential,
            "ko",
            markerfacecolor=marker_face,
        )

        self.ax.plot(
            record.iteration,
            record.log_residentialel,
            "rs",
            markerfacecolor=marker_face,
        )

        self.ax.set_xlim(0, max(1, record.iteration))
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def is_open(self) -> bool:
        """
        Return whether the convergence-history figure still exists.
        """

        return plt.fignum_exists(self.fig.number)