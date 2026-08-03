from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ConvergenceRecord:
    """
    Residual information from one nonlinear iteration.

    Parameters
    ----------
    iteration:
        Iteration number.

    residential:
        Residual associated with the RF gap voltage convergence.

    residentialel:
        Residual associated with the electron-efficiency convergence.

    electron_status:
        Status of the particle motion during this iteration.
        Typical values are ``"normal"``, ``"reflected"``, or ``"trapped"``.
    """

    iteration: int
    residential: float
    residentialel: float
    electron_status: str = "normal"

    @property
    def log_residential(self) -> float:
        """
        Base-10 logarithm of the RF gap-voltage residual.
        """

        return math.log10(max(self.residential, 1e-300))

    @property
    def log_residentialel(self) -> float:
        """
        Base-10 logarithm of the electron-efficiency residual.
        """

        return math.log10(max(self.residentialel, 1e-300))


@dataclass
class ConvergenceHistory:
    """
    Sequence of convergence records accumulated during the nonlinear solve.
    """

    records: list[ConvergenceRecord] = field(default_factory=list)

    def append(self, record: ConvergenceRecord) -> None:
        """
        Store one nonlinear iteration record.
        """

        self.records.append(record)

    @property
    def last(self) -> ConvergenceRecord | None:
        """
        Most recent convergence record.
        """

        if not self.records:
            return None
        return self.records[-1]