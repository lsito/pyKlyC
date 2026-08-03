import logging

from model.beam import BeamParameters


logger = logging.getLogger(__name__)


def _validate_beam_type(beam: BeamParameters) -> None:
    """
    Validate the beam geometry selected for the one-dimensional solver.

    The current implementation supports cylindrical beams. Sheet-beam
    parameters are recognized but their numerical branches are not yet
    implemented.
    """

    if beam.is_cylindrical:
        logger.info(
            "Cylindrical beam simulation will be started shortly."
        )
        return

    if beam.is_sheet:
        raise NotImplementedError(
            "The sheet-beam branches in KLJCmainsv22 are incomplete."
        )

    raise ValueError(
        "The beam type is inconsistent: select either a nonzero rc with "
        "yd equal to zero, or rc equal to zero with nonzero yd."
    )