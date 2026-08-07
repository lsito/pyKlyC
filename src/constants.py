"""
Physical and numerical constants used throughout the simulation package.

Physical constants use SI units. Numerical constants associated with
cylindrical eigenfunctions are calculated with SciPy special functions.
"""

import numpy as np
from scipy.constants import (
    c as c_scipy, # speed of light in vacuum m/s
    physical_constants,
)
from scipy.special import jn_zeros


# Values used in KLJCmainsv22.m are active so that numerical comparisons with
# MATLAB are not affected by differences in the physical-constant revision.
c: float = 299792458.0
# Higher-precision Python version: c = c_scipy

etom_signed: float = -1.7588e11
# Higher-precision Python version:
# etom_signed = physical_constants["electron charge to mass quotient"][0]

# For negative charged paticles, esign=1 
# For postive charged particles, esign=-1

esign: float = float(np.sign(-etom_signed))

# Electron charge-to-mass ratio in coulombs per kilogram amplitude
etom: float = abs(etom_signed)

# Characteristic impedance of vacuum in ohms
eta0: float = 4.0e-7 * np.pi * c
# Higher-precision Python version:
# eta0 = physical_constants["characteristic impedance of vacuum"][0]

# Zero of the zeroth-order Bessel function, J0
mu01: float = 2.40483
# Higher-precision Python version: mu01 = float(jn_zeros(0, 1)[0])
