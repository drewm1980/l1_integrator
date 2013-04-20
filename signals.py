# Functions for generating continuous test signals on domain [0,1],
# As well as parameterized sets of basis functions to use for
# representing the test signals

import casadi
import numpy

# One period of a sine wave
def fy_sin(t):
    return casadi.sin(t*2*casadi.pi)

# Polynomial Basis Functions centered around 0.5.
# Note: inherited from casadi.polyval is that the powers
# are in decreasing order,
# i.e. u_0*(t-.5)^N +...+ u_(N-1)*(t-.5)^1 + u_N
#u = casadi.ssym('u',N+1)
def fx_polybasis(u,t):
    f = casadi.sumAll(casadi.polyval(u,t-.5))
    return f


