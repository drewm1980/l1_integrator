import casadi
from casadi import ssym, SXFunction
from casadi import sin, cos, fabs
import numpy as np
import pylab

u = ssym("u") # Variable that is optimized
def fc(u,t):
    return u*sin(.3*t) - cos(.4*t)
t1 = 0.0; t2 = 1.0;

