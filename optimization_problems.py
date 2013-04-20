import casadi
from casadi import ssym, SXFunction
from casadi import sin, cos, fabs
import numpy as np
import pylab

from collections import namedtuple

OptimizationProblem = namedtuple('OptimizationProblem','u NU fc t1 t2')
problems = [] # A list of optimization problems

u = ssym("u") # Variable that is optimized
t1 = 0.0; t2 = 1.0;
# Toy example 
def fc(u,t):
    assert u.numel()==1
    return u*sin(.3*t) - cos(.4*t)
problems.append( OptimizationProblem(u=u,fc=fc,t1=t1,t2=t2, NU=1) )

from signals import fy_sin,fx_polybasis
N = 7
NU = N+1
u = casadi.ssym('u',NU)
def fc2(u,t):
    return fx_polybasis(u,t) - fy_sin(t)
problems.append( OptimizationProblem(u=u,fc=fc2,t1=t1,t2=t2, NU=NU) )

p = problems[0]
u,fc,t1,t2 = p.u,p.fc,p.t1,p.t2

