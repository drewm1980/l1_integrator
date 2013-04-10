import casadi
from casadi import ssym, SXFunction
from casadi import sin, cos, fabs, IpoptSolver
import numpy as np
import numpy
import pylab

from integrators import integrate_trap

def solve_trapezoidal_direct(fc,u,t1,t2,N=100):
    F,_ = integrate_trap(fc,u,t1,t2,N)
    f = casadi.SXFunction([u],[F])
    f.init()
    solver = casadi.IpoptSolver(f)
    solver.setOption('generate_hessian',True)
    solver.init()
    solver.solve()
    print solver.output('x')
    return solver.output('x')

def solve_trapezoidal_slacks(fc,u,t1,t2,N=100):

    # Formulate the objective
    grid = np.linspace(t1,t2,N+1)
    dt = grid[1] - grid[0]
    fs = [fc(u,t) for t in grid] # f on grid
    ss = casadi.ssym('s',N+1) # slack variables for the absolute value terms
    F=0
    for i in xrange(len(fs)-1):
        fa=fs[i]
        fb=fs[i+1]
        ha=ss[i]
        hb=ss[i+1]
        ha_plus_hb=ha+hb
        F=F+ha_plus_hb + (fa*fb-ha*hb)/(ha_plus_hb + 1e-10)

    F=F*dt/2
    us = casadi.vertcat([u,ss])
    f = casadi.SXFunction([us],[F])
    f.init()

    # Constraints are ss[i] >= fs[i], ss[i] >= -fs[i],
    # so that ss[i] gets pushed down to fabsf(fs[i])
    fs_ = casadi.vertcat(fs)
    constraints = casadi.vertcat([ss-fs_,ss+fs_])
    g = casadi.SXFunction([us],[constraints])
    g.init()
    solver = casadi.IpoptSolver(f,g)
    solver.setOption('generate_hessian',True)
    solver.init()
    lbx = numpy.zeros((N+1+1,))
    solver.setInput(lbx,'lbx')
    solver.setInput(0.0,'lbg')
    solver.setInput(casadi.inf,'ubg')
    solver.solve()
    print solver.output('x')
    return solver.output('x')

if __name__=='__main__':
    from optimization_problems import u,t1,t2,fc
    u1 = solve_trapezoidal_direct(fc,u,t1,t2,N=100)
    #u2 = solve_trapezoidal_slacks(fc,u,t1,t2,N=100)
    
