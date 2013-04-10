import casadi
from casadi import ssym, SXFunction
from casadi import sin, cos, fabs
import numpy as np
import pylab

from integrators import integrate_trap, integrate_rect

def compare_midpoint_to_trapezoid_rule(u,fc,t1,t2):
    legends = []
    pylab.figure()
    pylab.hold('off')
    for integrator in [integrate_rect,integrate_trap]:
        for N in [3, 8, 16]:
            F1, grid1 = integrator(fc,u,t1,t2,N)
            ugrid = np.linspace(0.0,10.0,1000)
            F = [casadi.evalf(F1,u,uu) for uu in ugrid]
            pylab.plot(ugrid,F)
            pylab.hold('on')
            if integrator==integrate_rect:
                legends.append("Rectangular, N = " + str(N))
            else:
                legends.append("Trapezoidal, N = " + str(N))
    pylab.xlabel('u')
    pylab.ylabel('f')
    pylab.legend(legends) 
    pylab.title("Midpoint rule integration is not accurate or differentiable.")

def visualize_derivatives(u,fc,t1,t2):

    # Compute the Value, Gradient, and Hessian on an extremely
    # fine grid and finite differences to act as a referenence.
    BIGN = 1000
    F_truth, foo = integrate_rect(fc,u,t1,t2,BIGN)
    ugrid = np.linspace(0.0,10.0,BIGN)
    F = casadi.SXFunction([u],[F_truth])
    F.init()
    f_truth = np.zeros([BIGN])
    for i in xrange(BIGN):
        uu = ugrid[i]
        F.setInput(uu)
        F.evaluate()
        f_truth[i] = float(F.output())
    g_truth = np.zeros(BIGN-2)
    h_truth = np.zeros(BIGN-2)
    ugrid_truth = np.zeros(BIGN-2)
    du = ugrid[1]-ugrid[0]
    for i in xrange(1,BIGN-1):
        ugrid_truth[i-1] = ugrid[i]
        # Central approximation to Gradient
        g_truth[i-1] = (f_truth[i+1] - f_truth[i-1])/(2*du)
        # Central approximation to Hessian
        h_truth[i-1] = (f_truth[i-1] - 2*f_truth[i] + f_truth[i+1])/(du*du)

    # Compute the Value, Gradient, and Hessian on a coarsish grid
    F_SX, tgrid = integrate_trap(fc,u,t1,t2,20)
    F = casadi.SXFunction([u],[F_SX])
    F.init()
    FGH = F.hessian()
    FGH.init()
    f = np.zeros([BIGN])
    g = np.zeros([BIGN])
    h = np.zeros([BIGN])
    for i in xrange(BIGN):
        uu = ugrid[i]
        FGH.setInput(uu)
        FGH.evaluate()
        h[i] = float(FGH.output(0))
        g[i] = float(FGH.output(1))
        f[i] = float(FGH.output(2))

    pylab.figure()
    pylab.hold('off')
    pylab.plot(ugrid,f)
    pylab.hold('on')
    pylab.plot(ugrid,g)
    pylab.plot(ugrid,h)

    pylab.plot(ugrid,f_truth)
    pylab.plot(ugrid_truth,g_truth)
    pylab.plot(ugrid_truth,h_truth)

    pylab.xlabel('u')
    pylab.legend(['f','g','h','f_truth','g_truth','h_truth'])
    pylab.title('Trapezoidal Approximation Derivatives match those of original function.')

if __name__=='__main__':

    u = ssym("u") # Variable that is optimized
    def fc(u,t):
        return u*sin(.3*t) - cos(.4*t)
    t1 = 0.0; t2 = 1.0;

    compare_midpoint_to_trapezoid_rule(u,fc,t1,t2)
    visualize_derivatives(u,fc,t1,t2)
    pylab.show()

