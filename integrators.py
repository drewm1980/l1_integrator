import casadi
from casadi import ssym, SXFunction
from casadi import sin, cos, fabs
import numpy as np
import pylab

# Piecewise constant approximation of integral of |f|, using left point.
def integrate_rect(fc,u,t1,t2,N):
    grid = np.linspace(t1,t2,N+1)
    dt = grid[1] - grid[0]
    fs = [fc(u,t+dt/2) for t in grid[:-1]] # f at the midpoints
    Fi = [fabs(f) for f in fs]
    return dt*sum(Fi),grid

# Compute the area of a bowtie.
# Here, a "bowtie" consists of two right triangles
# sharing an ange.  ha and hb are the heights
# of the two triangles, assumed to be positive!
def bowtie_area(ha,hb,dt):
    intersection_point=ha/(ha+hb) # From t1, unscaled by dt
    area_a=intersection_point*ha # unscaled by 0.5, dt
    area_b=(1-intersection_point)*hb #unscaled by 0.5, dt
    area=(area_a+area_b)*dt*0.5
    return area

# Similar to bowtie_area, but for a single trapezoid
def trapezoid_area(ha,hb,dt):
    return (ha+hb)*dt*.5

# Piecewise linear (trapezoidal) approximation of integral of |f|
def integrate_trap(fc,u,t1,t2,N,implementation=2):
    grid = np.linspace(t1,t2,N+1)
    dt = grid[1] - grid[0]
    fs = [fc(u,t) for t in grid] # f on grid
    hs = [fabs(f) for f in fs] # absolute values of f on grid
    F=0
    if implementation==1:
        # This formulation theoretically allows short-circuiting, branch prediction
        for i in xrange(len(fs)-1):
            fa=fs[i]
            fb=fs[i+1]
            ha=hs(i)
            hb=hs(i+1)
            samesign=casadi.sign(fa)==casadi.sign(fb)
            F=F+casadi.if_else(samesign,trapezoid_area(ha,hb,dt),
                               bowtie_area(ha,hb,dt))

    if implementation==2:
        # This formulation theoretically allows use of SIMD (vector) extensions
        for i in xrange(len(fs)-1):
            fa=fs[i]
            fb=fs[i+1]
            ha=hs[i]
            hb=hs[i+1]
            ha_plus_hb=ha+hb
            F=F+ha_plus_hb + (fa*fb-ha*hb)/ha_plus_hb

        F=F*dt/2

    return F,grid

