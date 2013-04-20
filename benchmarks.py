#!/usr/bin/env python

from optimization_problems import problems
from solvers import solvers

for p in problems[1:2]:
    for solver in solvers:
        #for N in [4,8,16,32,64,128,256]:
        for N in [4,8]:
            u,fc,t1,t2 = p.u,p.fc,p.t1,p.t2
            solver(fc,u,t1,t2,N)
            
            
    
