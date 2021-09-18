# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 03:05:21 2021

@author: L
"""

from math import sqrt
from scipy.stats import norm
import numpy as np
from Function_util import *
from numba import njit


def brownian_m(W0, n, dt, delta, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        W(t) = W(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.
    
    Written as an iteration scheme,

        W(t + dt) = W(t) + N(0, delta**2 * dt; t, t+dt)

     """

    W0 = np.asarray(W0)

    # For each element of W0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=W0.shape + (n,), scale=delta*sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(W0, axis=-1)

    return out


def L_m(Sj,m,N):
    E=np.empty((N,m))
    for mm in range(m):
        l_m=np.zeros(N)
        for k in range(mm+1):
            lmc1=((-1)**k/fact_f(k))
            lmc2=(fact_f(mm)/(fact_f(k)*fact_f(mm-k)))
            lmc=lmc1*lmc2
            l_m=l_m+(lmc*(Sj**k))
        
        E[:,mm]=l_m
    
    
    return E
    
















