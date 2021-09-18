# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 03:07:12 2021

@author: L
"""
import numpy as np
from numba import njit

def put_function(Sj,K):
    Zj=Sj.copy()
    ind=0
    for sj in Sj:
        Zj[ind]=np.max([K-sj,0])
        ind=ind+1
        
    return Zj

def CALL_function(Sj,K):
    Zj=Sj.copy()
    ind=0
    for sj in Sj:
        Zj[ind]=np.max([sj-K,0])
        ind=ind+1
        
    return Zj

def fact_f(mm):
    fact_mm=1
    if mm==0 or mm==1:
        return fact_mm
    else:
        for i in range(1,mm+1):
            fact_mm=fact_mm*i
        return fact_mm
        