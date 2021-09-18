# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 02:03:42 2021

@author: L
"""
import numpy as np
from  Functions_use import *
from Function_util import *
from scipy import linalg as lg
from pylab import plot, show, grid, xlabel, ylabel

#date échéance
T=1
print('date échéance : %d'%T)
#prix d’exercice strike
K=50  #euro 
print('prix d’exercice strike  :%d' %K)
print('////////////////////////////////////////////')
# Number of paths
N=5000

########  N realizations  of Brownian motion  ######

# The Wiener process parameter.
delta = 0.2
# Number of steps.
N_0=10
N_w=T*(N_0+1)
# Time step size
dt = T/(N_w)
# Create an empty array to store the realizations.
W = np.empty((N,N_w+1))
# Initial values of x.
W[:, 0] = 0

brownian_m(W[:,0], N_w+1, dt, delta, out=W[:,:])



t = np.linspace(0.0, (N_w)*dt, N_w+1)

'''
for k in range(N):
    plot(t, W[k])
xlabel('t', fontsize=16)
ylabel('W', fontsize=16)
grid(True)
show()
'''


########  N realizations  of S  ######
print('%d realizations  of S '%N)
W=W[:,:N_w+1:N_0+1]

mu=0.8
sigma=0.8
S0=50
S=np.empty((N,T+1))
for j in range(T+1):
    for n in range(N):
        S[n,j]=S0*np.exp( (mu-0.5*(sigma**2))*j+ sigma*(W[n,j]) )
    
print('S0 = %f'%S0)
########  N realizations  of Z  ######
print('%d realizations  of Z '%N)
Z=np.empty((N,T+1))
for n in range(N):
    Z[n,:]=CALL_function(S[n,:],K)


########  Basis matrix e(S) ######
print('Basis matrix')
m=5000
E_b=np.empty((T-1,N,m))
SS=S[:,1:T+1]
for j in range(T-1):
    EE=L_m(SS[:,j],m,N)
    E_b[j]=EE



##################### algo LSM #################""
print('LSM')
Tho=np.zeros((N,T))
Tho[:,-1]=T*np.ones(N)  
for j in range(T-1,0,-1):
    Eb=E_b[j-1,:,:]
    Mj=np.empty((m,m))
    for k in range(m):
        for l in range(m):
            Mj[k,l]=(1/N)*np.sum(Eb[:,k]*Eb[:,l])
            
    sj=np.zeros(m)
    for n in range(N):
        ZTho_j1_n=Z[n,int(Tho[n,j])]
        sj=sj+(1/N)*(ZTho_j1_n*Eb[n,:])
    
    ALphaj=np.dot(lg.inv(Mj),sj)  
    for n in range(N):
        test=np.sum(ALphaj*Eb[n,:])
        if Z[n,j]>=test:
            Tho[n,j-1]=j
        else:
            Tho[n,j-1]=Tho[n,j]
        


Z0=np.max([S0-K,0])
print('Z0 = %f'%Z0)
################ V0 ######################
term2=0.
for n in range(N):
    jj=int(Tho[n,0])
    term2=term2+(1./N)*Z[n,jj]
            
V0=np.max([Z0,term2])  
print("V0= %.30f"%V0)      
          
        
        












