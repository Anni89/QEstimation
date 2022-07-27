'''
This file contains the class Obs(). 
With this class we generate m artificial observations from a given 'truth' for our twin experiments. 
We assume that the observation operator is linear and no interpolations are needed. 
The observations are given by the true model state plus a realisation of a normally distributed random variable 
with mean zero and diagonal covariance matrix whereby the variance is set by the attribute d_var.

Attributes: 
mr        : number of observations in space
mt        : number of observations in time
m = mt*mr : size of observation space
dmr       : number of spatial gridpoints between  observations
dmt       : number of temporal gridpoints between observations
d_var     : variance of observation error
u_truth   : truth from which we sample the artifical observations
start     : if start = 1, first observation at spatial location dmr, if start=0, first observation at spatial location 0
error     : realisation of error for this observation 
    
'''

import numpy as np
from numpy import random as rd

class Obs():
    def __init__(self,mr,mt,dmr,dmt,d_var,u_truth,start=1):
        self.mr      = mr               
        self.mt      = mt               
        self.m       = self.mr*self.mt  
        self.dmr     = dmr              
        self.dmt     = dmt              
        self.d_var   = d_var            
        self.u_truth = u_truth          
        self.start   = start            
        self.error   = rd.multivariate_normal(np.zeros(self.m), self.d_var*np.eye(self.m))
    
    # Generates array of observation times 
    def t_obs(self,nt):
        return np.arange(self.dmt,nt+1,self.dmt) 

    # Generates array of locations/variables which are observed 
    def r_obs(self,nr):
        return np.arange(self.start*self.dmr,nr,self.dmr)  
    
    # Generates observation error covariance matrix 
    def Cd(self):
        return self.d_var*np.eye(self.m) 
    
    # Generates observations from truth, returns vector of length m containing the observatiosn 
    # Note that we always observe at time dt ie u[1] for the first time 
    def d(self):
        d = np.zeros(self.m) 
        
        for n in range(self.mt):
            for j in range(self.mr):
                d[n*self.mr+j] = self.u_truth[1+n*self.dmt,self.start*self.dmr+j*self.dmr]

        #add noise        
        d = d + self.error
        
        return d

    # Generates observation operator applied to some vector u 
    def h(self,u):
        h = np.zeros(self.m)

        for n in range(self.mt):
            for j in range(self.mr):
                h[n*self.mr+j] = u[1+n*self.dmt,self.start*self.dmr+j*self.dmr]

        return h

    # Generates Jacobian of observation operator
    # H consists of matrices H=(H_0 ... H_{nt+1}) where each H_k is of size mxnr and H_0=H_{nt+1}=0
    def H(self,nt,nr):
        H = np.zeros((nt+2,self.m,nr))
        
        for n in range(self.mt):
            for j in range(self.mr):
                H[1+n*self.dmt,n*self.mr+j,self.start*self.dmr+j*self.dmr] = 1

        return H