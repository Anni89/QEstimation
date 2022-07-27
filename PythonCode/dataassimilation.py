'''
This file contains two classes:
- DA_methods() contains methods to analyse and plot DA results, these methods can be passed down to specific DA techniques (so far only one is implemented). 
- WC4DVar(DA_methods) contains the weak-constraint data assimilation technique based on the representer method.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA
from numpy import random as rd
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tnrange, tqdm_notebook #progress bars
import time



############################## class DA_methods() #########################################
class DA_methods():
    def __init__(self):
        pass
    
    # Returns the root mean squared error between the background and the truth & between the analysis and the truth 
    def rms_compare(self,u_back,u_truth,u_analysis):
        rms1 = mean_squared_error(u_back,u_truth,squared=False)
        rms2 = mean_squared_error(u_analysis,u_truth,squared=False)
    
        return rms1, rms2
    
    # Plots temporal evolution of certain variables of the truth, the analysis, the background and the observations 
    def plots_temporal(self,nt,u_truth,my_Obs,my_model,u_analysis,u_back,gridp):
        num       = len(gridp)
        fig, axes = plt.subplots(nrows=num,ncols=1,figsize=(12,15))
        plt.tight_layout()
        
        nr        = my_model.nr
        dt        = my_model.dt
        t         = my_model.time(nt)
        r_obs     = my_Obs.r_obs(nr)
        t_obs     = my_Obs.t_obs(nt)
        H         = my_Obs.H(nt,nr)
        d         = my_Obs.d()

        for j in range(num): 
            if j == 0: 
                axes[j].plot(t,u_truth[:,gridp[j]],label='truth')
                axes[j].plot(t,u_back[:,gridp[j]],label='background')
                axes[j].plot(t,u_analysis[:,gridp[j]],label='analysis')
                if gridp[j] in r_obs:
                    axes[j].plot(t_obs*dt,H[t_obs,:,gridp[j]]@d,'.', label='observation')
                axes[j].legend()

            else: 
                axes[j].plot(t,u_truth[:,gridp[j]])
                axes[j].plot(t,u_back[:,gridp[j]])
                axes[j].plot(t,u_analysis[:,gridp[j]])
                if gridp[j] in r_obs:
                    axes[j].plot(t_obs*dt,H[t_obs,:,gridp[j]]@d,'.')
        
            if j == num-1:  
                axes[j].set_xlabel('time')
    
            if j == num-1:
                axes[j].set_ylabel('amplitude')
        
            axes[j].set_title(f'spatial gridpoint {gridp[j]}')   

    # Plots true state, analysis state, background state and observations at certain points in time        
    def plots_spatial(self,nt,u_truth,my_Obs,my_model,u_analysis,u_back,gridp):
        num    = len(gridp)
        fig, axes = plt.subplots(nrows=num,ncols=1,figsize=(12,15))
        plt.tight_layout()
        
        nr     = my_model.nr
        dr     = my_model.dr
        r      = my_model.space()
        r_obs  = my_Obs.r_obs(nr)
        t_obs  = my_Obs.t_obs(nt)
        H      = my_Obs.H(nt,nr)
        d      = my_Obs.d()


        for j in range(num): 
            if j == 0: 
                axes[j].plot(r,u_truth[gridp[j]],label='truth')
                axes[j].plot(r,u_back[gridp[j]],label='background')
                axes[j].plot(r,u_analysis[gridp[j]],label='analysis')
                if gridp[j] in t_obs:
                    axes[j].plot(r_obs*dr,(H[gridp[j],:,r_obs]@d),'.', label='observation')
                axes[j].legend()

            else: 
                axes[j].plot(r,u_truth[gridp[j]])
                axes[j].plot(r,u_back[gridp[j]])
                axes[j].plot(r,u_analysis[gridp[j]])
                if gridp[j] in t_obs:
                    axes[j].plot(r_obs*dr,(H[gridp[j],:,r_obs]@d),'.')
        
            if j == num-1:  
                axes[j].set_xlabel('spatial r direction')
    
            if j == num-1:
                axes[j].set_ylabel('amplitude')
        
            axes[j].set_title(f'time steps {gridp[j]}')





############ class WC4DVar(DA_methods) #######################################

class WC4DVar(DA_methods):
    '''
    This implementation of the weak constraint data assimilation technique is based on Algorithm 4 in 'Data Assimilation Fundamentals' 
    by Geir Evensen, Femke C. Vossepoel, Peter Jan van Leeuwen (freely accessible via https://link.springer.com/book/10.1007/978-3-030-96709-3).

    Note that for each outer iteration we have to solve 
    $$(\mathcal R+C_{d})b=\eta-H\delta x.$$
    We set $\nu:=\eta-H\delta x$ and $C:=\mathcal R+C_{d}$, i.e. we end up with the linear equation
    $$Cb=\nu.$$
    The attribute 'solver' of this class indicates the method used for solving this linear equation. 
    One can choose between: 
    - SD : steepest descent (=gradient descent)
    - CG : conjugate gradient
    '''

    def __init__(self,solver):
        self.solver = solver        
        DA_methods.__init__(self) # inherits methods from class DA_methods     
    

    def assimilate(self,my_model,nt,outer,u0,Cb,my_Obs,Cq):
        '''
        This methods performs the assimilation procedure between the model my_model and the observations my_Obs, given 
        - the background error covariance matrix Cb,
        - the assigned model error covariance matrix Cq
        - the background state u0
        - the length of the assimilation window nt
        - the number of outer iterations 

        Note that the total nunber of observations needs to be m>1. 
        '''  

        # retireve more parameters from my_model and my_Obs 
        nr = my_model.nr
        dt = my_model.dt
        d  = my_Obs.d()
        Cd = my_Obs.Cd()
        m  = my_Obs.m
        H  = my_Obs.H(nt,nr)
        
        
        # compute background
        u_back    = np.zeros((nt+1,nr))
        u_back[0] = u0.copy()
        for n in range(1,nt+1):
            u_back[n] = my_model.step_rk4(u_back[n-1]) 
        
        
        # initial guess for b and u
        b = np.zeros(m) 
        u = u_back.copy()
        
        # allocate memory for the values of the cost function and its different components 
        J  = np.zeros(outer)
        Jb = np.zeros(outer)
        Jo = np.zeros(outer)
        Jm = np.zeros(outer)
        
        
        ###########################################################################################
        # outer iteration 
        for i in tnrange(outer, desc='Outer iteration'):
            # allocate memory 
            xi = np.zeros((nt+1,nr))
            du = np.zeros((nt+1,nr))
            eta = np.zeros(m)
        
        
            # compute xi and eta 
            xi[0] = u_back[0]-u[0]
            for n in range(nt):
                xi[n+1] = u[n+1]-my_model.step_rk4(u[n])
            eta = d-my_Obs.h(u)
        
        
            # compute first guess residual
            du_f    = np.zeros((nt+1,nr))
            du_f[0] = xi[0].copy()
            for n in range(nt):
                du_f[n+1] = my_model.TLM_step(u[n],du_f[n])-xi[n+1]
        
    
            # compute the right hand side, note that H(nt,nr)[0]=0=H(nt,nr)[nt+1], note that range(1,nt+1) is 1...nt
            nu = eta-sum(my_Obs.H(nt,nr)[s]@ du_f[s] for s in range(1, nt+1))
        
        
            ##################   STEEPEST DESCENT also called Gradient descent ##################################### 
            if self.solver == 'SD':
                
                # allocate memory 
                psi  = np.zeros((nt+2,nr))
                psi1 = np.zeros((nt+2,nr))
                c    = np.zeros((nt+1,nr))
                c1   = np.zeros((nt+1,nr))
                
                # compute the initial residual 
                for k in range(nt,-1,-1):
                    psi[k] = my_model.AM_step(u[k],psi[k+1])+H[k].T @ b

                c[0] = Cb @ psi[0]
                for k in range(1,nt+1):
                    c[k] = my_model.TLM_step(u[k-1],c[k-1])+sum(Cq[:,:,k-1,s-1]@ psi[s] for s in range(1, nt+1))
                    
                res = nu-(sum(H[s]@ c[s] for s in range(1, nt+1))+Cd@b)
                
                rold = res@res
             

                #set maximal number of iterations for the iterative solver and stopping criterium for the while loop 
                max_it = 1000
                it     = 0
                crit   = 1 
               
                # inner iteration 
                while (crit>10**-6 and it<max_it):
           
                    print(f'Current inner iteration index {it}',end='\r') 
            
                    # compute alpha
                    for k in range(nt,-1,-1):
                        psi1[k] = my_model.AM_step(u[k],psi1[k+1])+H[k].T @ res
                    
                    c1[0] = Cb@my_model.AM_step(u[0],psi1[1])
                    for k in range(1,nt+1):
                        c1[k] = my_model.TLM_step(u[k-1],c1[k-1])+sum(Cq[:,:,k-1,s-1]@ psi1[s] for s in range(1, nt+1))
                    alpha = res.T@res/(res.T@ (sum(H[s]@ c1[s] for s in range(1, nt+1))+Cd@res))

                    # update b
                    b = b+alpha*res 
                    
                    # update the residual 
                    res = res-alpha*(sum(H[s]@ c1[s] for s in range(1, nt+1))+Cd@res)
                    
                   
                    rnew = res@res
                    crit = rnew
                    it   = it+1
                   
            
            ################ CONJUGATE GRADIENT ###################################################################
            elif self.solver == 'CG':
            ## this needs to be cleaned, not ready to be used 

                # allocate memory
                psi   = np.zeros((nt+2,nr))
                psi0  = np.zeros((nt+2,nr))
                c     = np.zeros((nt+1,nr))
                c0    = np.zeros((nt+1,nr))

                res  = np.zeros(m)
                Ap = np.zeros(m)
                Ab = np.zeros(m)
        
                ##########Compute the product Cb
                if m == 1:
                    for k in range(nt,-1,-1):
                        psi0[k] = my_model.AM(u[k]) @ psi0[k+1]+H[k].T * b
                    #c0[0] = Cb @ psi0[0]
                    c0[0]=Cb@my_model.AM(u[0])@psi0[1]
                    for k in range(1,nt):
                        c0[k] =  my_model.TLM(u[k-1])@c0[k-1]+sum(Cq[:,:,k-1,s-1]@ psi0[s] for s in range(1, nt+1))
                    Ab = sum(H[s]@ c0[s] for s in range(1, nt+1))+Cd*b

                else:
                    for k in range(nt,-1,-1):
                        psi0[k] = my_model.AM(u[k]) @ psi0[k+1]+H[k].T @ b
                    #c0[0] = Cb @ psi0[0]
                    c0[0]=Cb@my_model.AM(u[0])@psi0[1]
                    for k in range(1,nt):
                        c0[k] =  my_model.TLM(u[k-1])@c0[k-1]+sum(Cq[:,:,k-1,s-1]@ psi0[s] for s in range(1, nt+1))
                    Ab = sum(H[s]@ c0[s] for s in range(1, nt+1))+Cd@b

                ###########
        
                res = nu-Ab #compute initial residual 
                p = res.copy()
                if m == 1:
                    rinit=res*res
                else:
                    rinit= res@res
                    
                crit=1
                max_it = 1000
                it     = 0
                
                rsold=rinit.copy()
                
                while(crit>10**-6 and it<max_it):
                    print(f'Current inner iteration index {it}',end='\r')
                    #Compute C*d
                
                    if m==1:
                        for k in range(nt,-1,-1):
                            psi[k] = my_model.AM(u[k]) @ psi[k+1]+H[k].T * p
                        #c[0] = Cb @ psi[0]
                        c[0]=Cb@my_model.AM(u[0])@psi[1]
                        for k in range(1,nt):
                            c[k] =  my_model.TLM(u[k-1])@c[k-1]+sum(Cq[:,:,k-1,s-1]@ psi[s] for s in range(1, nt+1))
                        Ap = sum(H[s]@ c[s] for s in range(1, nt+1))+Cd*p

                        alpha = rsold/(p*Ap)
                    else:
                        for k in range(nt,-1,-1):
                            psi[k] = my_model.AM(u[k]) @ psi[k+1]+H[k].T @ p
                        #c[0] = Cb @ psi[0]
                        c[0]=Cb@my_model.AM(u[0])@psi[1]
                        for k in range(1,nt):
                            c[k] =  my_model.TLM(u[k-1])@c[k-1]+sum(Cq[:,:,k-1,s-1]@ psi[s] for s in range(1, nt+1))
                        Ap = sum(H[s]@ c[s] for s in range(1, nt+1))+Cd@p
                        
                        alpha = rsold/(p@Ap)


                    b  = b + alpha*p
            
                    res  = res - alpha*Ap
            
                    if m==1:
                        rsnew = res*res
                    else:
                        rsnew = res@res
                    
                    if np.sqrt(rsnew)<1e-8:
                        break
            
                    beta = rsnew/rsold
                    p    = res+beta*p 
            
                    rsold = rsnew
                    
                    crit=rsold/rinit
                    
                    it = it+1
    
                
            #############################################################################################    
                    
            print(f'Number of inner iterations in outer iteration number {i+1}: {it}')
            
            #####################################################################################################
            # backward integration
            dlam = np.zeros((nt+2,nr))
            for k in range(nt,-1,-1):
                dlam[k] =  my_model.AM_step(u[k],dlam[k+1])+H[k].T @ b

    
            # forward integration
            du[0] = xi[0]+Cb@my_model.AM_step(u[0],dlam[1])
            for k in range(1,nt+1):
                du[k] = my_model.TLM_step(u[k-1],du[k-1])-xi[k]+sum(Cq[:,:,k-1,s-1]@ dlam[s] for s in range(1,nt+1))
    
            u += du

            #######################################################################
            # compute value of the incremental cost function for the increment du
            # strong constraint 
            if (Cq==np.zeros((nr,nr,nt,nt))).all():
                Jb[i] = 0.5*(du[0]-xi[0]).T@LA.pinv(Cb,rcond=1e-3)@(du[0]-xi[0])
                Jo[i] = 0.5*(sum(H[s]@ du[s] for s in range(1, nt+1))-eta).T@LA.pinv(Cd,rcond=1e-3)@(sum(H[s]@ du[s] for s in range(1, nt+1))-eta)
                J[i]  = Jb[i]+Jo[i]
            # weak constraint 
            else:    
                Jb[i] = 0.5*(du[0]-xi[0]).T@LA.pinv(Cb,rcond=1e-3)@(du[0]-xi[0])
                Jo[i] = 0.5*(sum(H[s]@ du[s] for s in range(1, nt+1))-eta).T@LA.pinv(Cd,rcond=1e-3)@(sum(H[s]@ du[s] for s in range(1, nt+1))-eta)
                Jm[i] = 0.5*sum((du[r]-my_model.TLM_step(u[r-1],du[r-1])+xi[r]).T@LA.pinv(Cq[:,:,r-1,r-1],rcond=1e-3)@(du[r]-my_model.TLM_step(u[r-1],du[r-1])+xi[r]) for r in range(1,nt+1))
                J[i]  = Jb[i]+Jo[i]+Jm[i]
        
        print(f'Background contribution to cost function {Jb}')
        print(f'Observation contribution to cost function {Jo}')
        print(f'Model contribution to cost function  {Jm}')
        print(f'The values of the cost function total {J}')


        return u
