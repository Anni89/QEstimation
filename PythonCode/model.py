"""
This file contains two classes: 
- Model() 
- Lorenz96(Model) 
See below for further information. 

"""

import numpy as np
import matplotlib.pyplot as plt


###################  class Model() ################################
class Model():
    """ 
    The Model class contains several methods that are relevant for objects of the Lorenz96 model (see below). 
    We have separated these methods as they are relevant for other models as well that might be implemented in the future. 
    In particular the class contains numerical methods for solving ODEs such as the Euler method and Runge-Kutta.
    """
    def __init__(self):
        pass
    
    ############ ODE solvers ##########
    #Runge-Kutta method
    def step_rk4(self,u):
        k1 = self.dt*self.rhs(u)
        k2 = self.dt*self.rhs(u+k1/2.0)
        k3 = self.dt*self.rhs(u+k2/2.0)
        k4 = self.dt*self.rhs(u+k3)
    
        return u+(k1+2.0*(k2+k3)+k4)/6.0
    
    #Euler method
    def step_euler(self,u):
        return u+self.dt*self.rhs(u)

   
    ########## Generation of spatial error covariance matrices of Gaussian type ##########
    # Note that correlation length len is given in pixel not in physical length.
    # Note that we assume periodic boundaries, as for example given in the Lorenz96 model. 

    def error_covariance(self , corr_type , var , len=0):
        C = np.zeros((self.nr,self.nr))
        
        if corr_type == 'white':
            C = var*np.eye(self.nr)
                   
        if corr_type == 'spatial':
            for x in range(self.nr):
                C[x,x] = var
                for y in range(1,(self.nr)//2+1):
                    C[x,(x+y)%self.nr] = var*np.exp(-(y)**2/(2*len**2))
                    C[x,x-y]           = var*np.exp(-(y)**2/(2*len**2))

        return C


    ################### Plotting methods #####################
    # Generates array of temporal gridpoints for the model given the total time nt, including the time =0, i.e. nt+1 gridpoints in total
    def time(self,nt):
        return np.linspace(0,nt*self.dt,nt+1) 
        
    # Generates array of spatial gridpoints for the model given the spatial dimension and step length
    def space(self):
        return np.linspace(0,(self.nr-1)*self.dr,self.nr) 
    

    # Plots the model variables at different points in time 
    def spatial_plot(self,title,times=[0],*u):
        fig  = plt.figure()
        axes = fig.add_axes([0.1,0.1,1.5,0.8])
        
        for arg in u:
            for t in times:
                axes.plot(self.space(),np.atleast_2d(arg)[t], label=f't={t}')
        
        axes.set_xlabel('state variable')
        axes.set_ylabel('amplitude')
        axes.legend()
        axes.set_title(title)

    # Plots the temporal evolution of different model variables 
    def temporal_plot(self,title,gridp=[0],*u):
        fig  = plt.figure()
        axes = fig.add_axes([0.1,0.1,1.5,0.8])
        
        for arg in u:
            for x in gridp:
                axes.plot(self.time(nt),np.atleast_2d(arg)[:,x], label=f'variable={x}')
       
        axes.set_xlabel('time')
        axes.set_ylabel('amplitude')
        axes.legend()
        axes.set_title(title)

    # Plots the true model state, the background model state and observations at a certain point in time       
    def spatial_plot_observations(self,time,my_Obs,u_truth,u_back,nt):
        fig  = plt.figure()
        axes = fig.add_axes([0.1,0.1,1.5,0.8])
        axes.plot(self.space(),u_truth[time,:],label=f'truth')
        axes.plot(self.space(),u_back[time,:],label=f'background')

        if time in my_Obs.t_obs(nt):
            axes.plot(my_Obs.r_obs(self.nr)*self.dr,(my_Obs.H(nt,self.nr)[time,:,my_Obs.r_obs(self.nr)]@my_Obs.d()),'.', label=f'observations')
       
        axes.set_xlabel('state variable')
        axes.set_ylabel('amplitude')
        axes.legend()
        axes.set_title(f'at time step {time}')
    
    # Plots the evolution of a certain variable of the true model state, the background model state and observations over time      
    def temporal_plot_observations(self,location,my_Obs,u_truth,u_back,nt):
        fig  = plt.figure()
        axes = fig.add_axes([0.1,0.1,1.5,0.8])
        axes.plot(self.time(nt),u_truth[:,location],label=f'truth')
        axes.plot(self.time(nt),u_back[:,location],label=f'background')
        if location in my_Obs.r_obs(self.nr):
            axes.plot(my_Obs.t_obs(nt)*self.dt,(my_Obs.H(nt,self.nr)[my_Obs.t_obs(nt),:,location]@my_Obs.d()),'.', label=f'observations')
        axes.set_xlabel('time')
        axes.set_ylabel('amplitude')
        axes.legend()
        axes.set_title(f'state variable {location}')
        
    
##################### class Lorenz96(Model) ##########################

class Lorenz96(Model):
    """
    Objects of this class define the Lorenz96 model.  
    This class inherits methods from the class Model. 
    
    Attributes: 
    F     : forcing strength
    nr    : space dimension 
    dt    : temporal step size 
    dr    : 1 spatial gridsize, note that we do not discretise spatially, this is not a PDE
    """
    
    def __init__(self,nr,F,dt):
        self.nr = nr
        self.F  = F
        self.dt = dt
        self.dr = 1
        
        Model.__init__(self) #inherits methods from class Model 
    

    # Returns the right hand side of the ODE model at model state u
    def rhs(self,u):
        f = np.zeros(self.nr)
        
        for i in range(self.nr):
            f[i] = (u[(i+1)%self.nr]-u[i-2])*u[i-1]-u[i]+self.F
        
        return f 
    
    # Returns Jacobi matrix of rhs(u) applied to a vector u_b
    def jacob(self,u_b, u):
        jacob = np.zeros(self.nr)
        
        for i in range(self.nr):
            jacob[i] = (u[(i+1)%self.nr] - u[i-2])*u_b[i-1] + (u_b[(i+1)%self.nr] - u_b[i-2])*u[i-1] - u[i]
            
        return jacob
    
    # Returns ?
    def jacoba(self,u_b, u):
        jacoba = np.zeros(self.nr)

        for i in range(self.nr):
            jacoba[i] = (u_b[(i+2)%self.nr] - u_b[i-1])*u[(i+1)%self.nr] -u_b[(i+1)%self.nr]*u[(i+2)%self.nr] + u_b[i-2]*u[i-1]- u[i]
            
        return jacoba
    
    # Returns state of the tangent linear model (linearised arounf u_b) after one time step starting with u 
    # based on Runge-Kutta
    def TLM_step(self,u_b,u):
        k1  = self.dt*self.rhs(u_b)
        k2  = self.dt*self.rhs(u_b+ 0.5*k1)
        k3  = self.dt*self.rhs(u_b+ 0.5*k2)
        dk1 = self.dt*self.jacob(u_b,u)
        dk2 = self.dt*self.jacob(u_b + 0.5*k1, u+ 0.5*dk1)
        dk3 = self.dt*self.jacob(u_b + 0.5*k2, u + 0.5*dk2)
        dk4 = self.dt*self.jacob(u_b + k3, u + dk3)

        return u + (dk1 + 2.0*dk2 + 2.0*dk3 + dk4)/6.0
    
    # Returns state of the adjoint model (linearised arounf u_b) after one time step starting with u 
    # based on Runge-Kutta
    def AM_step(self,u_b,u):
        k1    = self.dt*self.rhs(u_b)
        k2    = self.dt*self.rhs(u_b + 0.5*k1)
        k3    = self.dt*self.rhs(u_b + 0.5*k2)
        dk1_a = self.dt*self.jacoba(u_b + k3, u)
        dk2_a = self.dt*self.jacoba(u_b + 0.5*k2, u + 0.5*dk1_a)
        dk3_a = self.dt*self.jacoba(u_b + 0.5*k1, u + 0.5*dk2_a)
        dk4_a = self.dt*self.jacoba(u_b, u + dk3_a)

        return u + (dk1_a + 2.0*dk2_a + 2.0*dk3_a + dk4_a)/6.0        
