import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import linalg as LA
from numpy import random as rd
from sklearn.metrics import mean_squared_error

import math 

from observations import Obs
from dataassimilation import WC4DVar
from dataassimilation import DA_methods

### function for correlation length fitting 
def func(x, l):
    return np.exp(-x**2/(2*l**2))


### function for Q estimation procedure 
def q_estimate(win,my_model,u0, p, nt,mr, mt, dmr, dmt,b_var,b_len,d_var,q_var_truth,l_truth):

    dt = my_model.dt
    nr = my_model.nr
    m  = mt*mr

    # allocate memory
    u_analysis = np.zeros((win,nt+1,nr))
    u_truth    = np.zeros((win,nt+1,nr))
    u_back     = np.zeros((win,nt+1,nr))
    rms        = np.zeros((win,2))

    var        = np.zeros(win)
    meanvar    = np.zeros(win)
    meanlen    = np.zeros(win)
    length     = np.zeros(win)

    H          = np.zeros((win,nt+2,m,nr))
    d          = np.zeros((win,m))

    beta       = np.zeros(((nt+1),nr))
    beta1      = np.zeros(((nt+1),nr))
    inno       = np.zeros(mt*mr)
    c          = np.zeros(((nt+1)*nr,1))
    ww         = np.zeros((nt,nr))
    w          = np.zeros((nt,nr))


    Cq0     = np.zeros((nr,nr,nt,nt))
    Cq      = np.zeros((nr,nr,nt,nt))

    #  background error covariance
    if b_len == 0:
         Cb    = my_model.error_covariance('white',b_var,0)
    else: 
         Cb    = my_model.error_covariance('spatial',b_var,b_len)

    # observation error variance
    R = d_var*np.eye(m) 

    # initial guess for model error covariance matrix: a multiple of the background error covariance matrix 
    for n in range(nt):
        Cq0[:,:,n,n] = Cb
  
    
    # true model error covariance matrix 
    if l_truth==0:
        Cq_truth    = my_model.error_covariance('white',q_var_truth,0)
    else:
        Cq_truth    = my_model.error_covariance('spatial',q_var_truth,l_truth)


        
    
    # iteration over assimilation windows 
    for j in range(win):
        print(f'ASSIMILATION WINDOW NUMBER {j}')

        #generate background
        if j==0:
            u_back[j,0] = u0.copy()
        else: 
            u_back[j,0] = u_analysis[j-1,-1].copy()
        for n in range(1,nt+1):
            u_back[j,n] = my_model.step_rk4(u_back[j,n-1])
    
        #generate truth 
        if j==0:
            u_truth[j,0]  = u0.copy()+rd.multivariate_normal(np.zeros(nr), Cb)
        else:
            u_truth[j,0]  = u_truth[j-1,-1].copy()
        for n in range(1,nt+1):
            u_truth[j,n] = my_model.step_rk4(u_truth[j,n-1])+rd.multivariate_normal(np.zeros(nr), Cq_truth)


        # generate observations
        my_Obs   = Obs(mr,mt,dmr,dmt,d_var,u_truth[j],0)
        H[j]     = my_Obs.H(nt,nr)
        d[j]     = my_Obs.d()

    
        # estimate of the model error covariance matrix for the data assimilation 
        if j == 0:
            Cq = Cq0
            
        else:
            #take the mean of the variances and correlation lengths from the previous assimilation windows 
            for n in range(nt):
                Cq[:,:,n,n] = my_model.error_covariance('spatial',meanvar[j-1],meanlen[j-1])


        ######################### DA STEP ##########################################      
        #perform data assimilation with new estimate for the model error covariance matrix 
        my_WC4DVar    = WC4DVar('SD')
        u_analysis[j] = my_WC4DVar.assimilate(my_model,nt,5,u_back[j,0],Cb,my_Obs,Cq)
        rms[j]        = my_WC4DVar.rms_compare(u_back[j],u_truth[j],u_analysis[j])
        print(f'RMS between background/truth and analysis/truth {rms[j]}')
    
        
        ######################## Q ESTIMATION ######################################
        # compute vector zd    
        beta[0]  = u_analysis[j,0]
        beta1[0] = u_back[j,0]
    
        for n in range(1,nt+1):
            beta[n,:]  = u_analysis[j,n]-my_model.step_rk4(u_analysis[j,n-1])
            beta1[n,:] = u_back[j,n]-my_model.step_rk4(u_back[j,n-1])
        
        zd = beta-beta1
        
        # compute innovations 
        for n in range(mt):
            inno[n*mr:(n+1)*mr] = d[j,n*mr:(n+1)*mr]-H[j,1+n*dmt,n*mr:(n+1)*mr]@u_back[j,1+n*dmt]

        # compute ctranspose 
        for n in range(1,nt+1):
            c[n*nr:(n+1)*nr,0] = np.transpose(inno)@H[j,n]
    
        # compute W matrix 
        for n in range(1,nt+1):
            for i in range(nr):
                ww[n-1,i] = 1/2*(1/mr*sum(zd[n,k*dmr]*c[n*nr+(k*dmr+i)%nr,0] for k in range(mr))+1/mr*sum((zd[n,(k*dmr+i)%nr])*(c[n*nr+k*dmr,0]) for k in range(mr)))
    
        if dmr!=1:
            for n in range(1,nt+1):
                for i in range(mr):
                    w[n-1,i*dmr] = ww[n-1,i*dmr]
        else: 
            w = ww

    
        # temporal averaging
        q2=1/mt*sum(w[k] for k in range(nt))

        # extract variance 
        var[j]=q2[0]
        # mean of the variances from the previous assimilation windows 
        if j==0:
            meanvar[0]=var[0]
        else: 
            meanvar[j]=np.mean(var[0:j+1])

        print(f'New estimate for variance coming from assimilation window {j}: {var[j]}')

        # extract correlation length 
        rr         = 1/q2[0]*q2[0:nr//2]
        xdata      = np.linspace(0,nr//2-1,nr//2)
        popt, pcov = curve_fit(func, xdata[0:p:dmr], rr[0:p:dmr])
        length[j]  = popt

        #tmean of correlation lengths from the previous assimilation windows 
        if j==0:
            meanlen[0] = length[0]
        else: 
            meanlen[j] = np.mean(length[0:j+1])
        
        print(f'New estimate for correlation length coming from assimilation window {j}: {length[j]}')

        # print the fit for the correlation length 
        plt.plot(xdata[0:p:1], func(xdata[0:p:1], *popt), 'r-')
        plt.plot(xdata[0:p:dmr], rr[0:p:dmr], 'b.', label='data')
        plt.show()

    
    return u_analysis, u_back, u_truth, var, length, meanvar, meanlen,  H, d
            

### functions for plotting results 

def q_estimate_analysis(win, var,length, meanvar, meanlen,q_var_truth,l_truth):

    print(f'estimated variances {var}')
    print(f'estimated correlation length {length}')

    x = np.linspace(0, win-1, win)
    
    fig, axes  = plt.subplots(nrows=2,ncols=2,figsize=(15,15))


    axes[0,0].plot(x,var, 'b.',  markersize=20, label='estimated variance')
    axes[0,0].plot(x,np.mean(var)*np.ones(win), 'm-', markersize=20,label='mean')
    axes[0,0].plot(x,q_var_truth*np.ones(win), 'r-',markersize=20, label='true variance')
    axes[0,0].set_xlabel('assimilation window number', fontsize=15)
    axes[0,0].set_ylim([0,2*q_var_truth])
    axes[0,0].tick_params(axis='x', labelsize=15)
    axes[0,0].tick_params(axis='y', labelsize=15)
    axes[0,0].legend(fontsize=15, loc='upper right')
    axes[0,0].set_title(f'mean = {np.mean(var):.4f} & std = {np.std(var):.4f}', fontsize=15)

    axes[0,1].plot(x,length, 'b.',  markersize=20, label='estimated correlation length')
    axes[0,1].plot(x,np.mean(length)*np.ones(win), 'm-', markersize=20,label='mean')
    axes[0,1].plot(x,l_truth*np.ones(win), 'r-',markersize=20, label='true correlation length')
    axes[0,1].set_xlabel('assimilation window number', fontsize=15)
    axes[0,1].set_ylim([0,2*l_truth])
    axes[0,1].tick_params(axis='x', labelsize=15)
    axes[0,1].tick_params(axis='y', labelsize=15)
    axes[0,1].legend(fontsize=15, loc='upper right')
    axes[0,1].set_title( f'mean = {np.mean(length):.4f} & std = {np.std(length):.4f}', fontsize=15)

    
    axes[1,0].plot(x,meanvar, 'b.',  markersize=20, label='averaged estimated variance ')
    axes[1,0].plot(x,q_var_truth*np.ones(win), 'r-',markersize=20, label='true variance')
    axes[1,0].set_xlabel('assimilation window number', fontsize=15)
    axes[1,0].set_ylim([0,2*q_var_truth])
    axes[1,0].tick_params(axis='x', labelsize=15)
    axes[1,0].tick_params(axis='y', labelsize=15)
    axes[1,0].legend(fontsize=15)
    
    axes[1,1].plot(x,meanlen, 'b.',  markersize=20, label='averaged estimated corr. length')
    axes[1,1].plot(x,l_truth*np.ones(win), 'r-',markersize=20, label='true correlation length')
    axes[1,1].set_xlabel('assimilation window number', fontsize=15)
    axes[1,1].set_ylim([0,2*l_truth])
    axes[1,1].tick_params(axis='x', labelsize=15)
    axes[1,1].tick_params(axis='y', labelsize=15)
    axes[1,1].legend(fontsize=15)


def evolution_over_windows(gp,win,mt,dmt,nt,dt,u_truth,u_back,u_analysis,H,d):
    t_obs = np.arange(1,mt*dmt+1,dmt)
    t     = np.linspace(0,nt*dt,nt+1)


    fig  = plt.figure(figsize=(10,5))
    axes = fig.add_axes([0,0,1,1])
    plt.axvline(x = 0, color = 'black')
    for i in range(win):
        plt.axvline(x = (i+1)*nt*dt, color = 'black')

        if i==0:
            axes.plot(i*nt*dt+t,u_truth[i,:,gp],color='blue',label='truth', markersize=20)
            axes.plot(i*nt*dt+t,u_back[i,:,gp],color='orange',label='background', markersize=20)
            axes.plot(i*nt*dt+t,u_analysis[i,:,gp],color='green',label='analysis', markersize=20)
            axes.plot(i*nt*dt+t_obs*dt,H[i,t_obs,:,gp]@d[i],'.', color='red',label='observation', markersize=20)
        else:
            axes.plot(i*nt*dt+t,u_truth[i,:,gp],color='blue', markersize=20)
            axes.plot(i*nt*dt+t,u_back[i,:,gp],color='orange', markersize=20)
            axes.plot(i*nt*dt+t,u_analysis[i,:,gp],color='green', markersize=20)
            axes.plot(i*nt*dt+t_obs*dt,H[i,t_obs,:,gp]@d[i],'.', color='red', markersize=20)

    axes.set_xlabel('time',fontsize=18)
    axes.set_ylabel('amplitude',fontsize=18)
    axes.set_title(f'state variable {gp}',fontsize=18)
    axes.legend(fontsize=18)
    axes.tick_params(axis='x', labelsize=20)
    axes.tick_params(axis='y', labelsize=20)


def confidence_interval(win,rep,meanvar,meanlen,q_var_truth,l_truth ):
    m  = np.zeros(win)
    c  = np.zeros(win)
    m1 = np.zeros(win)
    c1 = np.zeros(win)

    for i in range(win):
        m[i]  = np.mean(meanvar[:,i])
        c[i]  = 1.96 * np.std(meanvar[:,i])/np.sqrt(rep) 
        m1[i] = np.mean(meanlen[:,i])
        c1[i] = 1.96 * np.std(meanlen[:,i])/np.sqrt(rep)

    x = np.linspace(0, win-1, win)

    fig, ax = plt.subplots(1,2,figsize=(15,5))

    ax[0].plot(x,m,markersize=20,label='mean estimated variance')
    ax[0].fill_between(x,(m-c), (m+c), color='b', alpha=.1)
    ax[0].plot(q_var_truth*np.ones(win), 'r-',markersize=20, label='true variance')
    ax[0].set_ylim([0,2*q_var_truth])
    ax[0].set_xlabel('assimilation window number', fontsize=15)
    ax[0].tick_params(axis='x', labelsize=15)
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].legend(fontsize=15, loc='upper right')

    ax[1].plot(x,m1,markersize=20,label='mean estimated correlation length')
    ax[1].fill_between(x,(m1-c1), (m1+c1), color='b', alpha=.1)
    ax[1].plot(l_truth*np.ones(win), 'r-',markersize=20, label='true correlation length')
    ax[1].set_ylim([0,2*l_truth])
    ax[1].set_xlabel('assimilation window number', fontsize=15)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].legend(fontsize=15, loc='upper right')



