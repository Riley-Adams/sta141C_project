# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Implements a Metropolis-Hastings MCMC algorithm that uses random
# walk to generate new draws (only need prior, no likelihood)
def metropolis_hastings_rand_walk(prior, niter):
    x = []
    x.append(0)
   
    for i in range(1, niter):
        current = x[i-1]
        proposed = x[i-1] + np.random.normal(0, 1)
        ratio = prior(proposed) / prior(current)
        threshold = np.random.uniform(0, 1)

        if ratio > threshold:
            x.append(proposed)
        else:
            x.append(current)
            
    return x

def gibbs_bayesian(y, X, mu0, sig0, df0, psi0, niter):
    n = X.shape[0]
    p = X.shape[1]
    x_bar = np.sum(X, 0) / p
    beta_estimates = np.empty((p, niter))
    beta_estimates[:,0] = np.zeros(p)
    sigma_estimates = np.empty(niter)
    sigma_estimates[0] = 1
    beta_hat = np.linalg.inv(X.T@X)@X.T@y
    XtXinv = np.linalg.inv(X.T@X)
    
    for i in tqdm(range(1, niter)):
        sigma_current = sigma_estimates[i-1]
        
        # Obtain new estimates of betas
        mu = beta_hat
        sig = sigma_current*XtXinv
        proposal_beta = np.random.multivariate_normal(mu, sig)
        
        # Obtain new estimates of sigma^2
        diff = np.reshape(y - X @ proposal_beta, (n, 1))
        proposal_sigma = stats.invgamma.rvs(n/2, scale = diff.T @ diff / 2)
        
        # Set new value
        beta_estimates[:,i] = proposal_beta
        sigma_estimates[i] = proposal_sigma
            
    return beta_estimates, sigma_estimates

def metropolis_hastings_bayesian(X, p, prior, likelihood, niter):
    estimates = np.empty((p, niter))
    estimates[:,0] = np.ones(p) / p
    estimates[p-1] = 1
    
    for i in range(1, niter):
        current = estimates[:,i-1]
        proposed = np.random.multivariate_normal(current, 0.5*np.identity(p))
        proposed[p-1] = np.abs(proposed[p-1])
        log_ratio = (prior(proposed) + likelihood(X, proposed)) - (prior(current) + likelihood(X, current))
        print(log_ratio)
        ratio = np.exp(log_ratio)
        threshold = np.random.uniform(0, 1)
        
        if ratio > threshold:
            estimates[:,i] = proposed
        else:
            estimates[:,i] = current
            
    return estimates

def mvnorm_log_prior(x):
    n = len(x)
    return sp.stats.multivariate_normal.logpdf(x[0:n-1], mean = 50*np.ones(n-1), cov = x[n-1]) + \
        sp.stats.gamma.logpdf(x[n-1], 1, 1)
        
def mvnorm_log_likelihood(X, params):
    p = len(params)
    ll = 0
    for i in range(X.shape[1]):
        #print(sp.stats.multivariate_normal.logpdf(X[:,i], mean = X@params[0:p-1], cov = params[p-1]))
        ll += sp.stats.multivariate_normal.logpdf(X[:,i], mean = X@params[0:p-1], cov = params[p-1])
    return ll

#data = metropolis_hastings_rand_walk(sp.stats.norm.pdf, 10000)

#counts, bins = np.histogram(data)
#plt.hist(data, bins = np.linspace(-3, 3, 25))

X = pd.read_csv("data/X.txt", header = None)
Y = pd.read_csv("data/Y.txt", header = None)
Beta = pd.read_csv("data/Betas.txt", header = None)
Epsilon = pd.read_csv("data/err.txt", header = None)
G = pd.read_csv("data/G.txt", header = None)
sig2 = pd.read_csv("data/sig2", header = None)

X = np.array(X)#[0:150,0:100]
Y = np.array(Y).flatten()#[0:150].flatten()
Beta = np.array(Beta).flatten()#[0:100].flatten()
Epsilon = np.array(Epsilon).flatten()#[0:150].flatten()
G = np.array(G).flatten()#[0:100,0:100]

bayes_betas, bayes_sigmas = gibbs_bayesian(Y, X, Beta, np.identity(100), 0, np.identity(100), 10000)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
