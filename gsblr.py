import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import math as m

def gibbs_bayesian_linreg(X, y, niter, np_rng=None):
    assert len(y) == X.shape[0]
    assert np_rng is not None
    
    n = X.shape[0]
    p = X.shape[1]
    
    # Create empty variables to store values.
    beta_estimates = np.empty((p, niter))
    sigma2_estimates = np.empty(niter)
    
    # Create initial mean and variance for beta.
    ibm = np.zeros(p)
    ibv = 100*np.eye(p)
    
    # Assign initial values for shape and scale for sigma2.
    a_0 = 0.01
    bs_0 = 0.01
    
    # Sample initial value of beta and sigma2.
    beta_estimates[:,0] = np_rng.multivariate_normal(ibm, ibv)
    sigma2_estimates[0] = sp.stats.invgamma.rvs(a=a_0, scale=bs_0)
    
    # Compute matrix multiplication that can be used at each iteration.
    XtX = X.T @ X
    Xty = X.T @ y
    bvi = 0.01*np.eye(p)
    bvi_m = bvi@ibm
    
    for i in range(1, niter):
        sigma2_inv = 1 / sigma2_estimates[i-1]
        
        # Compute A and b for the current iteration.
        A = XtX * sigma2_inv + bvi
        b = X.T@y * sigma2_inv + bvi_m
        A_inv = np.linalg.inv(A)
        
        # Generate new proposal beta.
        beta_prop = np_rng.multivariate_normal(A_inv@b, A_inv)
        
        # Generate new sigma.
        diff = np.reshape(y - X@beta_prop, (n, 1))
        shape = a_0 + n/2
        scale = bs_0 + 1/2 * diff.T@diff
        sigma2_prop = sp.stats.invgamma.rvs(a=shape, scale=scale)
        
        # Set new values.
        beta_estimates[:, i] = beta_prop
        sigma2_estimates[i] = sigma2_prop
        
    return beta_estimates, sigma2_estimates

class Gsblr:
    def __init__(self, rseed=None, burn_prop=0.5):
        assert burn_prop > 0 and burn_prop < 1
        self.burn_prop = burn_prop
        self.samples = None
        self.coef = None
        self.rng = np.random.default_rng(seed=rseed)
        np.random.seed(seed=rseed)
    
    def fit(self, X, y, niter=5000):
        # Gibbs sampling.
        beta, sigma2 = gibbs_bayesian_linreg(X=X, y=y, niter=niter, np_rng=self.rng)
        
        # Create dataframe of samples.
        gibbs_dict = {'beta'+str(i):beta[i-1] for i in range(1, beta.shape[0]+1)}
        gibbs_dict.update({'sigma2':sigma2})
        self.samples = pd.DataFrame(gibbs_dict)
        
        # Compute number of iterations to burn.
        burn_num = int(np.floor(self.samples.shape[0] * self.burn_prop))
        # Compute coefficients.
        self.coef = self.samples.iloc[burn_num:].mean(axis=0)
    
    def get_samples(self, var=False):
        if not var:
            return self.samples.iloc[:,:-1]
        else:
            return self.samples
    
    def get_coef(self, var=False):
        if not var:
            return self.coef[:-1]
        else:
            return self.coef
    
    def predict(self, X_test):
        betas = self.get_coef().to_numpy()
        betas.reshape(X_test.shape[1], 1)
        y_hat = X_test@betas
        return y_hat