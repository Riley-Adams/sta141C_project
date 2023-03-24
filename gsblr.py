import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import math as m

def gibbs_bayesian_linreg(X, y, niter, np_rng=None):
    # Check that input arrays have the correct dimensions
    assert len(y) == X.shape[0]
    assert np_rng is not None
    
    # Get the number of data points (n) and the number of features (p)
    n = X.shape[0]
    p = X.shape[1]
    
    # Create empty arrays to store estimates of beta coefficients and sigma2 values
    beta_estimates = np.empty((p, niter))
    sigma2_estimates = np.empty(niter)
    
    # Set initial values for beta's mean and variance
    ibm = np.zeros(p)
    ibv = 100*np.eye(p)
    
    # Set initial values for the shape and scale parameters of the inverse gamma distribution
    a_0 = 0.01
    bs_0 = 0.01
    
    # Sample initial value of beta and sigma2.
    beta_estimates[:,0] = np_rng.multivariate_normal(ibm, ibv)
    sigma2_estimates[0] = sp.stats.invgamma.rvs(a=a_0, scale=bs_0)
    
    # Precompute values for matrix multiplications to improve efficiency
    XtX = X.T @ X
    Xty = X.T @ y
    bvi = 0.01*np.eye(p)
    bvi_m = bvi@ibm
    
    # Iterate through Gibbs sampling
    for i in range(1, niter):
        sigma2_inv = 1 / sigma2_estimates[i-1]
        
        # Compute A and b matrices for the current iteration
        A = XtX * sigma2_inv + bvi
        b = X.T@y * sigma2_inv + bvi_m
        A_inv = np.linalg.inv(A)
        
        # Generate a new proposal for beta
        beta_prop = np_rng.multivariate_normal(A_inv@b, A_inv)
        
        # Generate a new proposal for sigma2
        diff = np.reshape(y - X@beta_prop, (n, 1))
        shape = a_0 + n/2
        scale = bs_0 + 1/2 * diff.T@diff
        sigma2_prop = sp.stats.invgamma.rvs(a=shape, scale=scale)
        
        # Store the new proposals in the arrays
        beta_estimates[:, i] = beta_prop
        sigma2_estimates[i] = sigma2_prop
        
    return beta_estimates, sigma2_estimates

class Gsblr:
    def __init__(self, rseed=None, burn_prop=0.5):
        # Ensure the burn-in proportion is within the valid range
        assert burn_prop > 0 and burn_prop < 1
        self.burn_prop = burn_prop
        self.samples = None
        self.burn_samples = None
        self.coef = None
        # Set up random number generator
        self.rng = np.random.default_rng(seed=rseed)
        np.random.seed(seed=rseed)
    
    def fit(self, X, y, niter=5000):
        # Perform Gibbs sampling for the given data and number of iterations
        beta, sigma2 = gibbs_bayesian_linreg(X=X, y=y, niter=niter, np_rng=self.rng)
        
        # Store the sampled values in a DataFrame
        gibbs_dict = {'beta'+str(i):beta[i-1] for i in range(1, beta.shape[0]+1)}
        gibbs_dict.update({'sigma2':sigma2})
        self.samples = pd.DataFrame(gibbs_dict)
        
        # Compute the number of burn-in iterations
        burn_num = int(np.floor(self.samples.shape[0] * self.burn_prop))
        # Store samples after removing burn-in iterations
        self.burn_samples = self.samples.iloc[burn_num:]
        # Compute and store the coefficients (mean of the remaining samples)
        self.coef = self.samples.iloc[burn_num:].mean(axis=0)
    
    def get_samples(self, var=False, remove_burn=True):
        # Return the samples with different options:
        # - var: include sigma2 in the output
        # - remove_burn: remove the burn-in iterations
        if not var and remove_burn:
            return self.burn_samples.iloc[:,:-1]
        elif not var and not remove_burn:
            return self.samples.iloc[:,:-1]
        elif var and remove_burn:
            return self.burn_samples
        else:
            return self.samples
    
    def get_coef(self, var=False):
        # Return the coefficients (mean of the samples) with or without sigma2
        if not var:
            return self.coef[:-1]
        else:
            return self.coef
    
    def predict(self, X_test):
        # Predict the target variable for the given test data
        betas = self.get_coef().to_numpy()
        betas.reshape(X_test.shape[1], 1)
        y_hat = X_test@betas
        return y_hat