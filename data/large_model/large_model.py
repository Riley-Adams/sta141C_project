# import modules
import numpy as np

# create instance of generator
rng = np.random.default_rng(141)

# define n, p
n = 1050
p = 1000

## generate G
# a matrix with 1's and 0's on diag
# 0's make corresponding betas insignificant
G = np.diag(rng.binomial(n=1, p=0.5, size=p))

## generate betas (vector of length p)

# covariance matrix
betas_cov = np.diag(rng.uniform(0,20, p))

# means vector
betas_means = rng.uniform(-10,10,p)

# draw betas
betas = rng.multivariate_normal(betas_means, betas_cov)

## generate X (n rows, p columns)
X = rng.uniform(0,100,(n,p))

## generate error terms

# covariance matrix
err_cov = 25 * np.identity(n)

# means vector
err_mean = np.zeros(n)

# error terms
err = rng.multivariate_normal(err_mean, err_cov)
## -------------------------------------------------------

## generate Y
Y = X @ (G @ betas) + err

# store simulated data
np.savetxt('G.txt', G, delimiter=',')
np.savetxt('betas.txt', betas, delimiter=',')
np.savetxt('X.txt', X, delimiter=',')
np.savetxt('err.txt', err, delimiter=',')
np.savetxt('Y.txt', Y, delimiter=',')
np.savetxt('betas_cov.txt', betas_cov, delimiter=',')
np.savetxt('betas_means.txt', betas_means, delimiter=',')