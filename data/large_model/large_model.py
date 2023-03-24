# import modules
import numpy as np

# create instance of generator
rng = np.random.default_rng(141)

# define n, p
n = 1050
p = 1000

## generate betas (vector of length p)
# covariance matrix
betas_cov = np.diag(rng.uniform(0,20, p))
# means vector
betas_means = rng.uniform(-10,10,p)
# draw betas
betas = rng.multivariate_normal(betas_means, betas_cov)

# Function to sample X, y
def sampler(betas=betas, n=n, p=p):
    ## generate X (n rows, p columns)
    X = rng.uniform(0,100,(n,p))

    ## generate G
    # a matrix with 1's and 0's on diag
    # 0's make corresponding betas insignificant
    G = np.diag(rng.binomial(n=1, p=0.5, size=p))

    ## generate error terms
    # covariance matrix
    err_cov = 25 * np.identity(n)
    # means vector
    err_mean = np.zeros(n)
    # error terms
    err = rng.multivariate_normal(err_mean, err_cov)

    ## generate Y
    y = X @ (G @ betas) + err

    # output X, y, G
    return X, y, G

# Generate training, testing data.
X_train, y_train, G_train = sampler()
X_test, y_test, G_test = sampler()

## store simulated data
# beta
np.savetxt('data/large_model/betas.txt', betas, delimiter=',')
np.savetxt('data/large_model/betas_cov.txt', betas_cov, delimiter=',')
np.savetxt('data/large_model/betas_means.txt', betas_means, delimiter=',')
# train
np.savetxt('data/large_model/G_train.txt', G_train, delimiter=',')
np.savetxt('data/large_model/X_train.txt', X_train, delimiter=',')
np.savetxt('data/large_model/y_train.txt', y_train, delimiter=',')
# test
np.savetxt('data/large_model/G_test.txt', G_test, delimiter=',')
np.savetxt('data/large_model/X_test.txt', X_test, delimiter=',')
np.savetxt('data/large_model/y_test.txt', y_test, delimiter=',')
