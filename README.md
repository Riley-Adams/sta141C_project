# Analyzing MCMC vs Frequentist Methods in Regression

Abstract

Within this report, we implement a Gibbs sampler for conducting MCMC calculations to compute the
point estimates and distributions of linear regression under a Bayesian framework with non-informative
priors. We perform this analysis upon two simulated datasets: the first with n = 300 and p = 6, including
two nonsignificant predictors, and the second featuring a higher dimensional setup with n = 1050 and
p = 1000. Training and test sets of equal size are generated from the same model. We then the
coefficient estimates produced by our Gibbs sampler to those from common linear regression functions
within sklearn.linear_model including LinearRegression, RidgeCV and LassoCV by computing the
mean squared error (MSE) of prediction. We also compare runtimes for the higher dimensional model
across various methods. Our findings reveal that the Gibbs sampler is significantly more computationally
expensive than the sklearn.linear_model methods despite yielding similar prediction performance.
Ultimately, we conclude that LASSO is the preferred method for our example cases when only point
estimates are needed; however, our Gibbs sampling methodology enables the creation of credible intervals
for the predictors.

## Authors
Jordan Bowman, Samuel Van Gorden, Riley Adams
