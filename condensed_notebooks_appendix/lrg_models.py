'''
Please note that while the original code was in a Jupyter Notebook, we have condensed it into this .py file for formatting purposes in this appendix.
'''

# import modules
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
import gsblr

# import data
X_train = np.loadtxt('data/large_model/X_train.txt', delimiter=',')
y_train = np.loadtxt('data/large_model/y_train.txt', delimiter=',')
X_test = np.loadtxt('data/large_model/X_test.txt', delimiter=',')
y_test = np.loadtxt('data/large_model/y_test.txt', delimiter=',')

## LINEAR REGRESSION -------------------------------------
# begin timer
start = time.time()
# fit linear regression model
linreg_model = LinearRegression().fit(X_train, y_train)
# end timer
linreg_time = time.time() - start
# linear regression coefficients
linreg_coef = linreg_model.coef_
# predict with linreg model
linreg_pred = linreg_model.predict(X_test)
# MSE for linreg model
linreg_mse = mean_squared_error(y_test, linreg_pred)
## --------------------------------------------------------

## RIDGE --------------------------------------------------
# begin timer
start = time.time()
# fit ridge regression model with cross validation
ridge_model = RidgeCV().fit(X_train, y_train)
# end timer
ridge_time = time.time() - start
# ridge "alpha" parameter (lambda)
ridge_model.alpha_
# ridge coefficients
ridge_coef = ridge_model.coef_
# predict with ridge model
ridge_pred = ridge_model.predict(X_test)
# MSE for ridge model
ridge_mse = mean_squared_error(y_test, ridge_pred)
## --------------------------------------------------------

## LASSO --------------------------------------------------
# begin timer
start = time.time()
# fit lasso regression model with cross validation
lasso_model = LassoCV(random_state=141).fit(X_train, y_train)
# end timer
lasso_time = time.time() - start
# lasso "alpha" parameter (lambda)
lasso_model.alpha_ 
# lasso coefficients
lasso_coef = lasso_model.coef_
# predict with lasso model
lasso_pred = lasso_model.predict(X_test)
# MSE for lasso model
lasso_mse = mean_squared_error(y_test, lasso_pred)
## --------------------------------------------------------

## GIBBS --------------------------------------------------
# 1) Iterations: 5000, Burn proportion: 0.5
# begin timer
start = time.time()
# initialize gibbs sampler
gibbs = gsblr.Gsblr(rseed=141)
# fit gibbs sampler
gibbs.fit(X_train, y_train)
# end timer
gibbs_time = time.time() - start
# gibbs coefficients
gibbs_coef = gibbs.get_coef().values
# predict with gibbs
gibbs_pred = gibbs.predict(X_test)
# MSE for gibbs
gibbs_mse = mean_squared_error(y_test, gibbs_pred)

# 2) Iterations: 100, Burn proportion: 0.3
# begin timer
start = time.time()
#initialize gibbs sampler
gibbs_2 = gsblr.Gsblr(rseed=141, burn_prop=0.3)
# fit gibbs_2 sampler
gibbs_2.fit(X_train, y_train, niter= 100)
# end time
gibbs_time_2 = time.time() - start
# predict with gibbs_2
gibbs_2_pred = gibbs_2.predict(X_test)
# MSE for gibbs_2
gibbs_2_mse = mean_squared_error(y_test, gibbs_2_pred)
# results for gibbs_2
pd.Series({'Runtime': gibbs_time_2,
           'MSE': gibbs_2_mse})
## --------------------------------------------------------

## DATA SUMMARY -------------------------------------------
# create datafram to summarize results
lrg_results = pd.DataFrame({
    'Method': ['Linreg', 'Ridge', 'LASSO', 'Gibbs'],
    'MSE': [linreg_mse, ridge_mse, lasso_mse, gibbs_mse],
    'runtime': [linreg_time, ridge_time, lasso_time, gibbs_time]
})
# view results
lrg_results
# save results
lrg_results.to_csv('lrg_model_results.csv')