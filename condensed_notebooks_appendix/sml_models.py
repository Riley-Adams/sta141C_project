'''
Please note that while the original code was in a Jupyter Notebook, we have condensed it into this .py file for formatting purposes in this appendix.
'''

# import modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error
import gsblr

# import data
X_train = np.loadtxt('data/small_model/X_train.txt', delimiter=',')
y_train = np.loadtxt('data/small_model/y_train.txt', delimiter=',')
X_test = np.loadtxt('data/small_model/X_test.txt', delimiter=',')
y_test = np.loadtxt('data/small_model/y_test.txt', delimiter=',')
betas = np.loadtxt('data/small_model/betas.txt', delimiter=',')

## LINEAR REGRESSION (OLS) ----------------------------------------
# fit linear regression model
linreg_model = LinearRegression().fit(X_train, y_train)
# linear regression coefficients
linreg_coef = linreg_model.coef_
# predict with linreg model
linreg_pred = linreg_model.predict(X_test)
# MSE for linreg model
linreg_mse = mean_squared_error(y_test, linreg_pred)
## ----------------------------------------------------------------

## RIDGE ----------------------------------------------------------
# fit ridge regression model with cross validation
ridge_model = RidgeCV().fit(X_train, y_train)
# ridge "alpha" parameter (lambda)
ridge_model.alpha_ 
# ridge coefficients
ridge_coef = ridge_model.coef_
# predict with ridge model
ridge_pred = ridge_model.predict(X_test)
# MSE for ridge model
ridge_mse = mean_squared_error(y_test, ridge_pred)
## ----------------------------------------------------------------

## LASSO ----------------------------------------------------------
# fit lasso regression model with cross validation
lasso_model = LassoCV(random_state=141).fit(X_train, y_train)
# lasso "alpha" parameter (lambda)
lasso_model.alpha_ 
# lasso coefficients
lasso_coef = lasso_model.coef_
# predict with lasso model
lasso_pred = lasso_model.predict(X_test)
# MSE for lasso model
lasso_mse = mean_squared_error(y_test, lasso_pred)
## ----------------------------------------------------------------

## GIBBS ----------------------------------------------------------
# initialize gibbs sampler
gibbs = gsblr.Gsblr(rseed=141)
# fit gibbs sampler
gibbs.fit(X_train, y_train)
# gibbs coefficients
gibbs_coef = gibbs.get_coef().values
# predict with gibbs
gibbs_pred = gibbs.predict(X_test)
# MSE for gibbs
gibbs_mse = mean_squared_error(y_test, gibbs_pred)
## ----------------------------------------------------------------

## DATA SUMMARY ---------------------------------------------------
# create dataframe of all regression coefficients
coef_df = pd.DataFrame({'true_coef': betas,
                        'linreg_coef': linreg_coef,
                        'ridge_coef': ridge_coef,
                        'lasso_coef': lasso_coef,
                        'gibbs_coef': gibbs_coef}
                        )

coef_df
# create dataframe of MSE for each model
mse_df = pd.Series(data=[linreg_mse, ridge_mse, lasso_mse, gibbs_mse],
                   index= ['linreg_mse', 'ridge_mse', 'lasso_mse', 'gibbs_mse'])

mse_df

# dataframe of gibbs samples (excluding burned samples)
gibbs_samples = gibbs.get_samples()

# dataframe of gibbs samples (including burned samples)
gibbs_samples_all = gibbs.get_samples(remove_burn=False)

# save to csv
gibbs_samples.to_csv('gibbs_samples.csv')
gibbs_samples_all.to_csv('gibbs_samples_all.csv')
coef_df.to_csv('coefs_data.csv')
## ----------------------------------------------------------------