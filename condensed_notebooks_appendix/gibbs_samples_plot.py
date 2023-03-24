'''
Please note that while the original code was in a Jupyter Notebook, we have condensed it into this .py file for formatting purposes in this appendix.
'''

# import modules
import numpy as np
import pandas as pd

# read in data
gibbs_samples = pd.read_csv('../gibbs_samples.csv', index_col=0)
gibbs_samples_all = pd.read_csv('../gibbs_samples_all.csv', index_col=0)

# create plot with burn-in
ax = gibbs_samples_all.plot(title="Gibbs Samples for Small Model (Including Burn-In)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameter Value")

# create plot without burn-in
ax = gibbs_samples.plot(title="Gibbs Samples for Small Model (Excluding Burn-In)")
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameter Value")