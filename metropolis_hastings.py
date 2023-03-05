# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

# Implements a Metropolis-Hastings MCMC algorithm that uses random
# walk to generate new draws (only need prior, no likelihood)
def metropolis_hastings_rand_walk(prior, niter):
    x = []
    x.append(1)
   
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

print(metropolis_hastings_rand_walk(np.random.normal, 1000))