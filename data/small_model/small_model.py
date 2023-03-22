'''
Simulates a model:
y = B.T @ X + e
y = 7X_i1 + 1X_i2 + 0X_i3 + 4X_i4 + 3X_i5 + 0X_i6 + e_i

e ~ N(0,9)
b1 = 7, b2 = 1, b3 = 0, b4 = 4, b5 = 3, b6 = 0
'''

# import modules
import numpy as np

# create instance of generator
rng = np.random.default_rng(141)

# define n, p
n = 30
p = 6

# define betas
betas = np.array([7, 1, 0, 4, 3, 0])

# sample each x vector
x1 = rng.uniform(0,30,n)
x2 = rng.uniform(20,50,n)
x3 = np.zeros(n)
x4 = rng.uniform(0,50,n)
x5 = rng.uniform(0,10,n)
x6 = np.zeros(n)

# concatenate x vectors into X matrix
X = np.column_stack((x1, x2, x3, x4, x5, x6))

# sample epsilon
e = rng.normal(0,9,n)

# generate y
y = X @ betas[:, None] + e[:, None]

# save data
np.savetxt('data/small_model/betas.txt', betas, delimiter=',')
np.savetxt('data/small_model/X.txt', X, delimiter=',')
np.savetxt('data/small_model/y.txt', y, delimiter=',')