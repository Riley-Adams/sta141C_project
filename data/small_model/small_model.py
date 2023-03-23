'''
Simulates a model:
y = B.T @ X + e
y = 7X_i1 + 1X_i2 + 0X_i3 + 4X_i4 + 3X_i5 + 0X_i6 + e_i

e ~ N(0,9)
b1 = 7, b2 = 1, b3 = 0, b4 = 4, b5 = 3, b6 = 0

Each X_i is uniformly distributed with some set of parameters.
'''

# Import modules.
import numpy as np

# Create instance of generator.
rng = np.random.default_rng(141)

# Define n, p.
n = 300
p = 6

# Define betas.
betas = np.array([7, 1, 0, 4, 3, 0])

# sample each x vector.
x1 = rng.uniform(0,30,n)
x2 = rng.uniform(20,50,n)
x3 = rng.uniform(15,18,n)
x4 = rng.uniform(0,50,n)
x5 = rng.uniform(0,10,n)
x6 = rng.uniform(25,27,n)

# Concatenate x vectors into X matrix.
X = np.column_stack((x1, x2, x3, x4, x5, x6))

# Sample epsilon.
e = rng.normal(0,9,n)

# Generate y.
y = X @ betas[:, None] + e[:, None]

# Save data.
np.savetxt('data/small_model/betas.txt', betas, delimiter=',')
np.savetxt('data/small_model/X.txt', X, delimiter=',')
np.savetxt('data/small_model/y.txt', y, delimiter=',')