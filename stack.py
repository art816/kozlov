__author__ = 'art'

import numpy as np

primary = np.array([[40., 1160., 1.],
                    [40., 40., 1.],
                    [260., 40., 1.],
                    [260., 1160., 1.],
                    [260., 40., 1.],
                    [260., 1160., 1.]])

secondary = np.array([[610., 560., 1.],
                      [610., -560., 1.],
                      [390., -560., 1.],
                      [390., 560., 1.],
                      [390., -560., 1.],
                      [390., 564., 1.]])

A, res, rank, s = np.linalg.lstsq(primary, secondary)
print(np.dot(primary, A))
print(res, rank, s)
# Pad the data with ones, so that our transformation can do translations too
n = primary.shape[0]
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]
X = pad(primary)
print(X)
Y = pad(secondary)

# Solve the least squares problem X * A = Y
# to find our transformation matrix A
A, res, rank, s = np.linalg.lstsq(X, Y)
print(A, res, rank, s)

transform = lambda x: unpad(np.dot(pad(x), A))


print("Target:")
print(secondary)
print("Result:")
print(np.dot(pad(primary), A))
print(transform(primary))
print("Max error:", np.abs(secondary - transform(primary)).max())