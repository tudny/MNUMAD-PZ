import numpy as np
from divide_and_conquer_svd import divide_and_conquer_svd

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

U, S, Vt = divide_and_conquer_svd(A)

# print the result
print("U = ", U)
print("S = ", S)
print("Vt = ", Vt)

# check the result
print("A = ", U @ S @ Vt)
