import numpy as np
from divide_and_conquer_svd import divide_and_conquer_svd, _householder_from_k, _reduce_to_bidiagonal_form

A = np.array([
    [1,  2, 3,  4, 5],
    [21, 3, 4,  5, 6],
    [3, 44, 5,  6, 7],
    [41, 5, 6,  7, 8],
    [5,  6, 77, 8, 9],
], dtype=float)

u, s, v = divide_and_conquer_svd(A)
print(u)
print(s)
print(v)

print(np.allclose(A, u @ s @ v))
