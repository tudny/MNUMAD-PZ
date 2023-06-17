import numpy as np
from divide_and_conquer_svd import divide_and_conquer_svd, _householder_from_k, _reduce_to_bidiagonal_form, \
    LambdaFunction, _find_all_zeros

A = np.array([
    [31, 2, 3, 4, 5],
    [21, 3, 4, 5, 6],
    [3, 44, 5, 6, 7],
    [41, 5, 6, 7, 8],
    [5, 6, 77, 8, 9],
    [6, 7, 8, 9, 0],
], dtype=float)
#
# u, s, v = divide_and_conquer_svd(A)
# print(u)
# print(s)
# print(v)

# print(np.allclose(A, u @ s @ v))

print('=' * 120)

f = LambdaFunction(
    np.array([10, 17]),
    np.array([10, 100])
)

all_zeros = _find_all_zeros(f)

print(all_zeros)
