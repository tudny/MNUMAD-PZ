import numpy as np
from divide_and_conquer_svd import (
    divide_and_conquer_svd,
    _householder_from_k,
    _reduce_to_bidiagonal_form,
    LambdaFunction,
    _find_all_zeros,
    find_eignepairs_of_d_z_matrix
)
#
# A = np.array(
#     [
#         [31, 2, 3, 4, 5],
#         [21, 3, 4, 5, 6],
#         [3, 44, 5, 6, 7],
#         [41, 5, 6, 7, 8],
#         [5, 6, 77, 8, 9],
#         [6, 7, 8, 9, 0],
#     ],
#     dtype=float,
# )
#
# u, s, v = divide_and_conquer_svd(A)
# print(u)
# print(s)
# print(v)

# print(np.allclose(A, u @ s @ v))

print("=" * 120)

d = np.array([1, 2, 3, 4], dtype=float)
z = np.array([11, 24, -4, 17], dtype=float)

eignevalues, eigenvectors = find_eignepairs_of_d_z_matrix(d, z)

print(eignevalues)
print(eigenvectors)

# A = D + z * z.T
A = np.diag(d) + z.reshape(-1, 1) @ z.reshape(1, -1)

print(A)

for eigenvalue, eigenvector in zip(eignevalues, eigenvectors):
    print('lambda_i * v_i=', eigenvalue * eigenvector)
    print('A * v_i:      =', A @ eigenvector)
