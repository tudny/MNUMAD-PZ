import numpy as np
from divide_and_conquer_svd import (
    divide_and_conquer_svd,
    _householder_from_k,
    _reduce_to_bidiagonal_form,
    LambdaFunction,
    _find_all_zeros,
    find_eignepairs_of_d_z_matrix,
    _divide_and_conquer_svd_bidiagonal
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
#
# print("=" * 120)
#
# d = np.array([1, 2, 3, 4], dtype=float)
# z = np.array([11, 24, -4, 17], dtype=float)
#
# eignevalues, eigenvectors = find_eignepairs_of_d_z_matrix(d, z)
#
# print(eignevalues)
# print(eigenvectors)
#
# # A = D + z * z.T
# A = np.diag(d) + z.reshape(-1, 1) @ z.reshape(1, -1)
#
# print(A)
#
# for eigenvalue, eigenvector in zip(eignevalues, eigenvectors):
#     print('lambda_i * v_i=', eigenvalue * eigenvector)
#     print('A * v_i:      =', A @ eigenvector)
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
# # u, s, v = divide_and_conquer_svd(A)
# #
# # # Bidiagonal matrix
# # B = np.array(
# #     [
# #         [1, 2, 0, 0, 0],
# #         [0, 3, 4, 0, 0],
# #         [0, 0, 5, 6, 0],
# #         [0, 0, 0, 7, 8],
# #         [0, 0, 0, 0, 9],
# #     ],
# #     dtype=float,
# # )
# # print('B', B, sep='\n')
# # _divide_and_conquer_svd_bidiagonal(B)
#
# print('=' * 120)
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

# A = np.array(
#     [
#         [3],
#         [4]
#     ],
#     dtype=float,
# )

# A = np.array(
#     [
#         [6.1e+01, -2.5e+01],
#         [0, 7.2e+01]
#     ],
#     dtype=float,
# )

A = np.array(
    [
        [1., 2.],
        [0., 4.],
    ],
    dtype=float,
)

u, s, v = divide_and_conquer_svd(A)
print('A')
print(A)

print('u')
print(u)
print('s')
print(s)
print('v')
print(v)

print('combined')
print(u @ s @ v)
