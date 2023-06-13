import unittest

import numpy as np


def divide_and_conquer_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide and conquer SVD algorithm
    Decomposes a matrix A into U, S, V^T such that A = U S V^T
    Matrix U is orthogonal, matrix S is diagonal, and matrix V is orthogonal
    @param A: Matrix to be decomposed (m x n)
    @return: U, S, V^T such that A = U S V^T (m x m, m x n, n x n)
    """
    m, n = A.shape
    assert m >= n, "Matrix must have more rows than columns"
    u, s, vt = _divide_and_conquer_svd(A)
    assert np.allclose(A, u @ s @ vt), "Decomposition is incorrect"
    return u, s, vt


def _divide_and_conquer_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = A.shape

    # Step 1: Reduce to bidiagonal form
    u, b, v = _reduce_to_bidiagonal_form(A)

    # Step 2: Compute SVD of bidiagonal matrix

    return u, b, v


def _reduce_to_bidiagonal_form(
        A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reduces a matrix A to bidiagonal form using Householder reflections
    @param A: Matrix to be reduced to bidiagonal form (m x n)
    @return: U, B, V^T such that A = U B V^T (m x m, m x n, n x n)
    """
    m, n = A.shape
    assert m >= n, "Matrix must have more rows than columns"

    u = np.eye(m)
    v = np.eye(n)
    b = A.copy()

    for k in range(0, n - 1):
        H_k = _householder_from_k(b[:, k], k + 1)
        u = u @ H_k.T
        b = H_k @ b
        H__k = _householder_from_k(b[k, :], k + 2)
        v = H__k @ v
        b = b @ H__k.T

    return u, b, v


def _householder_from_k(x: np.ndarray, k: int) -> np.ndarray:
    """
    Constructs a Householder matrix such that multiplied by the vector `x` it will
    produce a vector with k non-zero elements (at the top of the vector).
    @param x: Vector to construct Householder matrix from
    @param k: Number of non-zero elements in the resulting vector
    @return: Householder matrix H such that H @ x = [*, ..., *, 0, ..., 0]^T where there are k '*'
    """
    upper_left_identity = np.eye(k - 1)
    lower_right_householder = _householder_from(x[k - 1:])
    n = len(x)
    return np.block(
        [
            [upper_left_identity, np.zeros((k - 1, n - k + 1))],
            [np.zeros((n - k + 1, k - 1)), lower_right_householder],
        ]
    )


def _householder_from(x: np.ndarray) -> np.ndarray:
    """
    Constructs a Householder matrix such that multiplied by the vector `v` it will
    produce a vector with at most on non-zero element (at the top of the vector).
    @param x: Vector to construct Householder matrix from
    @return: Householder matrix H such that H @ x = [*, 0, 0, ..., 0]^T
    """
    assert len(x.shape) == 1, "x must be a vector"
    n = len(x)
    v = x.copy()
    v_norm = np.linalg.norm(v)
    v[0] += np.sign(v[0]) * v_norm
    v /= np.linalg.norm(v)
    return np.eye(n) - 2 * np.outer(v, v)


# ==================================================================================================
# ========================================= Unit Tests =============================================
# ==================================================================================================


class HouseholderFromTest(unittest.TestCase):
    def _is_single_element(self, H, x):
        check = H @ x
        self.assertTrue(np.allclose(H @ x, check) or np.allclose(H @ x, -check))

    def test_householder_from_1(self):
        x = np.array([1, 2, 3, 4], dtype=float)
        H = _householder_from(x)
        self._is_single_element(H, x)

    def test_householder_from_2(self):
        x = np.array([-1, 2, 3, 4], dtype=float)
        H = _householder_from(x)
        self._is_single_element(H, x)


class HouseholderFromKTest(unittest.TestCase):
    def _is_k_elements(self, H, x, k):
        check = H @ x
        self.assertTrue(np.allclose(H @ x, check) or np.allclose(H @ x, -check))
        n = len(x)
        for i in range(k, n):
            self.assertTrue(np.isclose(check[i], 0))

    def test_householder_from_k_1(self):
        x = np.array([1, 2, 3, 4], dtype=float)
        k = 2
        H = _householder_from_k(x, k)
        self._is_k_elements(H, x, k)

    def test_householder_from_k_2(self):
        x = np.array([-1, 2, 3, 4], dtype=float)
        k = 3
        H = _householder_from_k(x, k)
        self._is_k_elements(H, x, k)


class ReduceToBidiagonalFormTest(unittest.TestCase):
    def test_reduce_to_bidiagonal_form_1(self):
        A = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ], dtype=float)
        u, b, v = _reduce_to_bidiagonal_form(A)
        self.assertTrue(np.allclose(A, u @ b @ v))
        self.assertTrue(np.allclose(u.T @ u, np.eye(4)))
        self.assertTrue(np.allclose(v.T @ v, np.eye(3)))

    def test_reduce_to_bidiagonal_form_2(self):
        A = np.array([
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8],
            [7, 8, 9, 10, 11],
            [10, 11, 12, 13, 14],
            [13, 14, 15, 16, 17],
            [16, 17, 18, 19, 20],
        ], dtype=float)
        u, b, v = _reduce_to_bidiagonal_form(A)
        self.assertTrue(np.allclose(A, u @ b @ v))
        self.assertTrue(np.allclose(u.T @ u, np.eye(6)))
        self.assertTrue(np.allclose(v.T @ v, np.eye(5)))

    def test_reduce_to_bidiagonal_form_3(self):
        A = np.fromiter(range(1, 101), dtype=float).reshape((10, 10))
        u, b, v = _reduce_to_bidiagonal_form(A)
        self.assertTrue(np.allclose(A, u @ b @ v))
        self.assertTrue(np.allclose(u.T @ u, np.eye(10)))
        self.assertTrue(np.allclose(v.T @ v, np.eye(10)))

