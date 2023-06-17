import unittest

import numpy as np


def sliding_window(collection, window_size):
    if len(collection) < window_size:
        return
    for i in range(len(collection) - window_size + 1):
        yield collection[i: i + window_size]


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
    """
    Divide and conquer SVD algorithm
    @param A: Matrix to be decomposed (m x n)
    @return: U, S, V^T such that A = U S V^T (m x m, m x n, n x n)
    """
    m, n = A.shape

    # Step 1: Reduce to bidiagonal form
    u, b, v = _reduce_to_bidiagonal_form(A)
    b = b[:n, :n]

    # Step 2: Compute SVD of bidiagonal matrix
    U, S, V = _divide_and_conquer_svd_bidiagonal(b)

    return u @ U, S, V @ v


def _divide_and_conquer_svd_bidiagonal(
        A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the SVD of a bidiagonal matrix
    @param A: Bidiagonal matrix (m x n)
    @return: U, S, V^T such that A = U S V^T (m x m, m x n, n x n)
    """
    n, m = A.shape
    return np.eye(m), A, np.eye(n)


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
# ======================================= Finding zeros ============================================
# ==================================================================================================


class LambdaFunction:
    def __init__(self, z: np.ndarray, d: np.ndarray):
        """
        Represents a function of the form $f(x) = 1 + \sum_i^N z_i^2 / (d_i - x)$
        @param z: Parameter vector (N)
        @param d: Parameter vector (N)
        """
        self.z = z
        self.d = d
        # Assert z and d and one-dimensional vectors
        assert len(z.shape) == 1, "z must be a vector"
        assert len(d.shape) == 1, "d must be a vector"
        # Assert z and d have the same length
        assert len(z) == len(d), "z and d must have the same length"

    def __call__(self, *args, **kwargs):
        """
        Evaluates the function at x
        @param x: Value to evaluate function at
        @return: f(x)
        """
        x = args[0]
        return 1 + np.sum(self.z ** 2 / (self.d - x))


def _find_zero_in(f: LambdaFunction, a: float, b: float, tol: float = 1e-6) -> float:
    """
    Finds a zero of the function f in the interval [a, b]
    @param f: Function to find zero of
    @param a: Left endpoint of interval
    @param b: Right endpoint of interval
    @param tol: Tolerance for finding zero
    @return: Zero of f in [a, b]
    """
    assert a < b, "a must be less than b"

    f_a = -1  # We only care about the sign of f(a)
    c = (a + b) / 2
    while abs(f(c)) > tol:
        if f_a * f(c) < 0:
            b = c
        else:
            a = c
        c = (a + b) / 2
    return c


def _find_zero_after(f: LambdaFunction, a: float, tol: float = 1e-6) -> float:
    """
    Finds a zero of the function f after the point a
    @param f: Function to find zero of
    @param a: Point after which to find zero
    @param tol: Tolerance for finding zero
    @return: Zero of f after a
    """
    value_at_a = -1
    maybe_b = a + 1
    step = 1
    while f(maybe_b) * value_at_a > 0:
        maybe_b += step
        step *= 2
    return _find_zero_in(f, a, maybe_b, tol)


def _find_all_zeros(f: LambdaFunction, tol: float = 1e-6) -> np.ndarray:
    """
    Finds all zeros of the function f. There are N zeros.
    N-1 are between d_i and d_{i+1} and the last one is after d_N.
    @param f: Function to find zeros of
    @param tol: Tolerance for finding zeros
    @return: Vector of all zeros of f
    """
    zeros = []
    for d_i, d_i1 in sliding_window(f.d, 2):
        zero_i = _find_zero_in(f, d_i, d_i1, tol)
        zeros.append(zero_i)
    zeros.append(_find_zero_after(f, f.d[-1], tol))
    return np.array(zeros)


def _find_eigenvalue_of_d_z_matrix(
        d: np.ndarray, z: np.ndarray, tol: float = 1e-6
) -> np.ndarray:
    """
    Finds the eigenvalues of the matrix D - Z^T Z
    @param d: Parameter vector (N)
    @param z: Parameter vector (N)
    @param tol: Tolerance for finding zeros
    @return: Eigenvalues of D - Z^T Z
    """
    f = LambdaFunction(z, d)
    zeros = _find_all_zeros(f, tol)
    return zeros


def _find_kth_eigenvector_of_d_z_matrix(
        d: np.ndarray, z: np.ndarray, kth_eigenvalue: float, tol: float = 1e-6
) -> np.ndarray:
    """
    Finds the kth eigenvector of the matrix D + z^T z
    @param d: parameter vector (N)
    @param z: parameter vector (N)
    @param kth_eigenvalue: kth eigenvalue of D + z^T z
    @param tol: Tolerance for finding eigenvectors
    @return: kth eigenvector of D + z^T z
    """
    m_rev = np.array([1 / (d_i - kth_eigenvalue) for d_i in d])
    return z * m_rev


def _find_eigenvectors_of_d_z_matrix(
        d: np.ndarray, z: np.ndarray, eigenvalues: np.ndarray, tol: float = 1e-6
) -> list[np.ndarray]:
    """
    Finds the eigenvectors of the matrix D + z^T z
    @param d: Parameter vector (N)
    @param z: Parameter vector (N)
    @param eigenvalues: Eigenvalues of D + z^T z
    @param tol: Tolerance for finding eigenvectors
    @return: Eigenvectors of D + z^T z
    """
    eigenvectors = []
    for eigenvalue in eigenvalues:
        eigenvector = _find_kth_eigenvector_of_d_z_matrix(d, z, eigenvalue, tol)
        eigenvectors.append(eigenvector)
    return eigenvectors


def find_eignepairs_of_d_z_matrix(
        d: np.ndarray, z: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Finds the eigenvalues and eigenvectors of the matrix D + z^T z
    @param d: Parameter vector (N)
    @param z: Parameter vector (N)
    @param tol: Tolerance for finding eigenvalues and eigenvectors
    @return: Eigenvalues and eigenvectors of D + z^T z
    """
    eigenvalues = _find_eigenvalue_of_d_z_matrix(d, z, tol)
    eigenvectors = _find_eigenvectors_of_d_z_matrix(d, z, eigenvalues, tol)
    return eigenvalues, eigenvectors


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
        A = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            dtype=float,
        )
        u, b, v = _reduce_to_bidiagonal_form(A)
        self.assertTrue(np.allclose(A, u @ b @ v))
        self.assertTrue(np.allclose(u.T @ u, np.eye(4)))
        self.assertTrue(np.allclose(v.T @ v, np.eye(3)))

    def test_reduce_to_bidiagonal_form_2(self):
        A = np.array(
            [
                [1, 2, 3, 4, 5],
                [4, 5, 6, 7, 8],
                [7, 8, 9, 10, 11],
                [10, 11, 12, 13, 14],
                [13, 14, 15, 16, 17],
                [16, 17, 18, 19, 20],
            ],
            dtype=float,
        )
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
