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
    i_m = np.eye(m)
    i_n = np.eye(n)

    # Step 1: Reduce to bidiagonal form

    # Step 2: Compute SVD of bidiagonal matrix

    return i_m, A, i_n


def _hessenberg_from(v: np.ndarray) -> np.ndarray:
    pass


def _reduce_to_bidiagonal_form(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pass
