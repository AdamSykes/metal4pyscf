"""
Layer 1: Backend unit tests.

Verifies that each backend (MLX, NumPy) produces correct results for the
unified API.  All tests are parametrized over available backends via the
``backend`` fixture in conftest.py.

Reference values come from NumPy / SciPy.
"""

import numpy as np
import scipy.linalg
import pytest

from .conftest import to_np, get_tolerance


# =========================================================================
# Array creation and manipulation
# =========================================================================

class TestArrayCreation:

    def test_zeros(self, backend):
        a = backend.xp.zeros((4, 5))
        out = to_np(backend, a)
        assert out.shape == (4, 5)
        np.testing.assert_array_equal(out, 0)

    def test_ones(self, backend):
        a = backend.xp.ones((3, 3))
        out = to_np(backend, a)
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(out, 1)

    def test_arange(self, backend):
        a = backend.xp.arange(10)
        out = to_np(backend, a)
        np.testing.assert_array_equal(out, np.arange(10))

    def test_eye(self, backend):
        a = backend.xp.eye(4)
        out = to_np(backend, a)
        np.testing.assert_array_equal(out, np.eye(4))

    def test_array_from_list(self, backend):
        a = backend.xp.array([1.0, 2.0, 3.0])
        out = to_np(backend, a)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])

    def test_reshape(self, backend):
        a = backend.xp.arange(12)
        b = backend.xp.reshape(a, (3, 4))
        out = to_np(backend, b)
        np.testing.assert_array_equal(out, np.arange(12).reshape(3, 4))

    def test_transpose(self, backend):
        a = backend.xp.array([[1.0, 2.0], [3.0, 4.0]])
        out = to_np(backend, a.T)
        np.testing.assert_array_equal(out, [[1.0, 3.0], [2.0, 4.0]])


# =========================================================================
# Data transfer
# =========================================================================

class TestDataTransfer:

    def test_roundtrip_float64(self, backend):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        gpu = backend.to_device(original)
        back = backend.to_host(gpu)
        np.testing.assert_array_equal(back, original)

    def test_roundtrip_float32(self, backend):
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gpu = backend.to_device(original, dtype=np.float32)
        back = backend.to_host(gpu)
        np.testing.assert_array_equal(back, original)

    def test_roundtrip_2d(self, backend):
        original = np.random.randn(50, 50)
        gpu = backend.to_device(original)
        back = backend.to_host(gpu)
        # MLX may truncate to float32 — use appropriate tolerance
        atol = get_tolerance(backend, tight=0, loose=1e-7)
        np.testing.assert_allclose(back, original, atol=atol)

    def test_is_device_array(self, backend):
        gpu = backend.to_device(np.array([1.0]))
        result = backend.is_device_array(gpu)
        if backend.BACKEND_NAME == 'numpy':
            assert result is False
        else:
            assert result is True

    def test_numpy_not_device_array(self, backend):
        assert backend.is_device_array(np.array([1.0])) is False


# =========================================================================
# Linear algebra — eigh
# =========================================================================

def _make_spd(n, seed=42):
    """Create a symmetric positive-definite matrix (NumPy, float64)."""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    return A @ A.T + np.eye(n)


class TestEigh:

    def test_eigenvalues(self, backend):
        A_np = _make_spd(20)
        ref_w, _ = scipy.linalg.eigh(A_np)
        A = backend.to_device(A_np)
        w, v = backend.eigh(A)
        w_np = to_np(backend, w)
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(w_np, ref_w, atol=atol)

    def test_eigenvectors_orthonormal(self, backend):
        A_np = _make_spd(10)
        A = backend.to_device(A_np)
        _, v = backend.eigh(A)
        v_np = to_np(backend, v)
        eye = v_np.T @ v_np
        atol = get_tolerance(backend, tight=1e-10, loose=1e-5)
        np.testing.assert_allclose(eye, np.eye(10), atol=atol)

    def test_reconstruction(self, backend):
        A_np = _make_spd(10)
        A = backend.to_device(A_np)
        w, v = backend.eigh(A)
        w_np, v_np = to_np(backend, w), to_np(backend, v)
        reconstructed = v_np @ np.diag(w_np) @ v_np.T
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(reconstructed, A_np, atol=atol)

    def test_known_2x2(self, backend):
        # [[2, 1], [1, 2]] has eigenvalues 1, 3
        A = backend.to_device(np.array([[2.0, 1.0], [1.0, 2.0]]))
        w, _ = backend.eigh(A)
        w_np = to_np(backend, w)
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(sorted(w_np), [1.0, 3.0], atol=atol)


# =========================================================================
# Linear algebra — eigvalsh
# =========================================================================

class TestEigvalsh:

    def test_matches_eigh(self, backend):
        A_np = _make_spd(15)
        A = backend.to_device(A_np)
        w_full, _ = backend.eigh(A)
        w_only = backend.eigvalsh(A)
        w1 = to_np(backend, w_full)
        w2 = to_np(backend, w_only)
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(w1, w2, atol=atol)

    def test_diagonal_matrix(self, backend):
        d = np.array([5.0, 2.0, 8.0, 1.0])
        A = backend.to_device(np.diag(d))
        w = backend.eigvalsh(A)
        w_np = sorted(to_np(backend, w))
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(w_np, sorted(d), atol=atol)


# =========================================================================
# Linear algebra — solve
# =========================================================================

class TestSolve:

    def test_identity(self, backend):
        n = 5
        I = backend.to_device(np.eye(n))
        b = backend.to_device(np.arange(n, dtype=np.float64))
        x = backend.solve(I, b)
        x_np = to_np(backend, x)
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(x_np, np.arange(n), atol=atol)

    def test_residual(self, backend):
        A_np = _make_spd(20)
        b_np = np.random.randn(20)
        A = backend.to_device(A_np)
        b = backend.to_device(b_np)
        x = backend.solve(A, b)
        x_np = to_np(backend, x)
        residual = A_np @ x_np - b_np
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(residual, 0, atol=atol)

    def test_multiple_rhs(self, backend):
        A_np = _make_spd(10)
        B_np = np.random.randn(10, 3)
        A = backend.to_device(A_np)
        B = backend.to_device(B_np)
        X = backend.solve(A, B)
        X_np = to_np(backend, X)
        residual = A_np @ X_np - B_np
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(residual, 0, atol=atol)


# =========================================================================
# Linear algebra — svd
# =========================================================================

class TestSVD:

    def test_reconstruction(self, backend):
        A_np = np.random.randn(8, 6)
        A = backend.to_device(A_np)
        u, s, vt = backend.svd(A)
        u_np, s_np, vt_np = to_np(backend, u), to_np(backend, s), to_np(backend, vt)
        reconstructed = u_np[:, :6] @ np.diag(s_np) @ vt_np[:6, :]
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(reconstructed, A_np, atol=atol)

    def test_singular_values(self, backend):
        A_np = np.random.randn(10, 10)
        ref_s = np.linalg.svd(A_np, compute_uv=False)
        A = backend.to_device(A_np)
        _, s, _ = backend.svd(A)
        s_np = to_np(backend, s)
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(sorted(s_np), sorted(ref_s), atol=atol)


# =========================================================================
# Linear algebra — inv
# =========================================================================

class TestInv:

    def test_identity_inverse(self, backend):
        I = backend.to_device(np.eye(5))
        Iinv = backend.inv(I)
        Iinv_np = to_np(backend, Iinv)
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(Iinv_np, np.eye(5), atol=atol)

    def test_inverse_product(self, backend):
        A_np = _make_spd(15)
        A = backend.to_device(A_np)
        Ainv = backend.inv(A)
        Ainv_np = to_np(backend, Ainv)
        product = A_np @ Ainv_np
        atol = get_tolerance(backend, tight=1e-8, loose=1e-3)
        np.testing.assert_allclose(product, np.eye(15), atol=atol)

    def test_known_2x2(self, backend):
        A = backend.to_device(np.array([[4.0, 7.0], [2.0, 6.0]]))
        Ainv = backend.inv(A)
        Ainv_np = to_np(backend, Ainv)
        expected = np.array([[0.6, -0.7], [-0.2, 0.4]])
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(Ainv_np, expected, atol=atol)


# =========================================================================
# Linear algebra — norm, dot, einsum
# =========================================================================

class TestNorm:

    def test_frobenius(self, backend):
        A_np = np.random.randn(10, 10)
        A = backend.to_device(A_np)
        n = backend.norm(A)
        n_np = float(to_np(backend, n))
        ref = np.linalg.norm(A_np)
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(n_np, ref, atol=atol)

    def test_vector_norm(self, backend):
        v_np = np.array([3.0, 4.0])
        v = backend.to_device(v_np)
        n = backend.norm(v)
        n_np = float(to_np(backend, n))
        atol = get_tolerance(backend, tight=1e-12, loose=1e-5)
        np.testing.assert_allclose(n_np, 5.0, atol=atol)


class TestDot:

    def test_matmul_2d(self, backend):
        A_np = np.random.randn(5, 4)
        B_np = np.random.randn(4, 3)
        A, B = backend.to_device(A_np), backend.to_device(B_np)
        C = backend.dot(A, B)
        C_np = to_np(backend, C)
        ref = A_np @ B_np
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(C_np, ref, atol=atol)

    def test_matvec(self, backend):
        A_np = np.random.randn(5, 5)
        v_np = np.random.randn(5)
        A, v = backend.to_device(A_np), backend.to_device(v_np)
        result = backend.dot(A, v)
        result_np = to_np(backend, result)
        ref = A_np @ v_np
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(result_np, ref, atol=atol)


class TestEinsum:

    def test_trace(self, backend):
        A_np = np.random.randn(5, 5)
        A = backend.to_device(A_np)
        t = backend.einsum('ii->', A)
        t_np = float(to_np(backend, t))
        ref = np.trace(A_np)
        atol = get_tolerance(backend, tight=1e-10, loose=1e-4)
        np.testing.assert_allclose(t_np, ref, atol=atol)

    def test_matmul(self, backend):
        A_np = np.random.randn(4, 3).astype(np.float32)
        B_np = np.random.randn(3, 5).astype(np.float32)
        A, B = backend.to_device(A_np), backend.to_device(B_np)
        C = backend.einsum('ij,jk->ik', A, B)
        C_np = to_np(backend, C)
        ref = A_np @ B_np
        np.testing.assert_allclose(C_np, ref, atol=1e-5)

    def test_elementwise_sum(self, backend):
        A_np = np.random.randn(4, 4).astype(np.float32)
        B_np = np.random.randn(4, 4).astype(np.float32)
        A, B = backend.to_device(A_np), backend.to_device(B_np)
        s = backend.einsum('ij,ij->', A, B)
        s_np = float(to_np(backend, s))
        ref = float(np.einsum('ij,ij->', A_np, B_np))
        np.testing.assert_allclose(s_np, ref, atol=1e-4)

    def test_batch_matmul(self, backend):
        A_np = np.random.randn(2, 3, 4).astype(np.float32)
        B_np = np.random.randn(2, 4, 5).astype(np.float32)
        A, B = backend.to_device(A_np), backend.to_device(B_np)
        C = backend.einsum('bij,bjk->bik', A, B)
        C_np = to_np(backend, C)
        ref = np.einsum('bij,bjk->bik', A_np, B_np)
        np.testing.assert_allclose(C_np, ref, atol=1e-4)


# =========================================================================
# Device / memory / stream managers
# =========================================================================

class TestDeviceManager:

    def test_device_count(self, backend):
        count = backend.device_manager.device_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_properties_keys(self, backend):
        props = backend.device_manager.get_device_properties()
        assert 'name' in props
        assert 'total_memory' in props
        assert 'backend' in props
        assert props['backend'] == backend.BACKEND_NAME

    def test_synchronize(self, backend):
        backend.device_manager.synchronize()  # should not raise

    def test_set_device_context(self, backend):
        with backend.device_manager.set_device(0):
            pass  # should not raise

    def test_current_device_id(self, backend):
        did = backend.device_manager.current_device_id()
        assert isinstance(did, int)
        assert did >= 0


class TestMemoryManager:

    def test_mem_info(self, backend):
        free, total = backend.memory_manager.get_mem_info()
        assert isinstance(free, (int, float))
        assert isinstance(total, (int, float))
        assert free >= 0
        assert total >= 0

    def test_pool_used(self, backend):
        used = backend.memory_manager.get_pool_used()
        assert isinstance(used, (int, float))
        assert used >= 0

    def test_free_all_blocks(self, backend):
        backend.memory_manager.free_all_blocks()  # should not raise

    def test_alloc_pinned(self, backend):
        buf = backend.memory_manager.alloc_pinned(1024)
        assert len(buf) >= 1024


class TestStreamManager:

    def test_get_current_stream(self, backend):
        s = backend.stream_manager.get_current_stream()
        assert hasattr(s, 'ptr')
        assert hasattr(s, 'synchronize')

    def test_synchronize_stream(self, backend):
        backend.stream_manager.synchronize_stream()  # should not raise

    def test_create_stream(self, backend):
        s = backend.stream_manager.create_stream()
        s.synchronize()  # should not raise
