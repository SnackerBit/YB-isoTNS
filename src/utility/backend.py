import os
from enum import Enum
import numpy as np
import scipy
import scipy.linalg

class Backend(Enum):
    NUMPY = 1
    JAX_CPU = 2
    JAX_GPU = 3

backend = Backend.NUMPY
array_type = np.ndarray
sparse_array_type = scipy.sparse.csr_array
nan = np.nan
inf = np.inf
dtype_complex = np.complex128
newaxis = np.newaxis

def set_backend(backend_enum):
    if backend_enum == Backend.NUMPY:
        backend = Backend.NUMPY
        array_type = np.ndarray
        sparse_array_type = scipy.sparse.csr_array
        nan = np.nan
        inf = np.inf
        dtype_complex = np.complex128
        newaxis = np.newaxis
    elif backend_enum == BACKEND.JAX_CPU:
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        backend = Backend.JAX_CPU
        array_type = jnp.array
        sparse_array_type = jax.experimental.sparse.BCOO
        nan = jnp.nan
        inf = jnp.inf
        dtype_complex = jnp.complex128
        newaxis = jnp.newaxis
    elif backend_enum == BACKEND.JAX_GPU:
        os.environ["JAX_PLATFORMS"] = "gpu"
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        backend = Backend.JAX_GPU
        array_type = jnp.array
        sparse_array_type = jax.experimental.sparse.BCOO
        nan = jnp.nan
        inf = jnp.inf
        dtype_complex = jnp.complex128
        newaxis = jnp.newaxis

log_matrix_ops = False

logged_matrix_ops_qr = {}
logged_matrix_ops_svd = {}
logged_matrix_ops_eigh = {}
logged_matrix_ops_tensordot = {}
logged_matrix_ops_kron = {}
logged_matrix_ops_trace = {}
logged_matrix_ops_dot = {}
logged_matrix_ops_reshape = {}
logged_matrix_ops_transpose = {}

def reset_log_matrix_ops():
    global logged_matrix_ops_qr
    global logged_matrix_ops_svd
    global logged_matrix_ops_eigh
    global logged_matrix_ops_tensordot
    global logged_matrix_ops_kron
    global logged_matrix_ops_trace
    global logged_matrix_ops_dot
    global logged_matrix_ops_reshape
    global logged_matrix_ops_transpose
    logged_matrix_ops_qr = {}
    logged_matrix_ops_svd = {}
    logged_matrix_ops_eigh = {}
    logged_matrix_ops_tensordot = {}
    logged_matrix_ops_kron = {}
    logged_matrix_ops_trace = {}
    logged_matrix_ops_dot = {}
    logged_matrix_ops_reshape = {}
    logged_matrix_ops_transpose = {}


def array(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.array(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.array(*args, **kwargs)

def safe_svd(A, full_matrices=True):
    """
    Computes the Singular Value Decomposition A = U@S@V. If the numpy svd does not converge,
    scipy's SVD with the less efficient but more involved general rectangular approach is used,
    which is more likely to converge.

    Parameters
    ----------
    A : np.ndarray of shape (n, m)
        The matrix that should be decomposed using SVD.
    full_matrices : bool, optional
        determines the shape of the output matrices. See official
        numpy documentation for more details.
    
    Returns
    -------
    U : np.ndarray of shape (n, chi)
        isometric matrix. A = U@np.diag(S)@V.
    S : np.ndarray of shape (chi, )
        vector containing the real singular values >= 0. A = U@np.diag(S)@V.
    V : np.ndarray of shape (chi, m)
        V.T is an isometric matrix. A = U@np.diag(S)@V.
    """
    if backend == Backend.NUMPY:
        try:
            return np.linalg.svd(A, full_matrices=full_matrices)
        except np.linalg.LinAlgError:
            if np.isnan(A).any() or np.isinf(A).any():
                print("[WARNING]: Trying to perform SVD on a matrix with nan or inf entries!")
            U, S, V = scipy.linalg.svd(A, full_matrices=full_matrices, lapack_driver='gesvd')
            if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
                print("[WARNING] scipy SVD did not converge!")
                m, n = A.shape
                k = min(m, n)
                return np.zeros(m, k), np.zeros(k), np.zeros(k, n)
            return U, S, V
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        U, S, V = jnp.linalg.svd(A, full_matrices=full_matrices)
        if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
            if jnp.isnan(A).any() or jnp.isinf(A).any():
                print("[WARNING]: Trying to perform SVD on a matrix with nan or inf entries!")
            U, S, V = jnp.scipy.linalg.svd(A, full_matrices=full_matrices, lapack_driver='gesvd')
            if np.isnan(U).any() or np.isinf(U).any() or np.isnan(S).any() or np.isinf(S).any() or np.isnan(V).any() or np.isinf(V).any():
                print("[WARNING] scipy SVD did not converge!")
                m, n = A.shape
                k = min(m, n)
                return np.zeros(m, k), np.zeros(k), np.zeros(k, n)
            return U, S, V


def sum(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sum(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.sum(*args, **kwargs)

def argsort(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.argsort(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.argsort(*args, **kwargs)

def norm(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.linalg.norm(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.linalg.norm(*args, **kwargs)

def random(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.random.random(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"random\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"random\" is not implemented for backend \"JAX_CPU\".")

def sqrt(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sqrt(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.sqrt(*args, **kwargs)
        
def diag(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.diag(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.diag(*args, **kwargs)

def abs(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.abs(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.abs(*args, **kwargs)
        
def conj(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.conj(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.conj(*args, **kwargs)
        
def eye(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.eye(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.eye(*args, **kwargs)

def isclose(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isclose(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.isclose(*args, **kwargs)

def all(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.all(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.all(*args, **kwargs)

def allclose(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.allclose(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.allclose(*args, **kwargs)

def real_if_close(a, tol=100):
    if backend == Backend.NUMPY:
        return np.real_if_close(a=a, tol=tol)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        if jnp.allclose(jnp.imag(a), 0.0, rtol=tol*1.e-15):
            return jnp.real(a)
        else:
            return a
        
def floor(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.floor(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.floor(*args, **kwargs)

def ascontiguousarray(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.ascontiguousarray(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.asarray(*args, **kwargs)

def round(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.round(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.round(*args, **kwargs)

def random_unitary(N):
    """
    Returns a random (N, N) unitary drawn from the Haar measure
    
    Parameters
    ----------
    N: int
        dimension of the unitary
    
    Returns
    -------
    U : backend.array_type of shape (N, N)
        random unitary
    """
    if backend == Backend.NUMPY:
        from scipy.stats import unitary_group
        return unitary_group.rvs(N)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"random_unitary\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"random_unitary\" is not implemented for backend \"JAX_CPU\".")

def expm(*args, **kwargs):
    if backend == Backend.NUMPY:
        return scipy.linalg.expm(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jax.scipy.linalg.expm(*args, **kwargs)

def flipud(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.flipud(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.flipud(*args, **kwargs)

def sparse_kron(*args, **kwargs):
    if backend == Backend.NUMPY:
        return scipy.sparse.kron(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"sparse_kron\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"sparse_kron\" is not implemented for backend \"JAX_CPU\".")

def min(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.min(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.min(*args, **kwargs)

def rand(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.random.rand(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")

def isnan(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isnan(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.isnan(*args, **kwargs)

def isinf(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.isinf(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.isinf(*args, **kwargs)

def sign(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sign(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.sign(*args, **kwargs)

def real(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.real(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.real(*args, **kwargs)

def where(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.where(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.where(*args, **kwargs)

def imag(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.imag(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.imag(*args, **kwargs)

def zeros(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.zeros(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.zerps(*args, **kwargs)

def ones(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.ones(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.ones(*args, **kwargs)

def log(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.log(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.log(*args, **kwargs)

def arctan(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.arctan(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.arctan(*args, **kwargs)

def arctan2(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.arctan2(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.arctan2(*args, **kwargs)

def sin(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.sin(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.sin(*args, **kwargs)

def cos(*args, **kwargs):
    if backend == Backend.NUMPY:
        return np.cos(*args, **kwargs)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.cos(*args, **kwargs)
        
# ==========================================================================
# ========================== Tensor contractions ===========================
# ==========================================================================

def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_trace:
            logged_matrix_ops_trace[a.shape] += 1
        else: 
            logged_matrix_ops_trace[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.trace(a, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out)

def kron(a, b):
    if log_matrix_ops:
        key = np.prod(a.shape)*np.prod(b.shape)
        if key in logged_matrix_ops_kron:
            logged_matrix_ops_kron[key] += 1
        else: 
            logged_matrix_ops_kron[key] = 1
    if backend == Backend.NUMPY:
        return np.kron(a, b)
    elif backend == BACKEND.JAX_CPU or backend == BACKEND.JAX_GPU:
        return jnp.kron(a, b)

def tensordot(a, b, axes=2):
    if log_matrix_ops:
        key = np.prod(a.shape)
        for axis, value in enumerate(b.shape):
            if axis == axes or ((isinstance(axes, tuple) or isinstance(axes, list)) and axis in axes[0]):
                continue
            else:
                key *= value
        if key in logged_matrix_ops_tensordot:
            logged_matrix_ops_tensordot[key] += 1
        else: 
            logged_matrix_ops_tensordot[key] = 1
    if backend == Backend.NUMPY:
        return np.tensordot(a, b, axes=axes)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"tensordot\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"tensordot\" is not implemented for backend \"JAX_CPU\".")

def dot(a, b, out=None):
    if log_matrix_ops:
        key = np.prod(a.shape)*np.prod(b.shape[1:])
        if a.shape in logged_matrix_ops_dot:
            logged_matrix_ops_dot[a.shape] += 1
        else: 
            logged_matrix_ops_dot[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.dot(a, b, out=out)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"dot\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"dot\" is not implemented for backend \"JAX_CPU\".")

# ==========================================================================
# ========================== Tensor decompositions =========================
# ==========================================================================

def qr(a, mode='reduced'):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_qr:
            logged_matrix_ops_qr[a.shape] += 1
        else: 
            logged_matrix_ops_qr[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.qr(a, mode=mode)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"qr\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"qr\" is not implemented for backend \"JAX_CPU\".")

def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_svd:
            logged_matrix_ops_svd[a.shape] += 1
        else: 
            logged_matrix_ops_svd[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"svd\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"svd\" is not implemented for backend \"JAX_CPU\".")

def eigh(a, UPLO='L'):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_eigh:
            logged_matrix_ops_eigh[a.shape] += 1
        else: 
            logged_matrix_ops_eigh[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.linalg.eigh(a, UPLO=UPLO)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"min\" is not implemented for backend \"JAX_CPU\".")

# ==========================================================================
# ==================== Other relevant tensor operations ====================
# ==========================================================================

def transpose(a, axes=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_transpose:
            logged_matrix_ops_transpose[a.shape] += 1
        else: 
            logged_matrix_ops_transpose[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.transpose(a, axes=axes)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"transpose\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"transpose\" is not implemented for backend \"JAX_CPU\".")

def reshape(a, shape=None, order='C', newshape=None, copy=None):
    if log_matrix_ops:
        if a.shape in logged_matrix_ops_reshape:
            logged_matrix_ops_reshape[a.shape] += 1
        else: 
            logged_matrix_ops_reshape[a.shape] = 1
    if backend == Backend.NUMPY:
        return np.reshape(a, shape=shape, order=order, newshape=newshape, copy=copy)
    elif backend == BACKEND.JAX_CPU:
        raise NotImplementedError("function \"reshape\" is not implemented for backend \"JAX_CPU\".")
    elif backend == BACKEND.JAX_GPU:
        raise NotImplementedError("function \"reshape\" is not implemented for backend \"JAX_CPU\".")