# (C) Copyright IBM Corp. 2023

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

"""
Python implementation of the kernel for the biharmonic equation on a disk and a ball.
@Authors:       Adam Byrne <byrne.adam@ibm.com>
                Jonathan Epperlein <jpepperlein@ie.ibm.com>
@Contributors:  Rodrigo Ordonez-Hurtado <rodrigo.ordonez.hurtado@ibm.com>
              
The kernel for the cross-stream direction is a 1-D kernel with Dirichlet boundary conditions (BCs), which is made up of
the eigenfunctions of d^4/dx^4, namely
    K_1(x, y) = \sum_{i=1}^\infty 1/(i \pi)^4 \sin(i \pi x) \sin(i \pi y) \qquad x,y\in[0,1].

The kernel for the other two directions is just a Thin plate spline (TPS) kernel:
    K_2(x, y) = r^2 \log(r) \quad r:=\|x-y\|_2 \quad x,y\in\mathbb{R}^2.

Note that this TPS kernel does not define a positive definite function, but only a _conditionally_ positive definite
function. Namely, the coefficients c_i and points x_i in \sum_{ij} c_i c_j K(x_i, x_j) \geq0 must satisfy the
additional condition
    \sum_{i} c_i (x_i)_j^\ell=0, \forall j, \ell=0, 1.

In other words: given a basis p_\ell, \ell=0, \dotsc, q-1 of the bias space (in this case, just span\{1, x_1, x_2\}, but
in the general case this can involve higher-order polynomials), and the matrix T=[p_\ell(x_i)]_{\ell, i}, the
coefficients c have to satisfy Tc=0.

We can use the RBF kernel
    K_2(x, y) = e^{-\alpha\|x-y\|_2^2}
where \alpha is a design parameter.

The mixed kernel
    K:[0, 1]\times\mathbb{R}^2\rightarrow\mathbb{R}
is
    K(x, y) = K_1(x_1, y_1) K_2([x_2, x_3], [y_2, y_3]).

NOTE: This introduces dependency on numba    

"""
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


def RBF_kernel(x, y, alpha=1.0):
    """Evaluates the Radial basis function (RBF) kernel on inputs.

    Parameters
    ----------
    x, y : array like
        [As in scipy.spatial.distance] Input values.

    alpha : float
        Width parameter of the RBF kernel, smaller alpha leades to wider kernel.

    Returns
    -------
    ndarray
        RBF kernel evaluated on x and y.
    """
    return np.exp(-(alpha**2) * cdist(x, y, metric="sqeuclidean"))


def sine_kernel_1D_def(max_freq=20, s=2):
    """Returns a function `func` so that `func(x,y)` is the 1-D sine kernel with `max_freq`
    terms. x and y are 1-D arrays, their dimensions are shifted automatically so that
    broadcasting yields a dim(x) by dim(y) array.

    Parameters
    ----------
    max_freq : int
        Number of frequencies used in kernel. Note that the kernel then spans a
        _finite_ dimensional space with dimension `max_freq`.
    s : int
        The order of derivative being penalized in the RKHS norm. s=2 corresponds to the
        classical splines, where ||f''|| is penalized.

    Returns
    -------
    function
        A function evaluating the sine kernel (with `max_freq` frequencies).
        The function signature is
          (1-D np.array size m, 1-D np.array size n) -> 2-D np.array shape (m,n)
    """
    frequencies = (np.arange(max_freq) + 1)[np.newaxis, :] * np.pi
    eigs = 1 / frequencies ** (2 * s)

    def kernel(x, y):
        return 2 * np.einsum(
            "ik,jk->ij",
            np.sin(frequencies * x[:, np.newaxis]) * eigs,
            np.sin(frequencies * y[:, np.newaxis]),
        )

    return kernel


def TPS_kernel(x, y, scale=2):
    """Computes the 2-D Thin plate spline (TPS) kernel
         K(x,y) = r**2 log(r),
    where r is the Euclidean distance between x and y, on all pairs
    `x[i,:], y[j,:]`.
    Parameter `scale` is required for numerical reasons only: to avoid overflows in
    r**r, Computations are scaled so that the worst-case power in the computation
    is max(dist(x,y))**scale.

     Parameters
     ----------

     x, y : (n,p) and (n,q) array like
         Input values.

     scale : float
         Scale, used to avoid overflows in r**r.

     Returns
     -------
     (p,q) ndarray
         The 2-D TPS kernel evaluated on all pairs x[i,:], y[j,:]
    """
    # Uses r^2 log(r) = r log(r^r) = r0 r log(r^(r/r0))
    r = cdist(x, y)
    r0 = np.max(r) / scale
    res = r0 * r * np.log(r ** (r / r0))

    return res


# --------------------------- #
# -- Kernel Implementation -- #
# --------------------------- #


# Numba-based vectorization. We must use decorator for every fn involved in x-y 'for' loops.
@njit
def log(x):
    """Vectorized version of numpy's element-wise natural logarithm function.

    Parameters
    ----------
    x : array like
        Input value.

    Returns
    -------
    ndarray
        Element-wise natural logarithm of x.
    """
    return np.log(x)


@njit
def norm(x):
    """Vectorized version of numpy.linalg.norm.

    Parameters
    ----------
    x : array like
        Input value.

    Returns
    -------
    float
        Norm of x, Frobenius if matrix or 2-norm if vectors.
    """
    return np.linalg.norm(x)


@njit
def GK2(x, y):
    """Vectorized version of Green's function for R^2.

    Parameters
    ----------
    x, y : array like
        Input values.

    Returns
    -------
    ndarray
        Green's function for R^2 on x and y.
    """
    x_n = norm(x)
    y_n = norm(y)
    xy_n = norm(x - y)
    xxy_n = norm((x / norm(x)) - (norm(x)) * y)

    return 0.25 * (
        2 * xy_n * (log(xy_n ** (xy_n / 2)))
        - xy_n**2
        - (norm(x / x_n - x_n * y) ** 2) * (log(xxy_n) - 1)
        - (x_n**2 - 1) * (y_n**2 - 1) * (0.5 - log(xxy_n))
    )


@njit
def GK3(x, y):
    """Vectorized version of Green's function for R^3.

    Parameters
    ----------
    x, y : array like
        Input values.

    Returns
    -------
    ndarray
        Greens function for R^3 on x and y.

    """
    x_n = norm(x)
    y_n = norm(y)
    xy_n = norm(x - y)
    xxy_n = norm((x / norm(x)) - (norm(x)) * y)

    return 0.5 * (
        -xy_n + xxy_n - 0.5 * ((x_n**2) - 1) * ((y_n**2) - 1) * (xxy_n ** (-1))
    )


@njit
def greens_matrix(x, z, g_matrix):
    """Vectorized version of Green's matrix.

    Parameters
    ----------
    x, z : array like
        Input values.

    g_matrix : matrix like
        Input Green's matrix. Remark: this input is completely rewritten, and its initial values never used. It is in
        place to prevent numba complaining about creation of a np.zeros matrix within a numba decorated function.

    Returns
    -------
    ndarray
        Updated Green's matrix.
    """
    n = x.shape[0]
    m = z.shape[0]

    # Done in this very clunky way to avoid using functions as first-class type objects, which numba does not like.
    if x.shape[1] == 2:
        for i in range(n):
            for j in range(m):
                g_matrix[i, j] = GK2(x[i, :], z[j, :])

    if x.shape[1] == 3:
        for i in range(n):
            for j in range(m):
                g_matrix[i, j] = GK3(x[i, :], z[j, :])

    return g_matrix


# 'Wrapping' to enable usage of np.zeros, which numba also does not like.
def greens_kernel_eval(x, z):
    """Evaluates Green's kernel on x and z.

    Parameters
    ----------
    x, z : array like
        Input values.

    Returns
    -------
    ndarray
        Green's function evaluated on x and z.
    """

    # NOTE: Since np.zeros cannot be used inside a @njit decorated function, it has to be passed as input argument.
    m = np.zeros([x.shape[0], z.shape[0]])
    greens_matrix(x, z, m)

    return m


def product_kernel_def(max_freq=20, k2=RBF_kernel):
    """Returns a function `func(x,y)` to compute the a kernel obtained as the product of
    a 1D sine kernel as yielded by `sine_kernel_1D_def` and an aribitrary kernel supplied
    as `k2`.
    The 1D kernel k1 is always applied to the first dimension of the input data:
        K(x, y) = k1(x[0], y[0]) · k2(x[1:], y[1:])

    Parameters
    ----------
    max_freq : int, default=20
        Maximum number of frequencies to be used.

    k2 : Kernel like, default=RBF_kernel
        Input kernel.

    Returns
    -------
    product_kernel : Function
        Kernel product between input kernel and sin_kernel.

    """
    sin_kernel = sine_kernel_1D_def(max_freq=max_freq)

    def product_kernel(x, y):
        return sin_kernel(x[:, 0], y[:, 0]) * k2(x[:, 1:], y[:, 1:])

    return product_kernel
