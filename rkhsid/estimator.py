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
Python implementation of the algorithm for the estimation using kernel sections and bias space.
@Authors:   Conor Dyson
            Rodrigo Ordonez-Hurtado <rodrigo.ordonez.hurtado@ibm.com>
            Jonathan Epperlein <jpepperlein@ie.ibm.com>
"""

import itertools
import scipy.special
import warnings

try:
    import cupy as cp
except ImportError:
    import numpy as cp

    cp.get_default_memory_pool = type(
        "",
        (),
        {"total_bytes": lambda *args: None, "free_all_blocks": lambda *args: None},
    )
    cp.get_default_pinned_memory_pool = type(
        "", (), {"free_all_blocks": lambda *args: None}
    )
    cp.core = type("", (), {"core": type("", (), {"ndarray": cp.ndarray})})
    cp.asnumpy = cp.array


from .scaler import IntervalScaler, NotFittedError


# %%
def linear_const_eval(x_):
    """Evaluates the constant and linear basis functions on the datapoints in `x_`

    Parameters
    ----------
    x_ : np.array
        Array of shape (N, n), with N datapoints of n dimensions. NOTE: ALWAYS must be 2-D, so add singleton dimension
        if necessary.

    Returns
    -------
    np.array
        Array of shape (N, n+1). Column[:,0] is the corresponding to the constant functions, columns[:,1:] = x_ giving
        linear functions.
    """
    if not len(x_.shape) == 2:
        raise ValueError("`x_` must have 2 dimensions!")

    return cp.hstack([cp.ones([x_.shape[0], 1]), x_])


# %%
def legendre_bias(x_, max_order=50):
    """Evaluates the first `max_order` Legendre basis functions on the datapoints `x_`. NOTE: This only yields the
    expected result if x_.shape[1]=1, i.e. these are uni-variate Legendre polynomials.

    Parameters
    ----------
    x_ : np.array
        Array of shape (N, 1), with N datapoints. NOTE: ALWAYS must be 2-D, so add singleton dimension if necessary.

    max_order : int, optional
        Determines up to which order of Legendre polynomials form the basis of the bias space.

    Returns
    -------
    np.array
        Array of shape (N, max_order). Column[:, 0] is the corresponding to the constant functions, columns[:, 1] = x_
        giving linear function, etc.
    """
    if not len(x_.shape) == 2:
        raise ValueError("`x_` must have 2 dimensions!")

    return cp.array(
        scipy.special.eval_legendre(cp.asnumpy(cp.arange(max_order)), cp.asnumpy(x_))
    )


# %%
class SinKernel:
    """Provides an `eval` method to evaluate the "sine kernel"
        K(x, y) = sum( lambda_I^{-s} sin(pi I x) sin(pi I y),
    where I = [i1, ..., in] is a multi-index and
        sin(pi I x) = sin(pi i1 x1)*sin(pi i2 x2) ... *sin(pi in xn)
    with lambda_I being the corresponding eigenvalues of the Laplacian with homogeneous Dirichlet boundary conditions.

    Attributes
    ----------
    s_power : int
        `s` in the definition.

    NUM_DIRS : int
         `n` in the definition, i.e. the dimension of the arguments.

    Methods
    -------
    eval :
        Evaluates the Kernel at all pairs of provided datapoints. Can be used as `kernel_func_eval` in
        FunctionEstimator().
    """

    def __init__(self, s_power=2, max_freq=30):
        """
        Parameters
        ----------
        s_power : int, optional
            Gives the power of the operator the user wants to define the Hilbert space on, by Sobolev embedding theorem
            4*s_power/state_space_dimension > 1 to guarantee convergence.

        max_freq : int, optional
            Defines the maximum frequency in each state-space dimension of the sine basis.
        """
        self.s_power = s_power
        self.max_freq = max_freq

        # Initializations
        self.frequencies = None
        self.inv_eigenvalues = None
        self.NUM_DIRS = None

    # %%
    def _sin_basis_func(self, x_):
        """Returns an array of individual basis vectors evaluated on the datapoints `x_`, with `shape(x_)[1]==n`.
        NOTE: the attributes `frequencies` and `inv_eigenvalues` are initialized on the first call, hence there might be
        some overhead on the first call only.

        Parameters
        ----------
        x_ : np.array
            Array of shape (N, n), with N datapoints of n dimensions. NOTE: ALWAYS must be 2-D, so add singleton
            dimension if necessary.

        Returns
        -------
        np.array
            Array of shape (N+1, max_freq). Row[0, :] is the corresponding eigenvalue (1/sum(f_i^2) for now), row [i, f]
            is the f'th frequency eigenfunction evaluated at the (i-1)-th datapoint. "frequency" here is a vector
            [f_1, f_2, ... , f_n] of wavenumbers 1 <= f_i <= max_freq.
        """

        if not len(x_.shape) == 2:
            raise ValueError("`x_` must have 2 dimensions!")

        if (self.frequencies is None) or (self.inv_eigenvalues is None):
            self.NUM_DIRS = x_.shape[1]
            frequencies = cp.array(
                tuple(
                    itertools.product(range(1, self.max_freq + 1), repeat=self.NUM_DIRS)
                )
            )
            self.inv_eigenvalues = (
                1 / cp.einsum("ij,ij->i", frequencies, frequencies) / cp.pi**2
            )
            self.frequencies = frequencies[:, cp.newaxis, :]

        return cp.vstack(
            (
                self.inv_eigenvalues ** (self.s_power * 2),
                cp.sqrt(2**self.NUM_DIRS)
                * cp.prod(cp.sin(cp.pi * (x_ * self.frequencies)), axis=2).T,
            )
        )

    # %%
    def eval(self, x_, z_):
        """Returns a Gram matrix of the kernel evaluated on all combinations of datapoint points between data vectors
        `x_` and `z_`.
        NOTE: the attributes `frequencies` and `inv_eigenvalues` are initialized on the first call, hence there might be
        some overhead on the first call only.

        Parameters
        ----------
        x_, z_ : np.arrays
            Arrays of shape (N, n) and (M, n), respectively. NOTE: ALWAYS must be 2-D, so add singleton dimensions if
            necessary.

        Returns
        -------
        np.array
            Array of shape (N, M), where each element (i, j) corresponds to K(X[i, :], Y[j, :]).
        """

        KX = self._sin_basis_func(x_)[0:1, :] * self._sin_basis_func(x_)[1:, :]
        KZ = self._sin_basis_func(z_)[1:, :]

        return KX @ KZ.T


# %%
class FunctionEstimator:
    """Function estimator class, with a `fit(X, y)` method to train on d-dimensional observations, and
    `eval(X)` method to evaluate on (a) new datapoint(s).

    Attributes
    ----------
    coef_slices_matrix : numpy array
        Gives the coefficients of the kernel slices in Kernel Slice/Bias space basis decomposition of the approximated
        model.

    coef_bias_matrix : numpy array
        Gives the coefficients of the bias space functions in Kernel Slice/Bias space basis decomposition of the
        approximated model.

    bias_basis_func_eval : numpy array
        Evaluates all the training input datapoints at each function in the bias space.

    kernel_func_eval : numpy array
        Evaluates the kernel at all combinations of the training input datapoints.

    _scaler :
        Scaler object with fit(), transform(), and inverse_transform() methods, like those in
        sklearn.preprocessing.StandardScaler.

    Methods
    -------
    fit(X, y) :
        Fits the model to the input and target data.

    eval(x_) :
        Evaluates the fitted model on an array of datapoints x_.
    """

    def __init__(
        self,
        bias_func_eval=None,
        kernel_func_eval=None,
        scaling=None,
        scale_margin=0,
        data_cost_weight=1,
        rkhs_weight=1,
        bias_weight=0.0,
    ):
        """
        Parameters
        ----------
        bias_func_eval : Callable, optional, default=None
            Computes the individual basis vectors evaluated on the datapoints in x_. It should read in a numpy array
            of shape (N, n), with N datapoints of n dimensions and return an array of shape (N,b), with b the number
            of basis functions. None by default, i.e. there is no bias space. NOTE: Inputs ALWAYS must be 2-D, so if
            your datapoints are 1-D, then make sure that X.shape == (N, 1) and NOT (N,). If you only have a single
            datapoint, then make sure that X.shape == (1, n) it should then return a numpy array of shape (N, m), where
            N is the number of datapoints and m is the number of bias space basis functions.

        kernel_func_eval : Callable, optional, default=None (which results in the use of SinKernel().eval)
            Computes the kernel Gram matrix for all combinations of input datapoints. Must take in two numpy arrays
            x_, z_ of shapes (N, n) and (M, n), respectively, and return an (N, M) array with (i, j) element equal to
            K(x_[i, :], z_[j, :]). By default None, which results in the use of SinKernel().eval. NOTE: Inputs ALWAYS
            must be 2-D as with bias_func_eval. Must return a np.array of shape (N, M), where each element (i, j) is
            K(x_[i, :], z_[j, :]), with K being the kernel for the given RKHS.

        scaling: 2-elem iterable, iterable of 2-elem iterable, or object with fit() & transform() methods, default=None
            Applies scaling to the input data before fitting, and also before every call to eval(). Useful if (part of)
            the domain of the RKHS has boundaries. None by default, which means no scaling.
            Examples:
                (0,1) -- linearly scale all variables to 0,1
                [(0,1), None, (-1,1)] -- scale 1st argument t0 [0,1], don't scale 2nd
                                         argument at all, and 3rd argument to [-1,1].
                sklearn.preprocessing.StandardScaler() -- Use one of the scikit-learn scalers

        scale_margin: 0 <= float < 0.5
            If bounds are supplied for scaling, the scaling is computed so the training
            data leaves a margin of scale_margin*range on both sides.
            Ignored if a scaler object is supplied for  `scaling`.

        data_cost_weight : float, optional
             relative weight on the data fit term in the minimisation problem


        rkhs_weight : float, optional
             relative weight on the Hilbert space norm of the orthogonal projection of M onto the V_2s Hilbert space

        bias_weight : float, optional
             relative weight on the bias space norm of the orthogonal projection of M onto the bias space
        NOTE: The above 3 parameters can be supplied to the `fit()` method as well, in which case the ones
        supplied to the constructor will be overwritten. They are included here mostly to facilitate use of
        scikit-learn's GridSearchCV and similar tools.
        """
        # Input initializations
        self.X_scaled = None
        self.bias_basis_func_eval = bias_func_eval

        if kernel_func_eval is None:
            self.kernel_func_eval = SinKernel().eval
        else:
            self.kernel_func_eval = kernel_func_eval

        # Scaling
        self._init_scaling(scaling, scale_margin)

        # Further initializations
        self.coef_slices_matrix = None
        self.coef_bias_matrix = None
        self.X = None
        self.y = None
        self.bias_func_eval_matrix = None
        self.bias_func_norms_matrix = None
        self.kernel_func_eval_matrix = None

        self.data_cost_weight = data_cost_weight
        self.rkhs_weight = rkhs_weight
        self.bias_weight = bias_weight

    # %%
    def _init_scaling(self, scaling, scale_margin):
        """(Re-)initializes the input scaling, and leaves everything else the same."""
        if scaling is None:
            self._scaler = None
        elif hasattr(scaling, "fit") and hasattr(scaling, "transform"):
            # We have a scaler object already
            self._scaler = scaling
        else:
            try:
                self._scaler = IntervalScaler(minmax=scaling, margin=scale_margin)
            except Exception as e:
                e.args += "Error initializing scaling. Check the docstring for the scaling argument."
                raise e

    # %%
    def fit(self, X, y, data_cost_weight=None, rkhs_weight=None, bias_weight=None):
        """Given V_2s is the original Hilbert space defined by the reproducing kernel and H_B is the appended bias
        space, then denote P_1 as the orthogonal projector from V_2s X H_B onto V_2s and P_0 as the orthogonal projector
        from V_2s X B onto H_B. And given y_i = M(D_i) + e, where e is some Gaussian noise, then
             M_hat =  argmin_{M_hat} (  r * ( sum_{i=1}^{N} ( y_i - M_hat(D_i) )**2
                                     + q1 * || P_1(M_hat) ||_{V_2s}**2
                                     + q2 * || P_0(M_hat) ||_{H_B}**2 )
        is our estimate for M with the above solution having an equivalent interpretation as the solution to a min-max
        problem on an ellipse. The function of the fit method is to solve for coef_slices_matrix and coef_bias_matrix,
        such that
             M_hat(D) = kernel_func_eval_matrix @ coef_slices_matrix + bias_func_eval_matrix @ coef_bias_matrix
        satisfies the above argmin equation.

        Parameters
        ---------
        X : np.array
             Array of shape (N, n), with N datapoints of dimension n (dimension of the state space of the model). NOTE:
             ALWAYS must be 2-D, so if your datapoints are 1-D, then make sure that X.shape == (N, 1) and NOT (N,). If
             you only have a single datapoint, then make sure that X.shape == (1, n).

        y : np.array
             Array of shape (N, d), with N outputs of dimension d (dimension of the model output). NOTE: ALWAYS must be
             2-D, so if your outputs are 1-D, then make sure that y.shape == (N, 1) and NOT (N,). If you only have a
             single output, then make sure that y.shape == (1, d).

        data_cost_weight : float, optional, default=None (which results in the use of values in __init__)
             Relative weight on the data fit term in the minimization problem.

        rkhs_weight : float, optional, default=None (which results in the use of values in __init__)
             Relative weight on the Hilbert space norm of the orthogonal projection of M onto the V_2s Hilbert space.

        bias_weight : float, optional, default=None (which results in the use of values in __init__)
             Relative weight on the bias space norm of the orthogonal projection of M onto the bias space.

        Returns
        -------
        self :
             The attributes `self.coef_slices_matrix` (array of shape (N, d), with N datapoints of dimension d
             (dimension of the model output) and `self.coef_bias_matrix` (array of shape (N, d), with N datapoints of
             dimension d (dimension of the model output)) are now populated, and `eval()` can be used.
        """

        # If parameters were supplied to fit(), overwrite the attributes.
        if data_cost_weight is not None:
            self.data_cost_weight = data_cost_weight
        if rkhs_weight is not None:
            self.rkhs_weight = rkhs_weight
        if bias_weight is not None:
            self.bias_weight = bias_weight

        r = self.data_cost_weight
        q1 = self.rkhs_weight
        q2 = self.bias_weight

        self.X = cp.array(X)
        self.y = cp.array(y)

        if self._scaler is None:
            X_scaled = self.X
        else:
            try:
                self._scaler.check_is_fitted()
            except NotFittedError:
                self._scaler.fit(X)
            except AttributeError:
                # In sklearn, check_is_fitted is a utility function that must be called on the estimator object.
                # Importing that function here would introduce sklearn dependence. Instead, we monkey-patch the _scaler.
                self._scaler.fit(X)
                self._scaler.check_is_fitted = lambda *args, **kwargs: None

            X_scaled = self._scaler.transform(X)
        self.X_scaled = cp.array(X_scaled)

        try:
            N, n = X.shape
            _, d = y.shape
        except ValueError as e:
            e.args += (
                f"X and y must have EXACTLY 2 dimensions. X has {X.ndim}, y has {y.ndim}",
            )
            raise e

        # If user does not wish to use bias space, they can set bias=None and the system will be solved without the
        # use of an appended space.

        self.kernel_func_eval_matrix = self.kernel_func_eval(
            self.X_scaled, self.X_scaled
        )

        I_matrix = cp.eye(N)

        # If bias space functions are provided, then the system will solve the full version of equations given below.
        # Solving this system of equations yields the coefficients of the kernel slice and bias space function
        # decomposition of the true answer, such that the true model is given by
        #       M_hat(x) = \sum_{i=1}^{N} a_i * K(D_i, x) + \sum_{j=1}^{m} b_i * g_i(x)
        if self.bias_basis_func_eval is not None:
            self.bias_func_eval_matrix = self.bias_basis_func_eval(self.X_scaled)
            self.bias_func_norms_matrix = cp.eye(self.bias_func_eval_matrix.shape[1])

            inv_q2_C_Plus_CTRC_Times_CTR = (
                cp.linalg.inv(
                    q2 * self.bias_func_norms_matrix
                    + r * self.bias_func_eval_matrix.T @ self.bias_func_eval_matrix
                )
                @ self.bias_func_eval_matrix.T
                * r
            )

            P_q2 = (
                r * I_matrix
                - r * self.bias_func_eval_matrix @ inv_q2_C_Plus_CTRC_Times_CTR
            )

            self.coef_slices_matrix = cp.linalg.solve(
                q1 * I_matrix + P_q2 @ self.kernel_func_eval_matrix, P_q2 @ self.y
            )

            self.coef_bias_matrix = inv_q2_C_Plus_CTRC_Times_CTR @ (
                self.y - self.kernel_func_eval_matrix @ self.coef_slices_matrix
            )

            if cp.isnan(self.coef_bias_matrix).any():
                raise ValueError(
                    "Resulting `coef_bias_matrix` has nan values. "
                    "Potential reason: exogenous input is a constant signal."
                )

        # If bias_func_eval=None, then we instead solve the simplified system of equations seen below to yield the
        # coefficients of the simpler decomposition written only in terms of kernel sections, whis is given by
        #       M_hat(x) = \sum_{i=1}^{N} c_i * K(D_i, x)
        else:
            self.coef_slices_matrix = cp.linalg.solve(
                (q1 / r) * I_matrix + self.kernel_func_eval_matrix, self.y
            )

        if cp.isnan(self.coef_slices_matrix).any():
            raise ValueError(
                "Resulting `coef_slices_matrix` has nan values. "
                "Potential reason: exogenous input is a constant signal."
            )

        return self

    # %%
    def eval(self, x_, return_cupy=False):
        """Evaluates the estimated functions at the points in x_.

        Parameters
        ----------
        x_ : np.array
            Array of shape (N, n), with N datapoints of dimension n (dimension of the state space of the model). NOTE:
            ALWAYS must be 2-D, so if your datapoints are 1-D, then make sure that x_.shape == (N, 1) and NOT (N,). If
            you only have a single datapoint, then make sure that x_.shape == (1, n).

        return_cupy : bool, default=False
            If True, returns output in cupy format (providing cupy support is available).

        Returns
        -------
        np.array
            Array of shape (N, d), with N datapoints of dimension d (dimension of the model output).
        """
        if self._scaler is None:
            x_ = cp.array(x_)
        else:
            x_ = cp.array(self._scaler.transform(x_))

        # With bias space, it gives: M_hat(x) = \sum_{i=1}^{N} a_i * K(D_i, x) + \sum_{j=1}^{m} b_i * g_i(x)
        if self.bias_basis_func_eval is not None:
            M_hat = self.coef_slices_matrix.T @ self.kernel_func_eval(
                self.X_scaled, x_
            ) + (self.coef_bias_matrix.T @ self.bias_basis_func_eval(x_).T)

        # Without bias space, it gives: M_hat(x) = \sum_{i=1}^{N} c_i * K(D_i, x)
        else:
            M_hat = self.coef_slices_matrix.T @ self.kernel_func_eval(self.X_scaled, x_)

        if return_cupy:
            return M_hat.T
        return cp.asnumpy(M_hat.T)

    # predict is defined as an alias for eval to adhere to sklear estimator protocol
    predict = eval

    def get_params(self, deep=False):
        """Returns the parameters changeable between calls to `fit()`.
        Exists for partial adherence to sklean estimator protocol.
        `deep` is ignored."""
        out = {
            "rkhs_weight": self.rkhs_weight,
            "data_cost_weight": self.data_cost_weight,
            "bias_weight": self.bias_weight,
        }
        return out

    def set_params(self, **params):
        """Sets the parameters changeable between calls to `fit()`. Call `get_params()`
        to see what those are.
        Exists for partial adherence to sklean estimator protocol so this class can be
        used in GridSearchCV etc"""
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                warnings.warn(
                    f"Parameter {key} is not settable for {self.__class__}.\n"
                    f"Currently settable parameters are {valid_params.keys()}."
                )
        return self

    # %%
    predict = eval  # NOTE: Predict is defined here as an alias for eval to adhere to sklearn estimator protocol.

    # %%
    def get_params(self, deep=False):
        """Returns the parameters changeable between calls to `fit()`.
        NOTE: Exists for partial adherence to sklearn estimator protocol.
        `deep` is ignored here but needed for compatibility.
        """
        out = {
            "rkhs_weight": self.rkhs_weight,
            "data_cost_weight": self.data_cost_weight,
            "bias_weight": self.bias_weight,
        }
        return out

    # %%
    def set_params(self, **params):
        """Sets the parameters changeable between calls to `fit()`. Call `get_params()` to see what those are.
        NOTE: Exists for partial adherence to sklean estimator protocol so this class can be used in GridSearchCV, etc.

        Parameters
        ----------
        **params : Keyword-arguments like
            Keyword arguments.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid_params = self.get_params()
        for key, value in params.items():
            if key in valid_params:
                setattr(self, key, value)
            else:
                warnings.warn(
                    f"Parameter {key} is not settable for {self.__class__}.\n"
                    f"Currently settable parameters are {valid_params.keys()}."
                )
        return self

    # %%
    @staticmethod
    def _parse_array(x, name):
        """Supporting funtion to parse input arguments.

        Parameters
        ----------
        x : array like (expected)
            Input arguments.

        name : string
            Name of the input argument, used to create verbose error messages.

        Returns
        -------
        self
        """

        x_ = x.copy()
        if hasattr(x_, "shape"):
            if not isinstance(x_, cp.ndarray):
                x_ = cp.array(x_)

            if len(x_.shape) == 1:
                return [cp.array([x_])]
            elif len(x_.shape) == 2:
                return [x_]
            elif len(x_.shape) == 3:
                return x_
            else:
                raise ValueError(
                    f"Value error for {name}: array/list of 2D arrays is expected."
                )

        elif isinstance(x_, list):
            for ii, xi in enumerate(x_):
                if not hasattr(xi, "shape"):
                    raise ValueError(
                        f"Value error for {name}: array/list of 2D arrays is expected."
                    )
                if len(xi.shape) != 2:
                    raise ValueError(
                        f"Value error for {name}: array/list of 2D arrays is expected."
                    )

                if not isinstance(xi, cp.ndarray):
                    x_[ii] = cp.array(xi)

            return x_

        else:
            raise ValueError(
                f"Value error for {name}: array/list of 2D arrays is expected."
            )
