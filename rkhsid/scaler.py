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


# %%
class NotFittedError(ValueError, AttributeError):
    """Custom Exception as it is defined in scikit-learn."""


class IntervalScaler:
    """Linear scaling to interval(s) [min, max], with optional margin."""

    def __init__(self, minmax=(0, 1), margin=0, n=None) -> None:
        """_summary_

        Parameters
        ----------
        minmax : 2-elem iterable, or iterable of 2-elem iterables, optional
            The input data will be scaled to [minmax[0], minmax[1]], potentially with
            margins. If a 2-elem iterable, then all dimensions will be scaled equally,
            else provide an iterable with a 2-elem list (or None) for each dimension, by default (0,1)
        margin : int, optional
            Margin, as fraction of range. If given, then the scaling is computed so that
            the input data is mapped to [minmax[0]+margin*range, minmax[1]-margin*range],
            where range = minmax[1]-minmax[0], by default 0.
        n : int, optional
            Dimensions to scale to, by default None which means `n` is inferred at fit time.
        """
        self._init_params(minmax=minmax, margin=margin, n=n)

    def _init_params(self, minmax, margin, n):
        if minmax is None:
            # A Scaler that doesn't do any scaling
            minmax = [None]
        if not hasattr(minmax[0], "__len__") and minmax[0] is not None:
            if n is not None:
                self.minmax = cp.repeat([[minmax[0]], [minmax[1]]], n, axis=1)
            else:
                self.minmax = minmax  #  need to infer when fit() is called
            self.n = n
        else:
            # replace all None with 2-elem +/- inf
            minmax_ = [mm if mm is not None else [cp.nan, cp.nan] for mm in minmax]
            self.minmax = cp.array(minmax_).T
            if n is not None and n != self.minmax.shape[1]:
                warnings.warn(
                    f"Supplied dimension n={n} and dimension of minmax "
                    f"({self.minmax.shape[1]}) disagree. The latter will be used."
                )
            self.n = self.minmax.shape[1]
        self.margin = margin
        self.coef_ = None
        # self.coef_ = None if self.n is None else cp.array([0,1]*self.n)  #  c, m in y = mx + c

    def fit(self, X):
        """Fit the scaling transform to the traning data `X`.

        Parameters
        ----------
        X : np.array
            Training data. If `X.shape==(N,)`, it is extended to `(N,1)`, i.e. it is
            assumed to be N 1-dim datapoints.

        Returns
        -------
        self
        """
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if self.n is None:
            n = X.shape[1]
            self._init_params(self.minmax, self.margin, n)
        # max and min
        xmax = cp.nanmax(X, axis=0, keepdims=True)
        xmin = cp.nanmin(X, axis=0, keepdims=True)

        # the slopes
        m = (1 - 2 * self.margin) * cp.diff(self.minmax, axis=0) / (xmax - xmin)
        # the intercepts
        c = (
            self.minmax[1, :] * (1 - self.margin)
            + self.margin * self.minmax[0, :]
            - m * xmax
        )
        # same as self.minmax[0] - m*xmin

        # the dimensions with no constraints
        c[cp.isnan(m)] = 0
        m[cp.isnan(m)] = 1

        self.coef_ = cp.vstack((c, m))
        return self

    def transform(self, X):
        """Scale `X`. Requires having called `fit()`.

        Parameters
        ----------
        X : numpy.array
            Data. If it is 1-D, it is assumed to be 1-dimensional datapoints.

        Returns
        -------
        Scaled X
            Same shape and type as `X`. Exception: If `X.shape==(N,)`, the output has shape `(N,1)`.
        """
        self.check_is_fitted()
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        return self.coef_[[1], :] * X + self.coef_[[0], :]

    def inverse_transform(self, X):
        """The inverse transform. Requires having called `fit()`"""
        self.check_is_fitted()
        if X.ndim == 1:
            X = X.reshape((-1, 1))

        return (X - self.coef_[[0], :]) / self.coef_[[1], :]

    def fit_transform(self, X):
        """Convenience function. Equivalent to `self.fit(X).transform(X)`"""
        return self.fit(X).transform(X)

    def check_is_fitted(self, msg=None):
        """Lightweight copy of a method all sklearn transformers have.
        Raises NotFittedError if the transformer hasn't been fitted yet.
        """
        # Check
        # https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/utils/validation.py#L1276
        # for details
        if msg is None:
            msg = "This estimator isn't fitted yet. Call `fit()` first."
        if self.coef_ is None:
            raise NotFittedError(msg)


class ScalerWrapper:
    def __init__(self, scaler, n, n_repeat) -> None:
        """Wraps a list of scalers, i.e. objects with a fit() and transform() method, into
        a single scaler:
            - scaler[i] accepts inputs of dimension n[i]
            - the resulting scaler will apply scaler[i] to n_repeat[i]*n[i] elements of
                the input vector

        This is useful for autoregressive models where the inputs are delayed versions of
        the same entities, e.g.
            y[t+1] = M( y[t], y[t-1], u[t], u[t-1], u[t-2] )
        With scaler = [scaler_y, scaler_u], the wrapped scaler
            ScalerWrapper( scaler, [dim_y, dim_u], [2, 3] )
        could be applied to the entire regressor
            [ y[t], y[t-1], u[t], u[t-1], u[t-2] ]

        Parameters
        ----------
        scaler : Iterable of scaler objects
            _description_
        n : Iterable of int
            n[i] is the dimensions of what scaler[i] accepts
        n_repeat : Iterable of int
            n_repeat[i] is how many copies of scaler[i] will be placed in sequence.
        """
        if not hasattr(scaler, "__len__"):
            scaler = [scaler]
        if not hasattr(n, "__len__"):
            n = [n]
        if not hasattr(n_repeat, "__len__"):
            n_repeat = [n_repeat]
        self._scaler = scaler
        self._n = n
        self._n_repeat = n_repeat
        self.__fit_has_been_called = False

    def fit(self, X):
        """Fit the wrapped scalers.

        Parameters
        ----------
        X : np.array
            Input data

        Returns
        -------
        ScalerWrapper
            Returns self
        """
        pointer_start = 0
        for ii in range(len(self._scaler)):
            n = self._n[ii]
            nr = self._n_repeat[ii]
            if not self._scaler[ii] is None:
                self._scaler[ii].fit(X[:, pointer_start : pointer_start + n])
            pointer_start += n * nr
        self.__fit_has_been_called = True
        return self

    def transform(self, X):
        """Applies the scaler to the input array `X`

        Parameters
        ----------
        X : np.array
            The input data to be transformed

        Returns
        -------
        X_scaled : np.array
            The scaled input data
        """
        pointer_start = 0
        X_out = X.copy()
        for ii in range(len(self._scaler)):
            n = self._n[ii]
            nr = self._n_repeat[ii]
            if not self._scaler[ii] is None:
                X_out[:, pointer_start : pointer_start + n * nr] = cp.hstack(
                    [
                        self._scaler[ii].transform(X[:, jj : jj + n])
                        for jj in pointer_start + n * cp.arange(nr)
                    ]
                )
            pointer_start += n * nr
        return X_out

    def check_is_fitted(self, msg=None):
        """
        Raises
        ------
        NotFittedError
            If fit() hasn't been called yet.
        """
        if not self.__fit_has_been_called:
            raise NotFittedError(msg)

    def inverse_transform(self, X):
        """Not Implemented"""
        raise NotImplementedError(
            "This Class is intended for use in the DynamicalSystemEstimator "
            "where the inverse transform is not required."
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class DummyScaler:
    """A scaler that does nothing, only returns its inputs"""

    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs):
        pass

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X

    def check_is_fitted(self, msg=None):
        pass


def _check_is_scaler(S):
    """Checks if `S` has `fit()` and `transform()` methods."""
    return hasattr(S, "fit") and hasattr(S, "transform")
