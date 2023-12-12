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
Utility functions for model evaluations (performance measures) and hyperparamater
selection (grid search).

As performance measures, we take
    \mathrm{RMS}(X, Y) = \frac{1}{N\cdot\hat{X}} \sqrt{\sum_I (X(I)-Y(I))^2},
where I can be a multidimensional index, N is the number of elements in X, and \hat{X} is a normalising value,
by default \max_I |X(I)|, and the normalized cross correlation
    \mathrm{NCC}(X, Y) = \frac{ \sum_I (X_I - \bar{X})(Y_I - \bar{Y}) }{\sigma_X \sigma_Y },
where \bar{\bullet} denotes the mean, and \sigma_Z = \sqrt{\sum_I (Z_I-\bar{Z})^2} is almost the stddev, just without
normalization.

pgrid_iterator and GridSearchCV are implementing a grid search, the latter is very similar
to the class of the same from Scikit-learn.
"""
import time
from itertools import product
from copy import deepcopy
import numpy as np
import sklearn.model_selection as skmod


def RMS(
    x_true,
    x_approx,
    axis=None,
    missing_value=np.nanmean,
):
    """Root Mean Square (RMS) error for the provided inputs.

    Parameters
    ----------
    x_true, x_approx : array like
        Input values.

    axis : None or int or tuple of ints, default=None
        [As in np.sum and np.max] Axis or axes along which max/sum operations are performed.
        By default None, in which case the arrays are flattened.

    missing_value : Callable, by default np.nanmean
        Method for the imputation of missing values in x_approx.
        Ensure that the function deals with NaNs gracefully, as the whole array is
        provided as input.

    Returns
    -------
    ndarray
        RMS error for the provided x_true and x_approx input parameters.
    """

    x2 = x_approx.copy()
    filler = missing_value(x_approx, axis=0, keepdims=True)
    x2[np.isnan(x_approx)] = np.broadcast_to(filler, x_approx.shape)[np.isnan(x_approx)]
    sq_diff = (x_true - x2) ** 2

    normalizer = np.max(np.abs(x_true), axis=axis)

    return np.sqrt(np.sum(sq_diff, axis=axis) / x_true.size) / normalizer


def NCC(x_true, x_approx, axis=None):
    """Normalized cross correlation (NCC) for the provided inputs.

    Parameters
    ----------
    x_true, x_approx : array like
        Input values.

    axis : None or int or tuple of ints, default=None
        [As in np.mean and np.sum] Axis or axes along which sum/mean operations are performed.

    Returns
    -------
    ndarray
        NCC for the provided x_true and x_approx input parameters.
    """
    m1 = np.mean(x_true, axis=axis, keepdims=True)
    m2 = np.mean(x_approx, axis=axis, keepdims=True)
    return np.sum((x_true - m1) * (x_approx - m2), axis=axis) / np.sqrt(
        np.sum((x_true - m1) ** 2, axis=axis) * np.sum((x_approx - m2) ** 2, axis=axis)
    )


def L2(x_true, x_approx, axis=None):
    """L2-norm error for the provided inputs.

    Parameters
    ----------
    x_true, x_approx : array like
        Input values.

    axis : None or int or tuple of ints, default=None
        [As in np.nansum] Axis or axes along which nansum operations are performed.

    Returns
    -------
    ndarray
        L2-norm error for the provided x_true and x_approx input parameters."""
    sq_diff = (x_true - x_approx) ** 2
    return np.sqrt(np.nansum(sq_diff, axis=axis) / np.nansum(x_true**2, axis=axis))


# Functions for the Grid Search.
def pgrid_iterator(p_grid: dict):
    """A grid generator. Best explained by example:
        ````
        p_grid = { 'p1': [4,5,6], 'p2': 'ab' }
        for p in pgrid_iterator(p_grid):
            print(p)
        ```
    yields
        ```
        {'p1':4, 'p2': 'a'}
        {'p1':4, 'p2': 'b'}
        {'p1':5, 'p2': 'a'}
        ...
        {'p1':6, 'p2': 'b'}
        ````
    """
    grid = sorted(p_grid.items())
    keys, values = zip(*grid)
    for grid_entry in product(*values):
        yield {k: v for k, v in zip(keys, grid_entry)}


def GridSearchCV(
    estimator,
    X,
    y,
    param_grid,
    scoring=RMS,
    average=np.mean,
    cv=5,
    refit=True,
    high_score_is_good=True,
    verbose=False,
):
    """Perform a grid search over all parameter combinations in the parameter grid `param_grid`.

    Parameters
    ----------
    estimator : estimator object
        Estimator object, assumed to implement the scikit-learn estimator interface,
        namely set_params(), fit(), and predict() methods have to exist.

    X, y : array like
        Training inputs and outputs. Note that X and y HAVE to be 2-D, and that
        X.shape[0] == y.shape[0] (i.e. the number of data points has to be the same)

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (str) as keys and lists of
        parameter settings to try as values, e.g.
            {'alpha':[0.1, 0.5, 0.9], 'fit_intercept': [True, False]}

    scoring : callable
        Strategy to evaluate the performance of the cross-validated model
        on the test set. By default RMS()

    average : np.mean-like function, default=np.mean
        Averaging to be performed on CV results for each parameter setting.

    cv : int, default=5
        Number of CV folds per tested parameter setting.

    refit : bool, default=True
        Refit an estimator using the best found parameter setting on the full dataset X,y
        If False, the estimator trained on the last fold using the best parameter setting
        is returned. Note that this relies on copy.deepcopy() being able to copy the
        estimator object.
        Generally, refit==True is preferred.

    high_score_is_good : bool
        If your scoring function is to be minimized, e.g. a loss or cost, then set this to
        False, indicating that a lower score is better.
        If your scoring function is to be maximized, e.g. a correlation or likelihood,
        then set this to True.
        By default False.

    verbose : bool
        Controls the verbosity.

    Returns
    -------
    estimator :
        The estimator fitted to the whole training set (X, y) using the best parameter
        settting if refit==True;
        otherwise, the estimator fitted during grid search to the last CV fold with the
        best parameter setting.

    cv_results : list of dictionaries
        cv_results[k] = {
            'params': Parameter values used in k-th combination.
            'cv_scores': The cross validation score of each fold.
            'mean_score', 'median_score', 'average_score': the corresponding average scores.
            'train_indices': The indices of the training data used in each fold.
            'coefs': Currently not used.
            }

    best_performance_index :
        Index of the parameter set with the best performance, namely cv_results[best_performance_index][params].
    """
    cv_results = {}
    sign = 1 if high_score_is_good else -1
    best_performance = -sign * np.inf
    best_performance_index = 0

    for i_g, params in enumerate(pgrid_iterator(param_grid)):
        splitter = skmod.KFold(n_splits=cv)
        estimator.set_params(**params)
        cv_scores_ = []
        cv_coefs_ = []
        cv_train_indices_ = []

        if verbose:
            t0 = time.perf_counter()

        for i_cv, (i_train, i_test) in enumerate(splitter.split(X, y)):
            y_est = estimator.fit(X[i_train, :], y[i_train, :]).predict(X[i_test, :])
            cv_scores_.append(scoring(y[i_test, :], y_est))
            cv_train_indices_.append(i_train)
            # cv_coefs_.append( estimator.coef_ )  # TODO: Should we remove this originally commented line?

        curr_performance = average(cv_scores_)
        if sign * curr_performance > sign * best_performance:
            best_performance_index = i_g
            best_performance = curr_performance
            if not refit:
                best_est = deepcopy(estimator)

        if verbose:
            print(f"Parameter combination {i_g} took {time.perf_counter() - t0:.2f}s.")

            print(
                f"\tCurrent best performance: {best_performance}. Last performance: {curr_performance}"
            )

        cv_results[i_g] = {
            "params": params,
            "scores": cv_scores_,
            "coefs": cv_coefs_,
            "average_score": curr_performance,
            "mean_score": np.mean(cv_scores_),
            "median_score": np.median(cv_scores_),
            "train_indices": cv_train_indices_,
        }
    if refit:
        if verbose:
            print("Refitting...")
        estimator.set_params(**cv_results[best_performance_index]["params"])
        estimator.fit(X, y)
        return estimator, cv_results, best_performance_index
    else:
        return best_est, cv_results, best_performance_index
