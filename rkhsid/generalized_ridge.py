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
Provides GeneralizedRidge, an extension (and subclass) of scikit-learn's Ridge class,
allowing for a prior estimate of coef_ (and intercept_) as well as a positive
definite weight matrix Q in the regularizer instead of just a scalar weight alpha.

@Authors:   Jonathan Epperlein <jpepperlein@ie.ibm.com>
"""

import numpy as np
from sklearn.linear_model import Ridge


class GeneralizedRidge(Ridge):
    """A sublcass of sklearn.linear_model.Ridge, allowing for an elliptical norm and
    non-zero prior guess of the weights to shrink towards. Fitting to data (X,y) minimizes
    the cost
        J(beta) =  || y - X @ beta||_2^2 + alpha (beta - beta0)^T @ Q @ (beta - beta0)
    where
        alpha is a scalar weight, as in Ridge
        Q is a positive definite matrix defining the norm used for the ridge penalty
        beta0 is an initial guess, or prior, to shrink towards

    Setting beta0 = 0, Q = np.eye(n_features) is (almost) equivalent to using Ridge.

    The intercept is absent in the cost, because it is absorbed into beta, and X is augmented
    with a column of 1's. In regular Ridge, the intercept is not penalized, but here this
    would require Q[0,:] = Q[:,0], which violates the positive definitess requirement, and
    hence can not be replicated.

    However, if `fit_intercept` is set to `False`, there is no such problem, and beta0 = 0,
    Q = np.eye(n_features) is exactly equivalent to using Ridge.

    The generalization is achieved by transforming the input data (X,y) as follows:
        P is the matrix square root of Q, i.e. the pd matrix s.t. Q = P@P.
        P_inv is its inverse.
        Then let
            Y = y - X @ beta0
            X' = X @ P_inv
        and convince yourself that if gamma minimizes
            J'(gamma) = ||Y - X' @ gamma||_2^2 + alpha ||gamma||_2^2
        (i.e. the regular Ridge cost function), then
            beta := P_inv @ gamma + beta0
        minimizes J(beta) as above.
    Hence, the regular Ridge's fit() method can be used once the data has been transformed.

    The necessary adjustments are made so that from the outside, GeneralizedRidge can be
    used just like Ridge and the other linear models.

    NOTE: If the output has more than 1 dimension, i.e. n_targets>1, all rows of beta (
    1 row per target) are regularized with the same Q-matrix; it is not possible to have
    different weights for different target dimensions, you have to solve the problem as
    multiple single-regression problems in this case.

    Attributes
    ----------
        Q : np.array
            (n_coeff, n_coeff) positive-definite matrix. Weight matrix for the regularization
            penalty. n_coeff==n_features+1 if fit_intercept, else n_coeff==n_features
        coef0 : np.array
            (n_targets, n_coeff) array; inital guess that the weights are shrank towards.
        alpha: scalar
            Just like in Ridge
        fit_intercept: True/False

        coef_ : np.array
            (n_targets, n_features) or (n_features,) array
        intercept_ : np.array
            (n_targets,) array if fit_intercept, else None
    Methods
    -------
        fit, predict, set_params, etc...
    Examples
    --------
        import numpy as np
        x = np.random.rand(100, 1) * 2
        y = 0.5 + 2.5 * x + np.random.randn(100, 1) * 0.2
        y2 = 0.25 + 1.3 * x + np.random.randn(100, 1) * 0.4
        Y = np.column_stack([y, y2]) # so n_target = 2
        ridge2 = GeneralizedRidge(
           alpha=1, fit_intercept=True,
           Q=np.diag([3, 10]), coef0=np.array([[0.5, 2], [0.2, 0]])
                    )
        ridge2.fit(x, Y)
    """

    def __init__(
        self,
        alpha=1.0,
        Q=1.0,
        coef0=0.0,
        *,
        fit_intercept=True,
        copy_X=True,
        max_iter=None,
        tol=1e-4,
        solver="auto",
        positive=False,
        random_state=None
    ):
        """
        Parameters
        ----------
        alpha : float, optional
            Scalar Ridge penalty, by default 1.0
        Q : np.array, optional
            Positive definite matrix defining the regularizer norm by coef @ Q @ coef. If
            fit_intercept is true, the first index corresponds to the intercept; in particular,
            if Q is a diagonal matrix, the first diagonal element is the penalty on the
            intercept. by default 1.0
        coef0 : np.array, optional
            Initial guess (or prior) for the parameter. Needs to be either a scalar, or
            a (n_targets, n_coeff) array. n_coeff has to be incremented by 1 if fit_intercept
            is true. by default 0.0
        fit_intercept : bool, optional
            As in Ridge, by default True
        """
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            positive=positive,
            random_state=random_state,
        )

        self.Q = None  # Gets set by code in next line, just has to exist first
        self.coef0 = coef0
        self = self.set_params(Q=Q, fit_intercept=fit_intercept)

    # ^ This is flagged by sklearn's check_estimator() as violating the sklearn API
    # > AssertionError: Estimator WeightedRidge should not set any attribute apart from parameters during init. Found attributes ['P_inv'].
    #  Could be done in fit() instead I guess, but not of immediate concern now.

    def set_params(self, **params):
        """See Ridge.set_params()"""

        # if "fit_intercept" in params.keys():
        #     if params["fit_intercept"]:
        #         self.__aug_x = True
        #     else:
        #         self.__aug_x = False
        #     # params['fit_intercept'] = False
        if "Q" in params.keys():
            eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(params["Q"]))
            # P = eigvecs.T @ np.diag(np.sqrt(eigvals)) @ eigvecs
            # self.P = P
            P_inv = eigvecs.T @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs
            self.P_inv = P_inv  # np.linalg.pinv(P)
        return super().set_params(**params)

    def fit(self, X, y, *args, **kwargs):
        """Fit the estimator to the training data, see Ridge.fit() for details.

        Parameters
        ----------
        X : np.array
            (N,d) array of training inputs: N points of dimension d. Take care that the
            dimensions of Q and X agree: Q.shape ==  (d,d) (or (d+1, d+1 if fit_intercept
            is True)
        y : np.array
            (N,p) array of training outputs

        Returns
        -------
        self
        """

        # If the intercept is fitted, it gets its own weight and needs
        # to be treated like the other parameters by augmenting the inputs
        if self.fit_intercept:
            X_aug = np.column_stack((np.ones((X.shape[0], 1)), X))
        else:
            X_aug = X

        # Count the number of features after augmentation (would count the intercept, too)
        self._check_n_features(X_aug, reset=True)

        # save fit_intercept state and set it to False for this instance
        _fit_intercept = self.fit_intercept
        self.fit_intercept = False

        # Adjust coef0, it should have shape (1,n_features,) or (n_targets, n_features)
        if y.ndim == 1:
            n_targets = 1
        else:
            n_targets = y.shape[1]
        if np.array(self.coef0).size == 1:
            # coef0 is a scalar of some sort
            # if n_targets==1:
            #     self.coef0 = np.full((self.n_features_in_, ), fill_value=self.coef0)
            # else:
            self.coef0 = np.full(
                (n_targets, self.n_features_in_), fill_value=self.coef0
            )
        elif self.coef0.ndim == 1:
            # if coef0 has more than 1 element, but only one dimension
            self.coef0 = self.coef0[np.newaxis, :]
        # coef0 is now 2-D, (n_targets, n_features), even if n_targets==1
        if y.ndim == 1:
            y = y[:, np.newaxis]
            # else below might broadcast into shape (1,n_datapoints)

        # Y = y - X*beta0
        Y = y - np.dot(X_aug, self.coef0.T)
        # X' = X*P_inv
        X_aug = X_aug @ self.P_inv

        self = super().fit(X_aug, Y, *args, **kwargs)
        # restore fit_intercept state
        self.fit_intercept = _fit_intercept
        # Transform the coef_ so it works with untransformed data
        self.coef_ = self.coef_ @ self.P_inv + self.coef0
        # (
        #     self.coef0 * np.ones(self.coef_.shape)
        #     if np.array(self.coef0).size == 1
        #     else self.coef0.reshape(self.coef_.shape)
        # )
        # Tease the coef_ apart, so that intercept_ and coef_ have the expected values
        # also reset n_features_in_, because the augmentation means that it's too larger
        # (by 1) now
        if self.fit_intercept:
            self.intercept_ = np.atleast_2d(self.coef_)[:, 0]  # .squeeze()
            self.coef_ = np.atleast_2d(self.coef_)[:, 1:]  # .squeeze()
            self._check_n_features(X, reset=True)
        # else we have nothing to do, coef_ is all there is
        return self
