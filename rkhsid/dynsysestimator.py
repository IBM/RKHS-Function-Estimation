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
Application of the function estimation functionalities in the `estimator` module to system
identification.

The class `DynamicalSystemEstimator` and the wraps the functionality of `FunctionEstimator`
to facilitate the estimation of the right-hand side M() in models of the form
    X_t        = M( X_(t-1), X_(t-2), ..., U_(t-1), U_(t-2), ...)
or
    X_t - X_(t-1)  = M( X_(t-1), X_(t-2), ..., U_(t-1), U_(t-2), ...)
and provides methods to use and evaluate estimate.

The class `DynamicalSystemEstimatorVAR` provides the same API but wraps around the 
`linear_model` class from scikit-learn and hence allows for the estimation of linear
right-hand sides, leading to what's known as Vector-AutoRegressive (VAR) models, hence
the name.

@Authors:   Conor Dyson
            Rodrigo Ordonez-Hurtado <rodrigo.ordonez.hurtado@ibm.com>
            Jonathan Epperlein <jpepperlein@ie.ibm.com>
"""

import warnings
from .estimator import FunctionEstimator
from .scaler import IntervalScaler, DummyScaler, ScalerWrapper, _check_is_scaler

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
class DynamicalSystemEstimator(FunctionEstimator):
    """Function estimator class responsible for the training and evaluation of a model of the form
            X_t        = M( X_(t-1), X_(t-2), ..., U_(t-1), U_(t-2), ...)
        for `target_increment==False` or
            X_t - X_(t-1)  = M( X_(t-1), X_(t-2), ..., U_(t-1), U_(t-2), ...)
        for `target_increment==True`.

    Attributes
    ----------
    new_ss : numpy array
        Array of shape (len_train_trajectory, k) where k is the dimension of the exogenous inputs plus number of delays
        times dimension of output space, and the construction of the new state space observations is defined by the
        training trajectory, number of delays, and exogenous inputs.

    Methods
    -------
    fit : Creates the state space given a set of training trajectories, delays under which the next measurement relies
        on, and some set of user provided exogenous inputs which are optional. It then takes this newly constructed
        state space and fits the model to it.

    simulate : Takes an initial condition, and approximates the next step in the trajectory from previous steps and
        exogenous inputs.

    k_step_ahead : Takes in a trajectory, and tests how the model deals with continually making predictions k steps
        ahead.
    """

    def __init__(
        self,
        bias_func_eval=None,
        kernel_func_eval=None,
        target_increment=False,
        y_scaling=None,
        y_scale_margin=0,
        exo_scaling=None,
        exo_scale_margin=0,
    ):
        """
        Parameters
        ----------
        bias_func_eval : Callable, optional, default=None
            Computes the individual basis vectors evaluated on the datapoints in x_. It should read in a numpy array
            of shape (N, n), with N datapoints of n dimensions, and return an array of shape (N,b), with b the number
            of basis functions. By default None, i.e., there is no bias space. NOTE: Inputs ALWAYS must be 2-D, so if
            your datapoints are 1-D, then make sure that X.shape == (N, 1) and NOT (N,). If you only have a single
            datapoint, then make sure that X.shape == (1, n), it should then return a numpy array of shape (N, m), where
            N is the number of datapoints and m is the number of bias space basis functions.

        kernel_func_eval : Callable, optional, default=None (which results in the use of SinKernel().eval)
            Computes the kernel Gram matrix for all combinations of input datapoints. Must take in two numpy arrays
            x_, z_ of shapes (N, n) and (M, n) respectively, and return an (N, M) array with (i, j) element equal to
            K(x_[i, :], z_[j, :]). By default None, which results in the use of SinKernel().eval. NOTE: Inputs ALWAYS
            must be 2-D as with bias_func_eval. Must return an np.array of shape (N, M), where each element (i, j) is
            K(x_[i, :], z_[j, :]), with K being the kernel for the given RKHS.

        target_increment : bool, optional, default=False
            If True, the right-hand side function M() is estimated based on the increments
            y[t] - y[t-1]:
                y_t - y_(t-1)  = M( y_(t-1), y_(t-2), ..., U_(t-1), U_(t-2), ...)
            If False, the output y is predicted directly
               y_t             = M( y_(t-1), y_(t-2), ..., U_(t-1), U_(t-2), ...)
            This mostly affects the behavior of the `eval()` method. `simulate()` and
            `k_step_ahead()` work as expected in either case; see their docstrings.

        y_scaling : 2-elem iterable, iterable of 2-elem iterables, or object with fit() & transform() methods, optional
            Scaling for the outputs. Can be one of the following:
            - None (then no scaling is applied).
            - A single interval [min, max] (then all dimensions are scaled to [min, max]).
            - An iterable containing [min_i, max_i] and None. Its length must match the output dimension. Each element
                then specifies the interval the corresponding dimension is scaled to.
            - A scaler object, i.e. one with a fit() and transform() method, working like  the ones available in
                sklearn.preprocessing (e.g., MinMaxScaler).

        y_scale_margin :  0 <= float < 0.5, optional, default=0
            If intervals are specified in y_scaling, the scaling is computed so the scaled data is
            y_scale_margin * (max_i-min_i) away from min_i and max_i.

        exo_scaling, exo_scale_margin : same as y_scaling and y_scale_margin, optional
            Same as above, just for exogenous inputs, by default None and 0, respectively.
        """

        # Storing the scaling parameters, but the details depend on the delays, which in turn are only specified in
        # fit(). So, we must re-initialize the scaling there.
        if y_scaling is None and exo_scaling is None:
            self._scaling = None  # So, no scaling at all.
        else:
            self._scaling = (y_scaling, exo_scaling)
            if all(
                [
                    y_scaling is not None,
                    exo_scaling is not None,
                    y_scale_margin != exo_scale_margin,
                ]
            ):
                self._scale_margin = max(y_scale_margin, exo_scale_margin)
                warnings.warn(
                    "Different scale margins are currently not supported. "
                    f"Using the larger one, i.e. {self._scale_margin}"
                )
            else:
                self._scale_margin = (
                    y_scale_margin if y_scaling is not None else exo_scale_margin
                )

        # Initialize with no scaling, the scaling will be dealt with in fit().
        super().__init__(bias_func_eval, kernel_func_eval, scaling=None, scale_margin=0)

        # Convention for self.target:
        # - 'increment' means M is such that y[k+1]-y[k] = M(y[k], ...)
        # - 'direct' means that y[k+1] = M(y[k], ...)
        self.target = "increment" if target_increment else "direct"
        self.target_increment = target_increment

        self.delays = None
        self.new_ss = None
        self.y_train_trajectories = None
        self.model_trajectory = list
        self.traj_delays = None
        self.exo_delays = None
        self.max_delays = None
        self.traj_len = None
        self.kth_step_trajectory = None
        self.best_rkhs_weight = None
        self.best_bias_weight = None

    # %%
    def fit(
        self,
        y_trajectories,
        traj_delays=1,
        exogenous_inputs=None,
        exo_delays=0,
        data_cost_weight=1,
        rkhs_weight=1,
        bias_weight=0,
    ):
        """Firstly constructs the appropriate state space from `y_trajectories`, `exogenous_inputs`, and associated
        delays. Given V_2s is the original Hilbert space defined by the reproducing kernel and H_B is the appended bias
        space, then denote P_1 as the orthogonal projector from V_2s X H_B onto V_2s and P_0 as the orthogonal projector
        from V_2s X B onto H_B. And given y_i = M(D_i) + e, where e is some gaussian noise, then
                M_hat =  argmin_{M_hat} (  r * ( sum_{i=1}^{N} ( y_i - M_hat(D_i) )**2
                                        + q1 * || P_1(M_hat) ||_{V_2s}**2
                                        + q2 * || P_0(M_hat) ||_{H_B}**2      )
        is our estimate for M with the above solution, having an equivalent interpretation as the solution to a
        min-max problem on an ellipse. The function of the fit method is to solve for coef_slices_matrix and
        coef_bias_matrix, such that
                M_hat(D) = kernel_func_eval_matrix @ coef_slices_matrix + bias_func_eval_matrix @ coef_bias_matrix
        satisfies the above argmin equation.

        Parameters
        ----------
        y_trajectories : list of numpy arrays
            List of arrays of shape (Nk, d), with d the of the model output and Nk the length of the k-th trajectory.
            Given y_trajectories, the fit function then constructs the corresponding state space.
            NOTE: Each array MUST be 2-D, so if your outputs are 1-D, then make sure that shape == (Nk, 1) and NOT (Nk,).
            If you only have a single time step, then make sure that shape == (1, d).
            For a 1-D array, d==1 is assumed; however you should not rely on this and explicitly ensuring 2-D arrays is
            advised.

        traj_delays : float, optional
            Defines how many previous delays the next observation in the y_trajectories array depends on.

        exogenous_inputs : list of numpy arrays or None, default=None
            List of arrays of shape (Nk, m), with m the dimension of the exogenous input and Nk the length of the
            k-th trajectory.
            NOTE: Each array MUST be 2-D, so if your inputs are 1-D, then make sure that shape == (Nk, 1) and NOT (Nk,).
            If you only have a single time step, then make sure that shape == (1, m).

        exo_delays : int, optional
            Defines how many previous exogenous inputs the next observation in the y_trajectories array depends on.

        data_cost_weight : float, optional
            Relative weight on the data fit term in the minimization problem.

        rkhs_weight : float, optional
            Relative weight on the Hilbert space norm of the orthogonal projection of M onto the V_2s Hilbert space.

        bias_weight : float, optional
            Relative weight on the bias space norm of the orthogonal projection of M onto the bias space.

        Returns
        -------
        self :
            The attributes `self.coef_slices_matrix` (array of shape (N, d), with N datapoints of dimension d (dimension
            of the model output)) and `self.coef_bias_matrix` (array of shape (N, d), with N datapoints of dimension d
            (dimension of the model output)) are now populated and `eval()` can be used.
        """
        self.traj_delays = traj_delays
        self.exo_delays = exo_delays
        self.max_delays = (
            traj_delays if exogenous_inputs is None else max(traj_delays, exo_delays)
        )

        y_trajectories = self._parse_array(y_trajectories, "y_trajectories")

        self.new_ss = [None] * len(y_trajectories)
        self.y_train_trajectories = [None] * len(y_trajectories)

        if exogenous_inputs is not None:
            if not len(exogenous_inputs) == len(y_trajectories):
                raise ValueError("Provide one array of exogenous inputs per trajectory")
            if self.exo_delays == 0:
                warnings.warn(
                    "[Warning] exogenous_inputs will not be used since"
                    " self.exo_delays=0"
                )
            exogenous_inputs = self._parse_array(exogenous_inputs, "exogenous_inputs")
        else:
            if self.exo_delays > 0:
                warnings.warn(
                    f"[Warning] exo_delays={exo_delays} has no effect since"
                    " exogenous_inputs is None"
                )

        # Loops through the list of trajectories to create the state space for each one individually first.
        for pp, yp in enumerate(y_trajectories):
            if yp.ndim == 1:
                yp = yp[:, cp.newaxis]
            up = None if exogenous_inputs is None else exogenous_inputs[pp]

            if up is None:
                obs_mapped_ss = cp.nan * cp.zeros(
                    (yp.shape[0] - self.traj_delays, yp.shape[1] * self.traj_delays)
                )

                for jj in range(len(yp) - self.traj_delays):
                    obs_mapped_ss[jj, :] = yp[jj : (jj + self.traj_delays), :].flatten()
                # This flattening operation results in the jj-th row of obs_mapped_ss
                #       [ yp[jj, 1] yp[jj, 2] ... yp[jj, d] yp[jj+1, 1] ... yp[jj+traj_delay-1, d] ]
            else:
                if yp.shape[0] != up.shape[0]:
                    raise ValueError(
                        "Provided outputs and exogenous inputs must have same number"
                        f" of timesteps, but in trajectory {pp} we have"
                        f" {yp.shape[0]} != {up.shape[0]}"
                    )

                obs_mapped_ss = self._arrange_output_and_exo_input_into_predictor(
                    yp=yp,
                    up=up,
                    traj_delay=self.traj_delays,
                    exo_delay=self.exo_delays,
                    max_delay=self.max_delays,
                )

            self.new_ss[pp] = obs_mapped_ss
            if self.target_increment:
                self.y_train_trajectories[pp] = cp.diff(
                    yp[(self.max_delays - 1) :, :], axis=0
                )
                # Note the -1. Say max_delays=2, then the first value that can be estimated is:
                #       y[2]-y[1] = M(y[1], y[0], u[1], u[0])
                # We also then have that:
                #       ( cp.diff(yp[(self.max_delays-1):, :], axis=0) ).shape == ( yp[self.max_delays:, :] ).shape
            else:
                self.y_train_trajectories[pp] = yp[self.max_delays :, :]

        # Flattens the list of state spaces from different trajectories into one large statespace and flattens the
        # trajectories into one large space of target values.
        new_ss_flat = cp.vstack(self.new_ss)
        y_train_trajectories_flat = cp.vstack(self.y_train_trajectories)

        self._parse_scaling(yp.shape[1], None if up is None else up.shape[1])
        # FIXME: What about replacing yp.shape[1] with y_trajectories[0].shape[1] in the previous line?

        self._fit(
            X=new_ss_flat,
            y=y_train_trajectories_flat,
            data_cost_weight=data_cost_weight,
            rkhs_weight=rkhs_weight,
            bias_weight=bias_weight,
        )

        return self

    # %%
    def _parse_scaling(self, d_y, d_u):
        """Sets self._scaler to its correct variation."""
        if self._scaling is None:
            self._scaler = None
        else:
            if d_u is None:
                if _check_is_scaler(self._scaling[0]):
                    self._scaler = ScalerWrapper(
                        [self._scaling[0]], n=[d_y], n_repeat=[self.traj_delays]
                    )
                elif hasattr(self._scaling[0], "__len__"):
                    self._scaler = IntervalScaler(
                        self._scaling[0] * self.traj_delays, self._scale_margin
                    )
                else:
                    try:
                        self._scaler = IntervalScaler(
                            self._scaling[0], self._scale_margin
                        )
                    # TODO: handle case where exo_delay is not used, but exo_scaling is set and y_scaling is None?
                    except Exception as e:
                        e.args += (
                            "Error initializing scaling. Check the docstring for the"
                            " scaling argument"
                        )
                        raise e
            else:
                if any(
                    [_check_is_scaler(_scaling) for _scaling in self._scaling]
                ):  # At least one is a scaling object already
                    _scaler = []
                    for _scaling in self._scaling:
                        if _scaling is None:
                            _scaler += [DummyScaler()]
                        elif _check_is_scaler(_scaling):
                            _scaler += [_scaling]
                        else:
                            try:
                                _scaler += [
                                    IntervalScaler(_scaling, margin=self._scale_margin)
                                ]
                            except Exception as e:
                                e.args += (
                                    "Error initializing scaling. Check the docstring"
                                    " for the scaling argument"
                                )
                                raise e
                    self._scaler = ScalerWrapper(
                        _scaler, [d_y, d_u], [self.traj_delays, self.exo_delays]
                    )
                else:  # We can build an IntervalScaler
                    _scaler_interval = []
                    for _scaling, n, nr in zip(
                        self._scaling, (d_y, d_u), (self.traj_delays, self.exo_delays)
                    ):
                        if _scaling is None:
                            _scaler_interval += [None] * (n * nr)
                        elif (_scaling[0] is None) or hasattr(_scaling[0], "__len__"):
                            # _scaling is a sequence of None and/or [min, max] intervals
                            _scaler_interval += _scaling * nr
                        else:
                            # _scaling is just a single interval
                            _scaler_interval += [_scaling] * (n * nr)
                    self._scaler = IntervalScaler(_scaler_interval, self._scale_margin)

    # %%
    def _fit(self, **kwargs):
        """Evaluates the superclass' fit method using the provided **kwargs."""
        super().fit(**kwargs)

    # %%
    @staticmethod
    def _arrange_output_and_exo_input_into_predictor(
        yp, up, traj_delay, exo_delay, max_delay, single_datapoint=False
    ):
        """This rearranging operation results in the jj-th row of obs_mapped_ss
              [ yp[jj+R, 1] yp[jj+R, 2] ... yp[jj+R, d] yp[jj+1+R, 1] ... yp[jj+max_delay-1, d],
                up[jj+Q, 1] up[jj+Q, 2] ... up[jj+Q, l] up[jj+1+Q, 1] ... up[jj+max_delay-1, l] ]
        where R=max_delay-traj_delay, Q=max_delay-exo_delay.

        If `single_datapoint` is False, the last datapoint is not included in the output, since it cannot be used to
        predict anything anymore (as the "next" time point does not exist anymore, `yp` (and `up`) ends).
        If `single_datapoint` is True, `yp` and `up` are rearranged so that they can be used to predict the next
        datapoint, even if it is not in the trajectory.

        Parameters
        ----------
        yp : np.array
            Training trajectories.

        up : np.array
            Exogenous inputs.

        traj_delay : int
            Defines how many previous delays the next observation in the yp array depends on.

        exo_delay : int
            Defines how many previous exogenous inputs the next observation in the yp array depends on.

        max_delay : int
            Maximum value between traj_delay and exo_delay.

        single_datapoint : Boolean, default=False
            To exclude last datapoint (if False) or rearrange `yp` and `up` (if True) in the prediction process.

        Returns
        -------
        obs_mapped_ss : Numpy array
            Resulting array od the rearranging operation so that yp[jj+max_delay, :] = func( obs_mapped_ss[jj, :] ).

        """
        if yp.ndim == 1:
            yp = yp[:, cp.newaxis]
        if up.ndim == 1:
            up = up[:, cp.newaxis]

        if single_datapoint:
            obs_mapped_ss = cp.hstack(
                (
                    yp[(max_delay - traj_delay) :, :].flatten(),
                    up[(max_delay - exo_delay) :, :].flatten(),
                )
            )[cp.newaxis, :]
        else:
            obs_mapped_ss = cp.nan * cp.zeros(
                (
                    len(yp) - max_delay,
                    yp.shape[1] * traj_delay + up.shape[1] * exo_delay,
                )
            )
            for jj in range(len(yp) - max_delay):
                obs_mapped_ss[jj, :] = cp.hstack(
                    (
                        yp[
                            (jj + max_delay - traj_delay) : (jj + max_delay), :
                        ].flatten(),
                        up[
                            (jj + max_delay - exo_delay) : (jj + max_delay), :
                        ].flatten(),
                    )
                )
        return obs_mapped_ss

    # %%
    def simulate(self, init_cond, traj_len=None, exogenous_inputs=None):
        """
        Generate trajectories of length `traj_len` by simulating the estimated system forward in time,
        starting from the initial condition(s) supplied in `init_cond` and, if applicable, using the
        supplied exogenous inputs.

        Parameters
        ----------
        init_cond : list of numpy arrays or single numpy array (one trajectory)
            List of initial conditions with an initial condition array of shape (L, n) where
            L = max(traj_delays, exo_delays) with the (L, n)'th element corresponding to the most recent measurement and
            the (L-1, n)'th corresponding to the first delay, and so forth.

        traj_len : int or None, default=None
            Defines the length of the trajectory the model will estimate. If `None`, the length of the supplied
            exogenous input is used instead. If `traj_len` is specified and `exogenous_inputs` is provided, the
            resulting trajectory `i` will be of length `min(traj_len, len(exogenous_inputs[i]))`.

        exogenous_inputs : list of numpy arrays, optional, default=None
            If the user defined exogenous inputs in the training of the model, then they must again provide a list of
            exogenous input arrays for each trajectory they want to estimate. They must provide a full list of arrays of
            shape (traj_len, k) where k is the dimension of the exogenous inputs.

        Returns
        -------
        model_trajectory : list of numpy arrays
            Each array `model_trajectory[k]` will have shape (traj_len + L, n),
            where L = max(traj_delays, exo_delays) and it corresponds to the simulated
            trajectory with initial conditions `init_cond[k]` and exogenous input (if
            applicable) `exogenous_inputs[k]`.
            If only one init_cond array is provided, then one output trajectory is a
            single numpy array (not a 1-element list).
        """

        if (exogenous_inputs is None) and (traj_len is None):
            raise ValueError(
                "`exogenous_inputs` and `traj_len` cannot both be `None`, specify at"
                " least one of them"
            )
        init_cond = self._parse_array(init_cond, "init_cond")
        self.model_trajectory = [None] * len(init_cond)

        if exogenous_inputs is not None:
            exogenous_inputs = self._parse_array(exogenous_inputs, "exogenous_inputs")
            if self.exo_delays == 0:
                warnings.warn(
                    "[Warning] exogenous_inputs will not be used since"
                    " self.exo_delays=0"
                )

        # Starts at initial conditions and then reads solution into next evaluation to obtain a full trajectory
        for pp, yp in enumerate(init_cond):
            if not yp.shape[0] == self.traj_delays:
                raise ValueError(
                    "Each trajectory's initial condition must have same number of"
                    f" datapoints as system delays = {self.traj_delays}"
                )

            up = None if exogenous_inputs is None else exogenous_inputs[pp]
            traj_len_p = (
                traj_len
                if up is None
                else min(i for i in [traj_len, len(up) + 1] if i is not None)
            )
            # TODO: we could issue a warning here if traj_len and len(up) disagree

            # Pre-allocate for the trajectory
            input_for_pth_traj = cp.nan + cp.empty((traj_len_p, yp.shape[1]))
            input_for_pth_traj[: self.traj_delays] = yp

            for jj in range(traj_len_p - self.traj_delays):
                if up is None:
                    x_ = (
                        input_for_pth_traj[jj : (jj + self.traj_delays)]
                        .flatten()
                        .reshape(1, -1)
                    )
                else:
                    x_ = self._arrange_output_and_exo_input_into_predictor(
                        yp=input_for_pth_traj[jj : (jj + self.max_delays), :],
                        up=up[jj : (jj + self.max_delays), :],
                        traj_delay=self.traj_delays,
                        exo_delay=self.exo_delays,
                        max_delay=self.max_delays,
                        single_datapoint=True,
                    )

                # In next lines, `.flatten()` required as cupy (if available) does not support broadcasting
                if cp.any(cp.isnan(x_)):
                    input_for_pth_traj[self.max_delays + jj, :] = cp.nan
                elif self.target_increment:
                    input_for_pth_traj[self.max_delays + jj, :] = (
                        input_for_pth_traj[self.max_delays + jj - 1, :]
                        + self.eval(x_, return_cupy=True)
                    ).flatten()
                else:
                    input_for_pth_traj[self.max_delays + jj, :] = self.eval(
                        x_, return_cupy=True
                    ).flatten()

            self.model_trajectory[pp] = cp.asnumpy(input_for_pth_traj)

        return (
            self.model_trajectory[0]
            if len(self.model_trajectory) == 1
            else self.model_trajectory
        )

    # %%
    def k_step_ahead(self, true_trajectory, exogenous_inputs=None, k=1):
        """Convenience function to compute a k-step ahead prediction based on a given observed trajectory.

        Parameters
        ----------
        true_trajectory : np.array
            Array of shape (N, d), with N outputs of dimension d (dimension of the model output). It's a true trajectory
            the user wishes to test their model against. NOTE: ALWAYS must be 2-D, so if your outputs are 1-D, then make
            sure that X.shape == (N, 1) and NOT (N,). If you only have a single output, then make sure that
            y.shape == (1, d).

        exogenous_inputs : list of numpy arrays, optional, default=None
            If the user defined exogenous inputs in the training of the model, then they must again provide a list of
            exogenous input arrays for each trajectory they want to estimate. They must provide a full list of arrays of
            shape (traj_len, k) where k is the dimension of the exogenous inputs. (As of now, user must provide all
            exogenous inputs at the beginning of construction of the estimated trajectory).

        k : int, default=1
            Determines the number of steps ahead the function will predict. NOTE: the function saves the predictions
            into an array; however, it always uses a true set of initial conditions when calculating each k'th step
            ahead prediction.

        Returns
        -------
        k_step_trajectory : list of numpy arrays
            Each array filled with k'th step ahead prediction of each provided true value after and including the
            (self.max_delays+k)'th true element
        """
        true_trajectory = self._parse_array(true_trajectory, "true_trajectory")

        if exogenous_inputs is not None:
            exogenous_inputs = self._parse_array(exogenous_inputs, "exogenous_inputs")
            if self.exo_delays == 0:
                warnings.warn(
                    "[Warning] exogenous_inputs will not be used since"
                    " self.exo_delays=0"
                )

        k_step_trajectory = [None] * len(true_trajectory)
        margin_steps = self.max_delays + k - 1

        for pp, yp in enumerate(true_trajectory):
            up = None if exogenous_inputs is None else exogenous_inputs[pp]

            len_p, d = yp.shape
            k_step_traj_pp = cp.asnumpy(
                cp.nan + cp.empty((len_p - margin_steps, d), dtype=float)
            )

            for tt in range(k_step_traj_pp.shape[0]):
                init_cond = [yp[tt : (tt + self.max_delays), :]]
                exo_inputs = None if up is None else [up[tt : (tt + margin_steps), :]]
                next_k_steps = self.simulate(
                    init_cond=init_cond,
                    exogenous_inputs=exo_inputs,
                    traj_len=(margin_steps + 1),
                )

                k_step_traj_pp[tt, :] = next_k_steps[-1, :]

            k_step_trajectory[pp] = k_step_traj_pp

        return (
            k_step_trajectory[0] if len(k_step_trajectory) == 1 else k_step_trajectory
        )


# %%
class DynamicalSystemEstimatorVAR(DynamicalSystemEstimator):
    """A dynamical system estimator estimating a Vector Auto-Regressive model of the form
            X_t           = A_1 X_(t-1) + A_2 X_(t-2) + ... + B_1 U_(t-1) + ...
    for `target_increment==False`, or
            X_t - X_(t-1) = A_1 X_(t-1) + A_2 X_(t-2) + ... + B_1 U_(t-1) + ...
    for `target_increment==True`, where X and U can be vectors, and A_i and B_i are matrices.
    """

    def __init__(self, linear_model, target_increment=False, **kwargs):
        """See DynamicalSystemEstimator docstring for all parameters, except the additional one described below.
        NOTE: Scaling parameters, if provided, are ignored.

        Parameters
        ----------
        linear_model : sklearn.linear_model._base.LinearModel
            A linear model object as the ones implemented in sklearn.linear_model. It needs to have at least `fit()` and
            `predict()` as well as `coef_` and `intercept_` properties.
            NOTE: The passed `LinearModel` object will be mutated, so it is not a good idea to reuse it. Example:
                    DO NOT:
                        linreg = lm.LinearRegression()
                        VAR1 = DynamicalSystemEstimatorVAR(linear_model=linreg)
                        VAR2 = DynamicalSystemEstimatorVAR(linear_model=linreg)
                    INSTEAD:
                        linreg1 = lm.LinearRegression()
                        linreg2 = lm.LinearRegression()
                        VAR1 = DynamicalSystemEstimatorVAR(linear_model=linreg1)
                        VAR2 = DynamicalSystemEstimatorVAR(linear_model=linreg2)
                    OR:
                        from sklearn.base import clone
                        linreg = lm.LinearRegression()
                        VAR1 = DynamicalSystemEstimatorVAR(linear_model=clone(linreg))
                        VAR2 = DynamicalSystemEstimatorVAR(linear_model=clone(linreg))
        """
        super().__init__(
            target_increment=target_increment,
            bias_func_eval=lambda x: None,
            kernel_func_eval=lambda x: None,
        )
        # Passing dummy kernel and bias space functions, as the parent classes expect them.
        # Reminder: these will be set by the init call.

        self._linmod = linear_model

    # %%
    def _fit(self, **kwargs):
        """Evaluates the fit method of the linear model using the provided **kwargs."""

        self._linmod.fit(X=cp.asnumpy(kwargs["X"]), y=cp.asnumpy(kwargs["y"]))

    # %%
    def eval(self, x_, return_cupy=False):
        """
        Evaluates the rhs at the datapoint `x_`, which consists of previous input and output samples according to the
        delays.

        NOTE: `x_` always has to have 2 dimensions (N, D), even if N==1 (one data point) or D==1 (1-D outputs).

        Typically, you probably want to use `simulate()` for predicting future outputs.
        """

        if return_cupy:
            return cp.array(self._linmod.predict(cp.asnumpy(x_)))

        return self._linmod.predict(cp.asnumpy(x_))

    # %%
    @property
    def coef_(self):
        """Returns the parameters of the identified linear system
            y[t] =  A_1 y[t-1] + A_2 y[t-2] + ... A_{traj_delays} y[t-traj_delays]
                    + B_1 u[t-1] + ... B_{exo_delays} u[t-exo_delays]
                    + C
        in a more intuitive/useful form.

        Returns
        -------
        A : np.array
            Array of shape (d, d, traj_delays), so that A[:, :, i] == A_{i+1}.
            Note the slight awkwardness due to 0-based and 1-based indexing.

        B : np.array
            Array of shape (d, k, exo_delays), so that B[:, :, j] == B_{j+1}.

        C : np.array
            Array of shape (d,), if `fit_intercept` was True (the default).
        """
        d = self._linmod.coef_.shape[0]
        return (
            cp.flip(
                self._linmod.coef_[:, : d * self.traj_delays].reshape(
                    (d, d, self.traj_delays), order="F"
                ),
                axis=2,
            ),
            cp.flip(
                self._linmod.coef_[:, d * self.traj_delays :].reshape(
                    (d, -1, self.exo_delays), order="F"
                ),
                axis=2,
            ),
            self._linmod.intercept_,
        )
