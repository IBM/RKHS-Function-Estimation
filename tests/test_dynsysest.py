import numpy as np
import rkhsid as RKHS
import sklearn.linear_model as lm
import pytest

# Some hard-coded input/output data
u = np.array(
    [
        [-0.57603692],
        [1.72452129],
        [0.26431217],
        [-0.52872561],
        [0.45844168],
        [0.40077381],
        [-0.13478592],
        [-0.9517885],
        [0.64750282],
    ]
)
Tu = u.shape[0]
y = np.array(
    [
        [0.66167347],
        [0.41572727],
        [0.9404332],
        [-0.56655245],
        [0.45959893],
        [-0.10567688],
        [0.19277054],
        [-0.53435945],
        [-0.11847774],
    ]
)
# k_ahead = 2

# Make a very simple estimator, since that isn't being tested here
cont_lin = RKHS.linear_const_eval
sin_kernel = RKHS.SinKernel(max_freq=1)
rkhs_est = RKHS.DynamicalSystemEstimator(
    bias_func_eval=cont_lin, kernel_func_eval=sin_kernel.eval
).fit([y], exogenous_inputs=[u], exo_delays=1, traj_delays=1)
# Also one that fits the increment
rkhs_est_increment = RKHS.DynamicalSystemEstimator(
    bias_func_eval=cont_lin, kernel_func_eval=sin_kernel.eval, target_increment=True
).fit([y], exogenous_inputs=[u], exo_delays=1, traj_delays=1)


# Make a VAR estimator
linreg = lm.LinearRegression()
VAR_est = RKHS.dynsysestimator.DynamicalSystemEstimatorVAR(linear_model=linreg).fit(
    [y], exogenous_inputs=[u], exo_delays=1, traj_delays=1
)

# Collect all the fitted estimators
ESTIMATORS = [rkhs_est, rkhs_est_increment, VAR_est]

# Set values for k_ahead
K_AHEAD_VALS = [1, 2]

# This should work, too, and would avoid having to add @pytest.mark.parametrize to every class, but somehow it fails.
# @pytest.fixture(params=ESTIMATORS)
# def est(request):
#     yield request.param


@pytest.mark.parametrize("est", ESTIMATORS)
class TestTrajectoryLength:
    def test_both_None(self, est):
        with pytest.raises(ValueError):
            est.simulate(init_cond=[y[[0]]], exogenous_inputs=None, traj_len=None)

    def test_traj_len_longer(self, est):
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=20)
        assert (
            y_sim.shape[0] == u.shape[0] + 1
        ), f"Generated traj. should have length len(u)+1, i.e. {Tu+1}"

    def test_traj_len_shorter(self, est):
        traj_len = 4
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=traj_len)
        assert (
            y_sim.shape[0] == traj_len
        ), f"Generated traj. should have length `traj_len`, i.e. {traj_len}"

    def test_traj_len_None(self, est):
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=None)
        assert (
            y_sim.shape[0] == u.shape[0] + 1
        ), f"Generated traj. should have length len(u)+1, i.e. {Tu+1}"


@pytest.mark.parametrize("est", ESTIMATORS)  # , ids=['RKHS','VAR'])
@pytest.mark.parametrize("k_ahead", K_AHEAD_VALS)
class TestKStepAhead:
    def test_no_exo_input(self, k_ahead, est):
        est.fit([y], traj_delays=1)
        y_k = est.k_step_ahead(true_trajectory=[y], k=k_ahead)
        assert y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1

    def test_no_exo_input_missing_val(self, k_ahead, est):
        est.fit([y], traj_delays=1)
        y_nan = y.copy()
        y_nan[-3:, 0] = np.nan
        y_k = est.k_step_ahead(true_trajectory=[y_nan], k=k_ahead)
        assert (y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1) and np.all(
            np.isnan(y_k[-3 + k_ahead :, 0])
        )

    def test_w_exo_input(self, k_ahead, est):
        est.fit([y], exogenous_inputs=[u], traj_delays=1, exo_delays=1)
        y_k = est.k_step_ahead(true_trajectory=[y], exogenous_inputs=[u], k=k_ahead)
        assert y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1

    def test_w_exo_input_missing_val(self, k_ahead, est):
        est.fit([y], exogenous_inputs=[u], traj_delays=1, exo_delays=1)
        y_nan = y.copy()
        y_nan[-3:, 0] = np.nan
        u_nan = u.copy()
        u_nan[2, 0] = np.nan
        y_k = est.k_step_ahead(
            true_trajectory=[y_nan], exogenous_inputs=[u_nan], k=k_ahead
        )
        assert (
            (y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1)
            and np.all(np.isnan(y_k[-3 + k_ahead :, 0]))
            and np.isnan(y_k[2, 0])
        )

    def test_no_exo_input_more_traj(self, k_ahead, est):
        est.fit([y], traj_delays=1)
        y_k = est.k_step_ahead(true_trajectory=[y] * 3, k=k_ahead)
        assert np.all(
            [
                yy_kk.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1
                for yy_kk in y_k
            ]
        )

    def test_w_exo_input_more_traj(self, k_ahead, est):
        est.fit([y], exogenous_inputs=[u], traj_delays=1, exo_delays=1)
        y_k = est.k_step_ahead(
            true_trajectory=[y] * 3, exogenous_inputs=[u] * 3, k=k_ahead
        )
        assert np.all(
            [
                yy_kk.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1
                for yy_kk in y_k
            ]
        )
