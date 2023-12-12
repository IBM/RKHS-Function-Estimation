import numpy as np
import rkhsid as RKHS
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
import pytest

# Some hard-coded input/output data
u = np.hstack(
    (
        np.array(
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
        ),
    )
    * 3
) ** np.array(
    [1, 2, 3]
)  # to break the linear dependence...
Tu = u.shape[0]
y = np.hstack(
    (
        np.array(
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
        ),
    )
    * 2
) ** np.array([1, 2])
nu = 3
ny = 2

# Needed for esimator construction
cont_lin = RKHS.linear_const_eval

ESTIMATOR_FACTORY = {
    "RKHS": lambda **kwargs: RKHS.DynamicalSystemEstimator(
        bias_func_eval=cont_lin,
        kernel_func_eval=RKHS.SinKernel(max_freq=1).eval,
        **kwargs,
    ),
    "RKHS_incr": lambda **kwargs: RKHS.DynamicalSystemEstimator(
        bias_func_eval=cont_lin,
        kernel_func_eval=RKHS.SinKernel(max_freq=1).eval,
        target_increment=True,
        **kwargs,
    ),
    # 'VAR': lambda **kwargs: RKHS.dynsysestimator.DynamicalSystemEstimatorVAR(
    #     linear_model=linreg, **kwargs
    #                 )
}


OUTPUT_SCALES = {
    "None": None,
    "Interval": [-1, 1],
    "Interval_None": [[-1, 1], None],
    "Intervals": [[-1, 1], [-0.9, 0.9]],
    "StandardScaler": StandardScaler(),
}
EXO_SCALES = {
    "None": None,
    "Interval": [-1, 1],
    "Intervals_None": [[-1, 1], None, [-0.9, 0.9]],
    "Intervals": [[-1, 1], [-0.9, 0.9], [-1, 0.9]],
    "StandardScaler": StandardScaler(),
}

MARGINS = [0, 0.1, -0.2]


def assemble_estimator(
    est_name, output_scale, exo_scale, margin, y=[y], u=[u], exo_delays=1, traj_delays=1
):
    return ESTIMATOR_FACTORY[est_name](
        y_scaling=OUTPUT_SCALES[output_scale],
        y_scale_margin=margin,
        exo_scaling=EXO_SCALES[exo_scale],
    ).fit(
        y,
        exogenous_inputs=u,
        exo_delays=exo_delays,
        traj_delays=traj_delays,
    )


# Set values for k_ahead
K_AHEAD_VALS = [1, 2]

# This should work, too, and would avoid having to add @pytest.mark.parametrize to every class, but somehow it fails.
# @pytest.fixture(params=ESTIMATORS)
# def est(request):
#     yield request.param


@pytest.mark.parametrize("est_type", ESTIMATOR_FACTORY)
@pytest.mark.parametrize("y_scale", OUTPUT_SCALES)
@pytest.mark.parametrize("exo_scale", EXO_SCALES)
@pytest.mark.parametrize("margin", MARGINS)
class TestTrajectoryLength:
    def test_both_None(self, est_type, y_scale, exo_scale, margin):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
        with pytest.raises(ValueError):
            est.simulate(init_cond=[y[[0]]], exogenous_inputs=None, traj_len=None)

    def test_traj_len_longer(self, est_type, y_scale, exo_scale, margin):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=20)
        assert (
            y_sim.shape[0] == u.shape[0] + 1
        ), f"Generated traj. should have length len(u)+1, i.e. {Tu+1}"

    def test_traj_len_shorter(self, est_type, y_scale, exo_scale, margin):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
        traj_len = 4
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=traj_len)
        assert (
            y_sim.shape[0] == traj_len
        ), f"Generated traj. should have length `traj_len`, i.e. {traj_len}"

    def test_traj_len_None(self, est_type, y_scale, exo_scale, margin):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
        y_sim = est.simulate(init_cond=y[[0]], exogenous_inputs=[u], traj_len=None)
        assert (
            y_sim.shape[0] == u.shape[0] + 1
        ), f"Generated traj. should have length len(u)+1, i.e. {Tu+1}"


@pytest.mark.parametrize("est_type", ESTIMATOR_FACTORY)
@pytest.mark.parametrize("y_scale", OUTPUT_SCALES)
@pytest.mark.parametrize("exo_scale", EXO_SCALES)
@pytest.mark.parametrize("margin", MARGINS)
@pytest.mark.parametrize("k_ahead", K_AHEAD_VALS)
class TestKStepAhead:
    def test_no_exo_input(self, est_type, y_scale, exo_scale, margin, k_ahead):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin, u=None)
        y_k = est.k_step_ahead(true_trajectory=[y], k=k_ahead)
        assert y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1

    def test_no_exo_input_missing_val(
        self, est_type, y_scale, exo_scale, margin, k_ahead
    ):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin, u=None)
        y_nan = y.copy()
        y_nan[-3:, 0] = np.nan
        y_k = est.k_step_ahead(true_trajectory=[y_nan], k=k_ahead)
        assert (y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1) and np.all(
            np.isnan(y_k[-3 + k_ahead :, 0])
        )

    def test_w_exo_input(self, est_type, y_scale, exo_scale, margin, k_ahead):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
        y_k = est.k_step_ahead(true_trajectory=[y], exogenous_inputs=[u], k=k_ahead)
        assert y_k.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1

    def test_w_exo_input_missing_val(
        self, est_type, y_scale, exo_scale, margin, k_ahead
    ):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin)
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

    def test_no_exo_input_more_traj(
        self, est_type, y_scale, exo_scale, margin, k_ahead
    ):
        est = assemble_estimator(est_type, y_scale, exo_scale, margin, u=None)
        y_k = est.k_step_ahead(true_trajectory=[y] * 3, k=k_ahead)
        assert np.all(
            [
                yy_kk.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1
                for yy_kk in y_k
            ]
        )

    def test_w_exo_input_more_traj(self, est_type, y_scale, exo_scale, margin, k_ahead):
        est = assemble_estimator(
            est_type,
            y_scale,
            exo_scale,
            margin,
        )
        y_k = est.k_step_ahead(
            true_trajectory=[y] * 3, exogenous_inputs=[u] * 3, k=k_ahead
        )
        assert np.all(
            [
                yy_kk.shape[0] == y.shape[0] - est.max_delays - k_ahead + 1
                for yy_kk in y_k
            ]
        )


#
