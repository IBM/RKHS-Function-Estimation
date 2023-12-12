import numpy as np
import rkhsid.scaler as scaler
import sklearn.linear_model as lm
import pytest


xrange = (0, 1)
x_unscaled = np.vstack(
    (
        np.linspace(*xrange, num=10),
        np.random.permutation(np.linspace(*xrange, num=10)),
    )
).T

# there are several ways to specify the interval, they all should work.
INTERVAL_SPECS = [None, (-3, 3), (None, None), (None, (-3.5, 2.5)), ((-2, 4), (-1, 5))]
INTERVAL_RANGE = (
    6  # make sure all custom intervals up there have this range, for convenience
)
INTERVAL_SPEC_IDs = ["None", "Single 2elem", "Both None", "None and 2elem", "2 2elem"]

MARGINS = [0, 0.2, -0.1]


@pytest.mark.parametrize(
    "num,interval", enumerate(INTERVAL_SPECS), ids=INTERVAL_SPEC_IDs
)
@pytest.mark.parametrize("margin", MARGINS)
def test_IntervalScaler(interval, margin, num):
    scler = scaler.IntervalScaler(minmax=interval, margin=margin)
    scler.fit(x_unscaled)
    x_scaled = scler.transform(x_unscaled)
    x_scaled_unscaled = scler.inverse_transform(x_scaled)
    # did it scale to the right range?
    if num == 1:
        interval = [interval, interval]
    for dim in (0, 1):
        np.testing.assert_almost_equal(
            x_scaled[:, dim].min(),
            xrange[0]
            if (interval is None or interval[dim] is None)
            else interval[dim][0] + margin * INTERVAL_RANGE,
        )
        np.testing.assert_almost_equal(
            x_scaled[:, dim].max(),
            xrange[1]
            if (interval is None or interval[dim] is None)
            else interval[dim][1] - margin * INTERVAL_RANGE,
        )
    # and is the inverse of the transform the original data again?
    np.testing.assert_array_almost_equal(
        x_unscaled,
        x_scaled_unscaled,
        err_msg="The inverting the transform did not yield the original data",
    )
