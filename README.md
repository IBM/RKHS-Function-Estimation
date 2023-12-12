# Function Estimation and System Identification Using RKHS Methods

## Installation

Just make sure the dependencies are installed and the `rkhsid` folder is on your Python 
path. Easiest way to do that is probably cloning the repo, navigating to the repo folder
and issuing
```bash
pip install -e .
```
`-e` makes the install "editable", so changes in the code are reflected immediately.

## MWE

After the package has been installed, this minimal example should work:
```python
import numpy as np
import matplotlib.pyplot as plt
import rkhsid as RKHS

# function to be estimated
f = lambda x: 1.6*x - x**2 + np.sin(3.2*np.pi*x) 
# generate some data
x_eval = np.linspace(0,1,num=75)
N = 30
X = np.random.rand(N, 1)
y = f(X) + np.random.rand(N,1)/5
# define an estimator with default settings and fit it
est = RKHS.FunctionEstimator()
est.fit(X, y, data_cost_weight=1000)
# plot the estimated function
plt.plot(x_eval, est.eval(x_eval[:, np.newaxis]), label='Estimated function')
plt.plot(X, y, '.', label='Data')
plt.plot(x_eval, f(x_eval), '--', label='True function')
plt.show()
# clearly, the weight on the data wasn't high enough.
```

For more, see the docstrings and the example notebook(s).

## Code Style

We're using [numpydoc docstyle](https://numpydoc.readthedocs.io/en/latest/format.html) 
and the formatter 
[black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html).
