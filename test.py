# %%
import matplotlib.pyplot as plt

import skfda
from skfda.datasets import fetch_growth
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)
# %%
dataset=skfda.datasets.fetch_cran("pbc","survival")
# %%
dataset
# %%
dataset = skfda.datasets.fetch_growth()
fd = dataset['data']
y = dataset['target']
fd.plot()
# %%
fd
# %%
y
# %%
grid_points = [0, 0.2, 0.5, 0.9, 1]  # Grid points of the curves
data_matrix = [
    [0, 0.2, 0.5, 0.9, 1],     # First observation
    [0, 0.04, 0.25, 0.81, 1],  # Second observation
]

fd = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
)

fd.plot()
plt.show()
# %%
fd.cov()
# %%
import numpy as np
import matplotlib.pyplot as plt
from localreg import *

np.random.seed(1234)
x = np.linspace(1.5, 5, 2000)
yf = np.sin(x*x)
y = yf + 0.5*np.random.randn(*x.shape)

y0 = localreg(x, y, degree=0, kernel=rbf.tricube, radius=0.3)
y1 = localreg(x, y, degree=1, kernel=rbf.tricube, radius=0.3)
y2 = localreg(x, y, degree=2, kernel=rbf.tricube, radius=0.3)

plt.plot(x, y, '+', markersize=0.6, color='gray')
plt.plot(x, yf, label='Ground truth ($\sin(x^2)$)')
plt.plot(x, y0, label='Moving average')
plt.plot(x, y1, label='Local linear regression')
plt.plot(x, y2, label='Local quadratic regression')
plt.legend()
# %%
