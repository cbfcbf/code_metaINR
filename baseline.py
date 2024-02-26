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
