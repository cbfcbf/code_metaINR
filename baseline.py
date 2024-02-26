# %%
import matplotlib.pyplot as plt
import random
import numpy as np

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

for _ in np.arange(100):

    A1s=random.random()
    A2s=random.random()*0.5
    # A = 1
    w1 = 1
    w2 = 2
    phi=random.random()*2*np.pi
    t=np.arange(0,10,0.01) # 1000 potential samples in total
    y= A1s*np.sin(w1*t) + A2s*np.sin(w2*t) 

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
