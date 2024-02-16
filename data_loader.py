# %%
import numpy as np
import matplotlib.pyplot as plt

# %%

# %%
class output:
    def __init__(
        self,
        entity_name,
        t,
        y):  
        self.entity_name = entity_name
        self.t = t
        self.y = y
        self.x = np.stack((t,y),axis=-1)
        # self.y_dim = y.shape[0]


def synthetic_data(w=1):

    t=np.arange(0,10,0.01)
    y=np.sin(w*t)+t
    out=output(
        "synthetic",t,y,
    )
    return out

def synthetic_data2(w=1):

    t=np.arange(0,10,0.01)
    y=np.sin(w*t)
    out=output(
        "synthetic",t,y,
    )
    return out

def synthetic_data3(w=1):

    t=np.arange(0,10,0.01)
    y=np.sin(w*t)*t
    out=output(
        "synthetic",t,y,
    )
    return out

dataset=synthetic_data()

# %%
