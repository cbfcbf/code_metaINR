# %%
from timeseries_INR import *
import data_loader as dl
import matplotlib.pyplot as plt
import modules

BATCH_SIZE = 10000
EPOCHS = 1000

#Siren parameters
HIDDEN_DIM = 256
FIRST_OMEGA_0 = 0.1 #与图像的频率越契合则越好学
HIDDEN_LAYERS = 3

w=10
dataset=dl.synthetic_data3(w)

model = modules.Siren(
    in_features=1,
    out_features=1,
    hidden_features=HIDDEN_DIM,
    hidden_layers=HIDDEN_LAYERS,
    first_omega_0=FIRST_OMEGA_0,
    outermost_linear=True,
)

# model=modules.Net()

tINR=TINR(dataset,model)
tINR.Representation_Learning(
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)


# %%
t,y,loss=tINR.visualize()
# %%
plt.plot(t,y)

# %%
