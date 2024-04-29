# %% packages
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import skfda
from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from localreg import *

from scipy import interpolate

from metaINR_utils import *
from NeuralLaplace_utils import *
# %% parameters
Number_of_task=100  #take 100 tasks for training in total 
total_samples_per_task = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# %% define dataset

def sine(trajectories_to_sample=Number_of_task, t_nsamples=total_samples_per_task): #100个sample 20个observation
    ti=torch.linspace(0, 10, t_nsamples).to(device)
    t=torch.arange(0,10,0.01) # 1000 potential obs in total
    trajs = []
    trajs_truth=[]
    for i in range(trajectories_to_sample):
        A1s=random.random()
        A2s=random.random()*0.5
        w1 = 1
        w2 = 2
        ground_truth= A1s*np.sin(w1*t) + A2s*np.sin(w2*t) 
        yi= A1s*np.sin(w1*ti) + A2s*np.sin(w2*ti) 
        trajs.append(yi)
        trajs_truth.append(ground_truth)
    y = torch.stack(trajs)
    y_true=torch.stack(trajs_truth)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    trajectories_true = y_true.view(trajectories_to_sample, -1, 1)

    return trajectories, ti, trajectories_true

# def sawtooth(trajectories_to_sample=Number_of_task, t_nsamples=total_samples_per_task):
#     # Toy sawtooth waveform. Simple to generate, for Differential Equation Datasets see datasets.py (Note more complex DE take time to sample from, in some cases minutes).
#     ti=torch.linspace(0, 10, t_nsamples).to(device)
#     t=torch.arange(0,10,0.01) # 1000 potential obs in total

#     def sampler(t, x0=0):
#         return (t + x0) / (2 * torch.pi) - torch.floor((t + x0) / (2 * torch.pi))

#     x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
#     trajs = []
#     trajs_truth=[]

#     for x0 in x0s:
#         trajs.append(sampler(ti, x0))
#         trajs_truth.append(sampler(t, x0))
#     y = torch.stack(trajs)
#     y_true=torch.stack(trajs_truth)
#     trajectories = y.view(trajectories_to_sample, -1, 1)
#     trajectories_true = y_true.view(trajectories_to_sample, -1, 1)
#     return trajectories, ti, trajectories_true

Data_class=sine
trajectories, ti ,trajectories_true = Data_class()

class Data_Class(Dataset):
    def __init__(self,trajectories, ti, trajectories_true):
        self.trajectories=trajectories
        self.ti=ti
        t=torch.arange(0,10,0.01) # 1000 potential obs in total
        self.datalist=[torch.stack((t,trajectories_true[i,:,0]),axis=-1).to(torch.float32) for i in range(trajectories_true.shape[0])]
        self.obslist=[torch.stack((ti,trajectories[i,:,0]),axis=-1).to(torch.float32) for i in range(trajectories.shape[0])]
        self.total_samples_per_task=self.trajectories.shape[1]
        self.trajectories_true=trajectories_true
    def __len__(self):
        return self.trajectories.shape[0]

    def __getitem__(self, idx):
        sample = np.arange(self.trajectories.shape[1])
        # sample=np.arange(0,1000,50)
        np.random.shuffle(sample)  # For random sampling the characters we want.
        x=self.obslist[idx][sample,:]
        # t=x[:,0]
        # y=x[:,1]

        return x

dataset=Data_Class(trajectories, ti,trajectories_true)


# %% MetaINR #参数到utils里调整
meta_weights=MetaINR_training(dataset,device)

# %% NeuralLaplace
best_laplace_rep_func,best_encoder=train_NL(trajectories, ti )

# %% Test data
def sine_test(t):
    y=0.6*torch.sin(t)+0.6*torch.sin(2*t)
    return y

def sawtooth_test(t):
    x0=-1
    y= (t + x0) / (2 * torch.pi) - torch.floor((t + x0) / (2 * torch.pi))
    return y

Test_fun=sine_test
t=torch.arange(0,10,0.01)
y=Test_fun(t)



# support_t=torch.linspace(0,4.7368,total_samples_per_task//2)

# support_t=5 * torch.rand(10).sort()[0]

# support_t=torch.linspace(0,5,total_samples_per_task//2-2)

support_y=Test_fun(support_t)

query_t=torch.arange(5,10,0.1)
query_y_true=Test_fun(query_t)

y_true=query_y_true.detach().numpy()
t_plot = query_t.detach().numpy()

#  metaINR
test_inner_train_step=50
meta_model, optimizer, loss_fn = model_init(device, meta_lr,first_omega_0)
fast_weights, inner_loss=inner_train(support_t,support_y,meta_weights,loss_fn, inner_lr,meta_model,inner_step = test_inner_train_step)
y_metaINR=meta_model.functional_forward(query_t.reshape(-1,1), fast_weights).detach().numpy()

# NL
input_dim=1
output_dim=1
encoder = ReverseGRUEncoder(
    input_dim,
    2,
    64 // 2,
).to(device)
encoder.load_state_dict(best_encoder)

laplace_rep_func = LaplaceRepresentationFunc(
    33, output_dim, 2
).to(device)
laplace_rep_func.load_state_dict(best_laplace_rep_func)

laplace_rep_func.eval(), encoder.eval()

support_t=support_t.reshape((1,-1))
support_y=support_y.reshape((1,-1,1))
query_t = query_t.reshape((1,-1))
p = encoder(support_y,support_t)  
predictions = laplace_reconstruct(
    laplace_rep_func, p, query_t, recon_dim=output_dim
)
# tp_to_predict = torch.squeeze(tp_to_predict).detach().numpy()
y_NL = torch.squeeze(predictions).detach().numpy()


# 
plt.plot(t_plot,y_NL,"b:",label="Neural Laplace")
plt.plot(t_plot,y_metaINR,'r:',label="MetaINR")
plt.plot(t_plot,y_true,'y',label="ground truth") 
plt.legend()
plt.savefig('more_t.pdf')

# %%

