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
# %% parameters
Number_of_task=100  #take 100 tasks for training in total 

number_of_samples_in_support_set=10
number_of_samples_in_query_set=10
total_samples_per_task = number_of_samples_in_support_set+number_of_samples_in_query_set
# Fix random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)


# %% define dataset

class sinusoid(Dataset):
    def __init__(self,total_samples_per_task):
        datasets_list_for_tasks=[] 
        for _ in np.arange(Number_of_task):
            A1s=random.random()
            A2s=random.random()*0.5
            A1c=random.random()
            A2c=random.random()

            # A = 1
            w1 = 1
            w2 = 2
            phi=random.random()*2*np.pi
            t=np.arange(0,10,0.01) # 1000 potential samples in total
            y= A1s*np.sin(w1*t) + A2s*np.sin(w2*t) 
            # y= A1s*(t-np.floor(t)) + A2s*np.log(t+1) 
            data_in_one_task=torch.tensor(np.stack((t,y),axis=-1)).to(torch.float32) #t,y各一列
            datasets_list_for_tasks.append(data_in_one_task)
        self.datalist=datasets_list_for_tasks
        self.total_samples_per_task = total_samples_per_task

        self.obslist=[]
        for i in range(Number_of_task):
            sample = np.arange(1000) 
            np.random.shuffle(sample)  # For random sampling the characters we want.
            self.sample=sample
            xi=self.datalist[i][self.sample[:self.total_samples_per_task],:]
            self.obslist.append(xi)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        x=self.obslist[idx]
        # t=x[:,0]
        # y=x[:,1]
        return x
    
dataset=sinusoid(total_samples_per_task)

data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        drop_last=True,
    )


# ground truth
t=np.arange(0,10,0.01)
grid_points = np.arange(0,10,0.01)
data_matrix_true = [yi[:,1].detach().numpy() for yi in dataset.datalist]

fd_true = skfda.FDataGrid(
    data_matrix=data_matrix_true,
    grid_points=grid_points,
)

# %% PACE cov
t_all=np.array([])
y_all=np.array([])
for task in data_loader:
    t_sample=task[0][:,0].detach().numpy()
    y_sample=task[0][:,1].detach().numpy()
    t_all=np.hstack((t_all,t_sample))
    y_all=np.hstack((y_all,y_sample))

t=np.arange(0,10,0.01)
mean_pace=localreg(t_all, y_all,x0=t, degree=1, kernel=rbf.gaussian, radius=0.2)
t_pace=np.arange(0,10.01,0.2)
mean_pace_for_t_pace=localreg(t_all, y_all,x0=t_pace, degree=1, kernel=rbf.gaussian, radius=0.2)
mean_pace_all=localreg(t_all, y_all, degree=1, kernel=rbf.gaussian, radius=0.2)

input=[]
z=[]
for i,task in enumerate(data_loader):
    t_sample=task[0][:,0].detach().numpy()
    y_sample=task[0][:,1].detach().numpy()
    X, Y = np.meshgrid(t_sample, t_sample)
    input_sample = np.array([X.ravel(), Y.ravel()]).T
    v_sample=np.array([y_sample-mean_pace_all[i*dataset.total_samples_per_task:(i+1)*dataset.total_samples_per_task]])
    z_sample = ((v_sample.T).dot(v_sample)).ravel()
    input.extend(list(input_sample))
    z.extend(list(z_sample))
input=np.array(input)
z=np.array(z)

t_pace=np.arange(0,10.01,0.2)
X0,Y0=np.meshgrid(t_pace,t_pace)
x0=np.array([np.ravel(X0), np.ravel(Y0)]).T

cov_pace = localreg(input[:], z[:], x0, degree=0,radius=0.2, kernel=rbf.gaussian)
cov_pace = cov_pace.reshape(X0.shape)


# true value PCA
fpca_discretized_true = FPCA(n_components=2)
fpca_discretized_true.fit(fd_true)
pc1_true,pc2_true=fpca_discretized_true.components_.data_matrix

# PACE FPCA
lambda_list,pc_list=np.linalg.eig(cov_pace)
pc1_pace=pc_list[:,0]
pc2_pace=pc_list[:,1]

# %% visualize PC1
l5=plt.plot(t,pc1_true,'y',label='pc1_true')
l7=plt.plot(t_pace,pc1_pace*5**0.5,'b:',label='pc1_PACE')
plt.title("PC1")
# plt.legend(bbox_to_anchor=(1.45, 1))

# %% PC2
l6=plt.plot(t,pc2_true,'y',label='pc2_true')
l8=plt.plot(t_pace,pc2_pace*5**0.5,'b:',label='pc2_PACE')
plt.title("PC2")


# %% mean estimation , mean function 刚好和prior接近！！！！！！

y_true=fd_true.mean().data_matrix[0,:,0]
plt.plot(t,y_true,'y',label='Ground truth')
plt.plot(t,mean_pace,'b:',label="PACE")
plt.legend()



# %% covariance estimation

fontsize=15
fig = plt.figure(figsize=(18, 6), facecolor='w')
t=np.arange(0,10,0.01)
X, Y = np.meshgrid(t, t)


c2=np.cov(fd_true.data_matrix[:,:,0].T)
Z2 = c2
ax2 = fig.add_subplot(1, 2, 1, projection='3d')
ax2.plot_surface(X,Y,Z2,alpha=0.2,cmap='winter')
ax2.contour(X,Y,Z2,zdir='z', offset=-0.1,cmap="rainbow")
ax2.contour(X,Y,Z2,zdir='x', offset=0,cmap="rainbow")  
ax2.contour(X,Y,Z2,zdir='y', offset=10,cmap="rainbow")
ax2.set_title("Ground truth",fontsize=fontsize)
ax2.set_zlim(-0.1,0.1)

c4=cov_pace
Z4 = c4
ax4 = fig.add_subplot(1, 2, 2, projection='3d')
ax4.plot_surface(X0,Y0,Z4,alpha=0.2,cmap='winter')
ax4.contour(X0,Y0,Z4,zdir='z', offset=-0.1,cmap="rainbow")
ax4.contour(X0,Y0,Z4,zdir='x', offset=0,cmap="rainbow")  
ax4.contour(X0,Y0,Z4,zdir='y', offset=10,cmap="rainbow")
ax4.set_title("PACE",fontsize=fontsize)
ax4.set_zlim(-0.1,0.1)

# %% PACE recovery
id=18

t_sample=t_all[id*dataset.total_samples_per_task:(id+1)*dataset.total_samples_per_task]
y_sample=y_all[id*dataset.total_samples_per_task:(id+1)*dataset.total_samples_per_task]


f1=interpolate.interp1d(t_pace,pc1_pace)
phi_1=f1(t_sample)
f2=interpolate.interp1d(t_pace,pc2_pace)
phi_2=f2(t_sample)

# plt.plot(t_sample,phi_1,'o')
# plt.plot(t_pace,pc1_pace)
# plt.plot(t_sample,phi_2,'o')

X0_sample,Y0_sample=np.meshgrid(t_sample,t_sample)
x0_sample=np.array([np.ravel(X0_sample), np.ravel(Y0_sample)]).T
cov_pace_id = localreg(input[:], z[:], x0_sample, degree=0,radius=0.2, kernel=rbf.gaussian)
cov_pace_id = np.matrix(cov_pace_id.reshape(X0_sample.shape))+np.matrix(0.00001*np.eye(X0_sample.shape[0])) #正则化 防止不可逆
cov_pace_id_inv=np.linalg.inv(cov_pace_id)
# cov_pace_id_inv=np.array(cov_pace_id_inv)

mu_id=mean_pace_all[id*dataset.total_samples_per_task:(id+1)*dataset.total_samples_per_task]

a_1=np.array(lambda_list[0]*np.matrix(phi_1)*cov_pace_id_inv*(np.matrix(y_sample-mu_id).T))[0][0]
a_2=np.array(lambda_list[1]*np.matrix(phi_2)*cov_pace_id_inv*(np.matrix(y_sample-mu_id).T))[0][0]

y_pace=mean_pace_for_t_pace+a_1*pc1_pace+a_2*pc2_pace

#  Recovery

t=np.arange(0,10,0.01)
plt.plot(t_sample,y_sample,'^',color='olive',markersize=10,label='Observations')
plt.plot(t,fd_true.data_matrix[id][:,0],'y',label='Ground truth')
plt.plot(t_pace,y_pace,'b:',label='PACE')
plt.legend()

# %%
