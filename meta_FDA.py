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

train_inner_train_step = 1
val_inner_train_step = 10

first_omega_0=3
inner_lr = 0.001
meta_lr = 0.0001

max_epoch = 10000
inner_batch_size=1 #没什么用 每个batch只能对应一个任务

# eval_batches = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# %% model definition

class SineLayer(torch.nn.Module):

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=True,
        first_omega_0=3000,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.hidden_layers=hidden_layers
        self.first_omega_0=first_omega_0
        self.hidden_omega_0=hidden_omega_0
        self.outermost_linear=outermost_linear

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords
    
    def functional_forward(self,x,params):
        x=torch.sin(self.first_omega_0*F.linear(x,weight=params[f'net.0.linear.weight'],bias=params[f"net.0.linear.bias"]))
        for i in np.arange(1,self.hidden_layers+1):
            x=torch.sin(self.hidden_omega_0*F.linear(x,weight=params[f'net.{i}.linear.weight'],bias=params[f"net.{i}.linear.bias"]))
        i=self.hidden_layers+1
        if self.outermost_linear:
            x=F.linear(x,weight=params[f"net.{i}.weight"],bias=params[f"net.{i}.bias"])
        else:
            x=torch.sin(self.hidden_omega_0*F.linear(x,weight=params[f'net.{i}.linear.weight'],bias=params[f"net.{i}.linear.bias"]))
        return x

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

# define dataset

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
            data_in_one_task=torch.tensor(np.stack((t,y),axis=-1)).to(torch.float32) #t,y各一列
            datasets_list_for_tasks.append(data_in_one_task)
        self.datalist=datasets_list_for_tasks
        self.total_samples_per_task = total_samples_per_task

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample = np.arange(1000) 
        np.random.shuffle(sample)  # For random sampling the characters we want.
        x=self.datalist[idx][sample[:self.total_samples_per_task],:]
        t=x[:,0]
        y=x[:,1]
        return x
    
# training function
def Metaepoch(model,optimizer,data_loader,loss_fn,
               inner_train_step = 1, inner_lr=0.1, train=True,visualize=False):
    criterion, task_loss= loss_fn, []
    # task_D_loss=[]
    for train_batch in data_loader: #[10,70,2] 共10个task 
        # for i in range(inner_batch_size): # batch 中的第 i 个 task 共10个task #这个循环可能可以去掉
        # Get data
        i=0
        support_set = train_batch[i][: number_of_samples_in_support_set] #[50,2]
        support_t = support_set[:,0].reshape(-1,1)
        support_y = support_set[:,1].reshape(-1,1)

        query_set = train_batch[i][number_of_samples_in_support_set :]
        query_t = query_set[:,0].reshape(-1,1)
        query_y = query_set[:,1].reshape(-1,1)

        fast_weights = OrderedDict(model.named_parameters())
        
        #inner train
        for inner_step in range(inner_train_step):
            # Simply training
            y_true =  support_y
            y_predict  = model.functional_forward(support_t, fast_weights)
            loss = criterion(y_predict, y_true)
            # Inner gradients update! #
            """ Inner Loop Update """
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
            
        #inner validation
        y_true =  query_y
        y_predict  = model.functional_forward(query_t, fast_weights)
        loss = criterion(y_predict, y_true)
        task_loss.append(loss)




    # 此时一个epoch结束
        
    # 计算微分方程上的损失 D_loss
    # pi_t=torch.arange(0,10,0.1).reshape(-1,1)
    # pi_y,coord=model.forward(pi_t)
    # Dy=gradient(pi_y,coord)
    # DDy=gradient(Dy,coord)
    # D_loss = loss_fn(pi_y,-DDy)

    if train:
        #此时所有task的loss收集结束，考虑优化参数
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_loss).mean()
        meta_batch_loss.backward(retain_graph=True)
        optimizer.step()
        meta_loss= meta_batch_loss.detach().numpy()
    else:
        if visualize:
        # validation set上进行训练后产生的loss(最后一个task)，不进行元模型的进一步优化 
            plt.plot(query_t.detach().numpy(),y_true.detach().numpy(),"r+")
            plt.plot(query_t.detach().numpy(),y_predict.detach().numpy(),"b+")
        meta_loss=torch.stack(task_loss).mean().detach().numpy()


    return meta_loss

def inner_train(t,y, meta_weights, inner_step = val_inner_train_step):
    fast_weights=meta_weights
    for inner_step in range(inner_step):
        # Simply training
        y_true =  y.reshape(-1,1)
        y_predict  = meta_model.functional_forward(t.reshape(-1,1), fast_weights)
        loss = loss_fn(y_predict, y_true)
        # Inner gradients update! #
        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
        fast_weights = OrderedDict(
            (name, param - inner_lr * grad)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )
    inner_loss=loss
    return fast_weights,inner_loss
# %% initialization

# data divide into batches each batch one task
def dataloader_init(data, shuffle=True, num_workers=0, inner_batch_size=inner_batch_size):
    train_set, val_set = data
    train_loader = DataLoader(
        train_set,
        batch_size=inner_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=inner_batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=True
    )
    train_iter = iter(train_loader) #output some samples from #inner_batch_size tasks 一个batch中有来自10组task的随机的sample    
    val_iter = iter(val_loader)
    return (train_loader, val_loader), (train_iter, val_iter)

dataset=sinusoid(total_samples_per_task)

train_split = int(0.8 * len(dataset))
val_split = len(dataset) - train_split
train_set, val_set = torch.utils.data.random_split(dataset, [train_split, val_split])
(train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set))

data_loader = DataLoader(
        dataset,
        batch_size=inner_batch_size,
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

# baseline
t=np.arange(0,10,0.01)
grid_points = np.arange(0,10,0.01)
data_matrix_base=[]
for task in data_loader:
    t_sample=task[0][:,0].detach().numpy()
    y_sample=task[0][:,1].detach().numpy()
    y_base=localreg(t_sample, y_sample,x0=t, degree=1, kernel=rbf.gaussian, radius=1)
    data_matrix_base.append(y_base)

fd_base = skfda.FDataGrid(
    data_matrix=data_matrix_base,
    grid_points=grid_points,
)



# metaINR model init
def model_init():
    meta_model = Siren(in_features=1,hidden_features=40,hidden_layers=3,out_features=1,first_omega_0=first_omega_0).to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = nn.MSELoss().to(device)
    return meta_model, optimizer, loss_fn

meta_model, optimizer, loss_fn = model_init()


# prior before meta training

# t=torch.tensor(np.arange(0,10,0.01)).to(torch.float32).reshape(-1,1)
# y,coord= meta_model.forward(t)
# plt.plot(t.detach().numpy(),y.detach().numpy())

# 单训一次的结果
# val_meta_loss = Metaepoch(meta_model,optimizer,val_loader,loss_fn,inner_train_step=val_inner_train_step,inner_lr=inner_lr,train=False,visualize=True)


# %% training
train_meta_loss_list=[]
val_meta_loss_list=[]
mean_loss_list=[]

for epoch in range(max_epoch):
    train_meta_loss = Metaepoch(meta_model,optimizer,train_loader,loss_fn,inner_train_step=train_inner_train_step,inner_lr=inner_lr,train=True)
    val_meta_loss= Metaepoch(meta_model,optimizer,val_loader,loss_fn,inner_train_step=val_inner_train_step,inner_lr=inner_lr,train=False)
    print("Epoch :" ,"%d" % epoch, end="\t")
    print("Train loss :" ,"%.3f" % train_meta_loss, end="\t")
    print("Validation loss :" ,"%.3f" % val_meta_loss)
    train_meta_loss_list.append(train_meta_loss)
    val_meta_loss_list.append(val_meta_loss)

    #看是否元模型学习了task的均值
    t1= torch.tensor(np.arange(0,10,0.01)).to(torch.float32).reshape(-1,1)
    y,coord = meta_model.forward(t1)
    y=y.detach().numpy()[:,0]
    y_true_mean=fd_true.mean().data_matrix[0,:,0]
    mean_loss = ((y-y_true_mean)**2).mean()
    print("mean_loss :" ,"%.3f" % mean_loss)
    mean_loss_list.append(mean_loss)
    if val_meta_loss<0.001:
        break
train_meta_loss_list=np.array(train_meta_loss_list)
val_meta_loss_list=np.array(val_meta_loss_list)
mean_loss_list=np.array(mean_loss_list)
# 可视化训练结果
#  val_meta_loss = Metaepoch(meta_model,optimizer,val_loader,loss_fn,inner_train_step=val_inner_train_step,inner_lr=inner_lr,train=False,visualize=True)

# 保存每个任务的格点拟合
meta_weights = OrderedDict(meta_model.named_parameters())

fast_weights_list=[]
for task in data_loader:
    t=task[0][:,0].reshape(-1,1)
    y=task[0][:,1].reshape(-1,1)
    fast_weights, inner_loss=inner_train(t,y,meta_weights=meta_weights,inner_step = val_inner_train_step)
    fast_weights_list.append(fast_weights)
    # print(inner_loss)

dense_data = []
t=torch.arange(0,10,0.01)
for fast_weights in fast_weights_list:
    y_dense=meta_model.functional_forward(t.reshape(-1,1), fast_weights).detach().numpy()
    dense_data.append(y_dense)
t=t.detach().numpy() 

grid_points = t # Grid points of the curves
data_matrix = dense_data
fd = skfda.FDataGrid(
    data_matrix=data_matrix,
    grid_points=grid_points,
)


# %% learning mean function
plt.plot(np.log(train_meta_loss_list),'r',label='Training loss')
plt.plot(np.log(val_meta_loss_list),'g',label='Validation loss')
plt.plot(np.log(mean_loss_list),'b',label='$\| \mu(t)-F_{\\phi_0}(t))\Vert^2 $')

plt.legend()
plt.savefig("./figure/learning_mean.pdf")

# prior after meta training
# t= torch.tensor(np.arange(0,10,0.01)).to(torch.float32).reshape(-1,1)
# y,coord= meta_model.forward(t)
# plt.plot(t.detach().numpy(),y.detach().numpy(),'r')


# %% learning periodicity

t=torch.arange(0,10,0.01)
y=torch.sin(t)

support_t=torch.arange(5,10,0.5)
support_y=torch.sin(support_t)

fast_weights,inner_loss = inner_train(support_t,support_y,meta_weights=meta_weights,inner_step = val_inner_train_step)

y_true =  support_y.reshape(-1,1)
y_predict  = meta_model.functional_forward(support_t.reshape(-1,1), fast_weights).detach().numpy()
ft_predict = meta_model.functional_forward(t.reshape(-1,1), fast_weights).detach().numpy()

t=t.detach().numpy()
support_t=support_t.detach().numpy()
support_y=support_y.detach().numpy()
y_base=localreg(support_t, support_y, x0=t, degree=2, kernel=rbf.gaussian, radius=1)

# print(inner_loss)
plt.plot(support_t,support_y,"y^",markersize=10,label="Observations")
# plt.plot(support_t,y_predict.detach().numpy(),"b+",label="fitted data point")


plt.plot(t,y,"y",label='Ground truth')
plt.plot(t,ft_predict,"r",label='MetaINR')
plt.plot(t,y_base,"g",label="Local polynomial regression")
plt.ylim(-2,2)
plt.legend()
plt.savefig("./figure/learning_periodicity.pdf")


# %% PACE baseline
t_all=np.array([])
y_all=np.array([])
for task in data_loader:
    t_sample=task[0][:,0].detach().numpy()
    y_sample=task[0][:,1].detach().numpy()
    t_all=np.hstack((t_all,t_sample))
    y_all=np.hstack((y_all,y_sample))

t=np.arange(0,10,0.01)
mean_pace=localreg(t_all, y_all,x0=t, degree=1, kernel=rbf.gaussian, radius=0.2)
t_pace=np.arange(0,10,0.2)
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

t_pace=np.arange(0,10,0.2)
X0,Y0=np.meshgrid(t_pace,t_pace)
x0=np.array([np.ravel(X0), np.ravel(Y0)]).T

cov_pace = localreg(input[:], z[:], x0, degree=0,radius=1, kernel=rbf.gaussian)
cov_pace = cov_pace.reshape(X0.shape)
# %%

# %% FPCA

# MetaINR FPCA

fpca_discretized = FPCA(n_components=2)
fpca_discretized.fit(fd)
pc1,pc2=fpca_discretized.components_.data_matrix

# true value PCA

fpca_discretized_true = FPCA(n_components=2)
fpca_discretized_true.fit(fd_true)
pc1_true,pc2_true=fpca_discretized_true.components_.data_matrix

# baseline local polynomial regression PCA

fpca_discretized_base = FPCA(n_components=2)
fpca_discretized_base.fit(fd_base)
pc1_base,pc2_base=fpca_discretized_base.components_.data_matrix

# PACE FPCA
lambda_list,pc_list=np.linalg.eig(cov_pace)

pc1_pace=pc_list[:,0]
pc2_pace=pc_list[:,1]
# %%
# visualize PC1
l5=plt.plot(t,pc1_true,'y',label='pc1_true')
l1=plt.plot(t,pc1,'r:',label='pc1_metaINR')
l3=plt.plot(t,-pc1_base,'g:',label='pc1_Pre-smoothing')
l7=plt.plot(t_pace,-pc1_pace,'b:',label='pc1_PACE')
plt.title("PC1")
plt.ylim(-0.75,0.75)
# plt.legend(bbox_to_anchor=(1.45, 1))
plt.savefig("./figure/PC1.pdf")

# %%
# PC2
l6=plt.plot(t,pc2_true,'y',label='pc2_true')
l2=plt.plot(t,pc2,'r:',label='pc2_metaINR')
l4=plt.plot(t,pc2_base,'g:',label='pc2_Pre-smoothing')
l8=plt.plot(t_pace,pc2_pace,'b:',label='pc2_PACE')
plt.title("PC2")
plt.ylim(-0.75,0.75)
# plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig("./figure/PC2.pdf")


# %% mean estimation , mean function 刚好和prior接近！！！！！！

y_true=fd_true.mean().data_matrix[0,:,0]
plt.plot(t,y_true,'y',label='Ground truth')

mean=fd.mean().data_matrix[0,:,0]
plt.plot(t,mean,'r:',label="MetaINR")

mean_base=fd_base.mean().data_matrix[0,:,0]
plt.plot(t,mean_base,'g:',label="Pre-smoothing")

plt.plot(t,mean_pace,'b:',label="PACE")


# t1= torch.tensor(np.arange(0,10,0.01)).to(torch.float32).reshape(-1,1)
# y,coord= meta_model.forward(t1)
# plt.plot(t,y.detach().numpy(),'g',label="meta model prior")
plt.legend()
plt.savefig("./figure/mean.pdf")



# %% covariance estimation

fontsize=15
fig = plt.figure(figsize=(18, 6), facecolor='w')
t=np.arange(0,10,0.01)
X, Y = np.meshgrid(t, t)

c=np.cov(fd.data_matrix[:,:,0].T)
Z = c
ax1 = fig.add_subplot(1, 4, 2, projection='3d')
ax1.plot_surface(X,Y,Z,alpha=0.2,cmap='winter')
ax1.contour(X,Y,Z,zdir='z', offset=-0.1,cmap="rainbow")
ax1.contour(X,Y,Z,zdir='x', offset=0,cmap="rainbow")  
ax1.contour(X,Y,Z,zdir='y', offset=10,cmap="rainbow")
ax1.set_title("MetaINR",fontsize=fontsize)
ax1.set_zlim(-0.1,0.1)

c3=np.cov(fd_base.data_matrix[:,:,0].T)
Z3 = c3
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3.plot_surface(X,Y,Z3,alpha=0.2,cmap='winter')
ax3.contour(X,Y,Z3,zdir='z', offset=-0.1,cmap="rainbow")
ax3.contour(X,Y,Z3,zdir='x', offset=0,cmap="rainbow")  
ax3.contour(X,Y,Z3,zdir='y', offset=10,cmap="rainbow")
ax3.set_title("Pre-smoothing",fontsize=fontsize)
ax3.set_zlim(-0.1,0.1)

c2=np.cov(fd_true.data_matrix[:,:,0].T)
Z2 = c2
ax2 = fig.add_subplot(1, 4, 1, projection='3d')
ax2.plot_surface(X,Y,Z2,alpha=0.2,cmap='winter')
ax2.contour(X,Y,Z2,zdir='z', offset=-0.1,cmap="rainbow")
ax2.contour(X,Y,Z2,zdir='x', offset=0,cmap="rainbow")  
ax2.contour(X,Y,Z2,zdir='y', offset=10,cmap="rainbow")
ax2.set_title("Ground truth",fontsize=fontsize)
ax2.set_zlim(-0.1,0.1)

c4=cov_pace
Z4 = c4
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
ax4.plot_surface(X0,Y0,Z4,alpha=0.2,cmap='winter')
ax4.contour(X0,Y0,Z4,zdir='z', offset=-0.1,cmap="rainbow")
ax4.contour(X0,Y0,Z4,zdir='x', offset=0,cmap="rainbow")  
ax4.contour(X0,Y0,Z4,zdir='y', offset=10,cmap="rainbow")
ax4.set_title("PACE",fontsize=fontsize)
ax4.set_zlim(-0.1,0.1)

plt.savefig("./figure/covariance.pdf")

# %% PACE recovery
id=12

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
cov_pace_id = localreg(input[:], z[:], x0_sample, degree=0,radius=1, kernel=rbf.gaussian)
cov_pace_id = np.matrix(cov_pace_id.reshape(X0_sample.shape))+np.matrix(0.00001*np.eye(X0_sample.shape[0])) #正则化 防止不可逆
cov_pace_id_inv=np.linalg.inv(cov_pace_id)
# cov_pace_id_inv=np.array(cov_pace_id_inv)

mu_id=mean_pace_all[id*dataset.total_samples_per_task:(id+1)*dataset.total_samples_per_task]

a_1=np.array(lambda_list[0]*np.matrix(phi_1)*cov_pace_id_inv*(np.matrix(y_sample-mu_id).T))[0][0]
a_2=np.array(lambda_list[1]*np.matrix(phi_2)*cov_pace_id_inv*(np.matrix(y_sample-mu_id).T))[0][0]

y_pace=mean_pace_for_t_pace+a_1*pc1_pace+a_2*pc2_pace
# %%
# %%
# %%
# %%
fig = plt.figure(figsize=(18, 6), facecolor='w')
c4=cov_pace
Z4 = c4
ax4 = fig.add_subplot(1, 4, 4, projection='3d')
ax4.plot_surface(X0,Y0,Z4,alpha=0.2,cmap='winter')
ax4.contour(X0,Y0,Z4,zdir='z', offset=-0.1,cmap="rainbow")
ax4.contour(X0,Y0,Z4,zdir='x', offset=0,cmap="rainbow")  
ax4.contour(X0,Y0,Z4,zdir='y', offset=10,cmap="rainbow")
ax4.set_title("PACE",fontsize=fontsize)
# ax4.set_zlim(-0.1,0.1)

ax4.scatter(X0_sample,Y0_sample,cov_pace_id,alpha=0.2,cmap='winter')

# %% Recovery

t=np.arange(0,10,0.01)
plt.plot(t_sample,y_sample,'^',color='olive',markersize=10,label='Observations')
plt.plot(t,fd_true.data_matrix[id][:,0],'y',label='Ground truth')
plt.plot(t,fd.data_matrix[id][:,0],'r:',label='MetaINR')
plt.plot(t,fd_base.data_matrix[id][:,0],'g:',label='Pre-smoothing')
plt.plot(t_pace,y_pace,'b:',label='PACE')
plt.legend()
plt.savefig("./figure/recovery.pdf")
# %% 总体画图


# %%
