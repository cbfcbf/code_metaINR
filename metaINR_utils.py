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

train_inner_train_step = 1
val_inner_train_step = 10
first_omega_0=3
inner_lr = 0.001
meta_lr = 0.001
# first_omega_0=0.5
# inner_lr = 0.001
# meta_lr = 0.0001

max_epoch = 1000
inner_batch_size=1 

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

# training function
def Metaepoch(model,optimizer,data_loader,loss_fn,
               inner_train_step = 1, inner_lr=0.1, train=True,visualize=False):
    criterion, task_loss= loss_fn, []
    # task_D_loss=[]
    for train_batch in data_loader: #[10,70,2] 共10个task 
        # for i in range(inner_batch_size): # batch 中的第 i 个 task 共10个task #这个循环可能可以去掉
        # Get data
        i=0
        number_of_samples_in_support_set=int(len(train_batch[i])*0.5)
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

def inner_train(t,y, meta_weights,loss_fn, inner_lr,meta_model,inner_step = 0.0001):
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

# dataloader initialization

def dataloader_init(data, shuffle=True, num_workers=0, inner_batch_size=1):
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


# metaINR model init
def model_init(device, meta_lr,first_omega_0):
    meta_model = Siren(in_features=1,hidden_features=40,hidden_layers=3,out_features=1,first_omega_0=first_omega_0).to(device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
    loss_fn = nn.MSELoss().to(device)
    return meta_model, optimizer, loss_fn


# %% training
def MetaINR_training(dataset,device):

    train_split = int(0.8 * len(dataset))
    val_split = len(dataset) - train_split
    train_set, val_set = torch.utils.data.random_split(dataset, [train_split, val_split])
    (train_loader, val_loader), (train_iter, val_iter) = dataloader_init((train_set, val_set))

    data_loader = DataLoader(
            dataset,
            batch_size=inner_batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True
        )

    meta_model, optimizer, loss_fn = model_init(device, meta_lr,first_omega_0)


    train_meta_loss_list=[]
    val_meta_loss_list=[]
    mean_loss_list=[]
    best_loss = float("inf")
    waiting = 0

    for epoch in range(max_epoch):
        train_meta_loss = Metaepoch(meta_model,optimizer,train_loader,loss_fn,inner_train_step=train_inner_train_step,inner_lr=inner_lr,train=True)
        val_meta_loss= Metaepoch(meta_model,optimizer,val_loader,loss_fn,inner_train_step=val_inner_train_step,inner_lr=inner_lr,train=False)
        print("Epoch :" ,"%d" % epoch, end="\t")
        print("Train loss :" ,"%.3f" % train_meta_loss, end="\t")
        print("Validation loss :" ,"%.3f" % val_meta_loss)
        train_meta_loss_list.append(train_meta_loss)
        val_meta_loss_list.append(val_meta_loss)

        #看是否元模型学习了task的均值
        # t1= torch.tensor(np.arange(0,10,0.01)).to(torch.float32).reshape(-1,1)
        # y,coord = meta_model.forward(t1)
        # y=y.detach().numpy()[:,0]
        # y_true_mean=fd_true.mean().data_matrix[0,:,0]
        # mean_loss = ((y-y_true_mean)**2).mean()
        # print("mean_loss :" ,"%.3f" % mean_loss)
        # mean_loss_list.append(mean_loss)


        val_mse=val_meta_loss
        if val_mse < best_loss-0.0002:
            best_loss = val_mse
            waiting = 0
            meta_weights = OrderedDict(meta_model.named_parameters())

        elif waiting > 200:
            break
        else:
            waiting += 1

        if best_loss<0.001:
            break

    train_meta_loss_list=np.array(train_meta_loss_list)
    val_meta_loss_list=np.array(val_meta_loss_list)
    mean_loss_list=np.array(mean_loss_list)

    # # 保存每个任务的格点拟合
    # fast_weights_list=[]
    # for task in data_loader:
    #     t=task[0][:,0].reshape(-1,1)
    #     y=task[0][:,1].reshape(-1,1)
    #     fast_weights, inner_loss=inner_train(t,y,meta_weights,loss_fn, inner_lr,meta_model,inner_step = val_inner_train_step)
    #     fast_weights_list.append(fast_weights)
    #     # print(inner_loss)

    # dense_data = []
    # t=torch.arange(0,10,0.01)
    # for fast_weights in fast_weights_list:
    #     y_dense=meta_model.functional_forward(t.reshape(-1,1), fast_weights).detach().numpy()
    #     dense_data.append(y_dense)
    # t=t.detach().numpy() 

    # grid_points = t # Grid points of the curves
    # data_matrix = dense_data
    # fd = skfda.FDataGrid(
    #     data_matrix=data_matrix,
    #     grid_points=grid_points,
    # )

    return meta_weights
# %%
