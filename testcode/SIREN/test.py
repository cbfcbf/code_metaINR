# %%
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

inner_train_step = 1000
inner_lr=0.001
first_omega_0=3
# %%
support_t=torch.arange(0,10,0.5)
support_y=torch.sin(support_t)
plt.plot(support_t,support_y,"r+")
# %%
a=np.arange(1,4+1)
a
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% y是一维的比较好

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# %%

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


model=Siren(in_features=1,hidden_features=40,hidden_layers=3,out_features=1,first_omega_0=first_omega_0).to(device)
loss_fn = nn.MSELoss().to(device)

# fast_weights = OrderedDict(model.named_parameters())
# fast_weights.keys()

# %%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1   = nn.Linear(1, 40)
        self.fc2   = nn.Linear(40, 40)
        self.fc3   = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def functional_forward(self,x,params):
        """
        Arguments:
        x: input timepoints [batch, 1]
        params: model parameters,
                i.e. weights and biases of convolution
                     and weights and biases of
                                   batch normalization
                type is an OrderedDict
        """
        x=F.relu(F.linear(x,weight=params["fc1.weight"],bias=params["fc1.bias"]))
        x=F.relu(F.linear(x,weight=params["fc2.weight"],bias=params["fc2.bias"]))
        x=F.linear(x,weight=params["fc3.weight"],bias=params["fc3.bias"])

        return x


# model = Net().to(device)
# loss_fn = nn.MSELoss().to(device)

# %%
fast_weights = OrderedDict(model.named_parameters())
for inner_step in range(inner_train_step):
    # Simply training
    y_true =  support_y.reshape(-1,1)
    y_predict  = model.functional_forward(support_t.reshape(-1,1), fast_weights)
    loss = loss_fn(y_predict, y_true)
    # Inner gradients update! #
    """ Inner Loop Update """
    grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
    fast_weights = OrderedDict(
        (name, param - inner_lr * grad)
        for ((name, param), grad) in zip(fast_weights.items(), grads)
    )
    print(loss)
    
y_true =  support_y.reshape(-1,1)
y_predict  = model.functional_forward(support_t.reshape(-1,1), fast_weights)

# %%
plt.plot(support_t,support_y,"r+")
plt.plot(support_t,y_predict.detach().numpy(),"b+")
# %%
