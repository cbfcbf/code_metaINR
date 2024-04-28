# %%

from copy import deepcopy
from pathlib import Path
from time import strftime, time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import random

from torchlaplace import laplace_reconstruct
from torchlaplace.data_utils import basic_collate_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

learning_rate=1e-3

# Fix random seeds
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
    

def sawtooth(trajectories_to_sample=100, t_nsamples=200):
    # Toy sawtooth waveform. Simple to generate, for Differential Equation Datasets see datasets.py (Note more complex DE take time to sample from, in some cases minutes).
    t_end = 20.0
    t_begin = t_end / t_nsamples
    ti = torch.linspace(t_begin, t_end, t_nsamples).to(device)

    def sampler(t, x0=0):
        return (t + x0) / (2 * torch.pi) - torch.floor((t + x0) / (2 * torch.pi))

    x0s = torch.linspace(0, 2 * torch.pi, trajectories_to_sample)
    trajs = []
    for x0 in x0s:
        trajs.append(sampler(ti, x0))
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti


def sine(trajectories_to_sample=100, t_nsamples=20): #100个sample 20个observation
    ti=torch.linspace(0, 10, t_nsamples).to(device)
    t=np.arange(0,10,0.01) # 1000 potential obs in total
    trajs = []
    for i in range(trajectories_to_sample):
        A1s=random.random()
        A2s=random.random()*0.5
        w1 = 1
        w2 = 2
        ground_truth= A1s*np.sin(w1*t) + A2s*np.sin(w2*t) 
        yi= A1s*np.sin(w1*ti) + A2s*np.sin(w2*ti) 
        sample = np.arange(1000) 
        np.random.shuffle(sample)  # For random sampling the characters we want.
        trajs.append(yi)
    y = torch.stack(trajs)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    return trajectories, ti

Data_class=sine
trajectories, ti= Data_class()


# %% NL 的内容
samples = trajectories.shape[0]
dim = trajectories.shape[2]
traj = (
    torch.reshape(trajectories, (-1, dim))
    - torch.reshape(trajectories, (-1, dim)).mean(0)
) / torch.reshape(trajectories, (-1, dim)).std(0)
trajectories = torch.reshape(traj, (samples, -1, dim))
train_split = int(0.8 * trajectories.shape[0])
test_split = int(0.9 * trajectories.shape[0])
traj_index = torch.randperm(trajectories.shape[0])
train_trajectories = trajectories[traj_index[:train_split], :, :]
val_trajectories = trajectories[traj_index[train_split:test_split], :, :]
test_trajectories = trajectories[traj_index[test_split:], :, :]


# import matplotlib.pyplot as plt
# plt.plot(train_trajectories[3,:,0],)

dltrain = DataLoader(
    train_trajectories,
    batch_size=128,
    shuffle=True,
    collate_fn=lambda batch: basic_collate_fn(
        batch,
        ti,
        data_type="train",
        extrap=True,
    ),
)
dlval = DataLoader(
    val_trajectories,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: basic_collate_fn(
        batch,
        ti,
        data_type="test",
        extrap=True,
    ),
)
dltest = DataLoader(
    test_trajectories,
    batch_size=128,
    shuffle=False,
    collate_fn=lambda batch: basic_collate_fn(
        batch,
        ti,
        data_type="test",
        extrap=True,
    ),
)

# for batch in dlval:
#     observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
#     trajs_to_encode = batch[
#             "observed_data"
#         ]  # (batch_size, t_observed_dim, observed_dim)
#     tp_to_predict = batch["tp_to_predict"]
#     data_to_predict=batch["data_to_predict"]
#     break

# data_to_predict.detach().cpu().numpy().shape
# plt.plot(tp_to_predict.squeeze().detach().cpu().numpy(), data_to_predict.squeeze().detach().cpu().numpy()[0, :])

# # %%
# plt.plot(observed_tp.squeeze().detach().cpu().numpy(), trajs_to_encode.squeeze().detach().cpu().numpy()[0, :])

class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self, dimension_in, latent_dim, hidden_units, encode_obs_time=True):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            trajs_to_encode = torch.cat(
                (
                    observed_data,
                    observed_tp.view(1, -1, 1).repeat(observed_data.shape[0], 1, 1),
                ),
                dim=2,
            )
        reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])


class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi


def visualize(tp_to_predict, predictions, data_to_predict, path_run_name, epoch):
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update({"font.size": 12})
    fig = plt.figure(figsize=(12, 4), facecolor="white")
    ax_one = fig.add_subplot(131, frameon=False)
    ax_two = fig.add_subplot(132, frameon=False)
    ax_three = fig.add_subplot(133, frameon=False)

    tp_to_predict = torch.squeeze(tp_to_predict)
    predictions = torch.squeeze(predictions)
    y_true = torch.squeeze(data_to_predict)

    y_margin = 1.1
    ax_one.cla()
    ax_one.set_title("Sample 0")
    ax_one.set_xlabel("t")
    ax_one.set_ylabel("x")
    ax_one.plot(tp_to_predict.cpu().numpy(), y_true.cpu().numpy()[0, :], "k--")
    ax_one.plot(tp_to_predict.cpu().numpy(), predictions.cpu().numpy()[0, :], "b-")
    ax_one.set_xlim(tp_to_predict.cpu().min(), tp_to_predict.cpu().max())
    ax_one.set_ylim(y_true.cpu().min() * y_margin, y_true.cpu().max() * y_margin)

    ax_two.cla()
    ax_two.set_title("Sample 1")
    ax_two.set_xlabel("t")
    ax_two.set_ylabel("x")
    ax_two.plot(tp_to_predict.cpu().numpy(), y_true.cpu().numpy()[1, :], "k--")
    ax_two.plot(tp_to_predict.cpu().numpy(), predictions.cpu().numpy()[1, :], "b-")
    ax_two.set_xlim(tp_to_predict.cpu().min(), tp_to_predict.cpu().max())
    ax_two.set_ylim(y_true.cpu().min() * y_margin, y_true.cpu().max() * y_margin)

    ax_three.cla()
    ax_three.set_title("Sample 2")
    ax_three.set_xlabel("t")
    ax_three.set_ylabel("x")
    ax_three.plot(tp_to_predict.cpu().numpy(), y_true.cpu().numpy()[2, :], "k--")
    ax_three.plot(tp_to_predict.cpu().numpy(), predictions.cpu().numpy()[2, :], "b-")
    ax_three.set_xlim(tp_to_predict.cpu().min(), tp_to_predict.cpu().max())
    ax_three.set_ylim(y_true.cpu().min() * y_margin, y_true.cpu().max() * y_margin)

    fig.tight_layout()
    plt.draw()


input_dim = train_trajectories.shape[2]
output_dim = input_dim
encoder = ReverseGRUEncoder(
    input_dim,
    2,
    64 // 2,
).to(device)
laplace_rep_func = LaplaceRepresentationFunc(
    33, output_dim, 2
).to(device)

params = list(laplace_rep_func.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
loss_fn = torch.nn.MSELoss()

best_loss = float("inf")
waiting = 0

for epoch in range(1000):
    iteration = 0
    epoch_train_loss_it_cum = 0
    start_time = time()
    laplace_rep_func.train(), encoder.train()
    for batch in dltrain:
        optimizer.zero_grad()
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(
            laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
        )
        loss = loss_fn(
            torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1)
        optimizer.step()
        epoch_train_loss_it_cum += loss.item()
        iteration += 1
    epoch_train_loss = epoch_train_loss_it_cum / iteration
    epoch_duration = time() - start_time

    # Validation step
    laplace_rep_func.eval(), encoder.eval()
    cum_val_loss = 0
    cum_val_batches = 0
    for batch in dlval:
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(
            laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
        )
        cum_val_loss += loss_fn(
            torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
        ).item()
        cum_val_batches += 1
    # if (epoch % 6 == 0):
    #     visualize(
    #         tp_to_predict.detach(),
    #         predictions.detach(),
    #         batch["data_to_predict"].detach(),
    #         "bofan-test1",
    #         epoch,
    #     )
    val_mse = cum_val_loss / cum_val_batches
    print(
        "[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_mse={}\t|".format(
            epoch, epoch_duration, epoch_train_loss, val_mse
        )
    )

    # Early stopping procedure
    if val_mse < best_loss:
        best_loss = val_mse
        best_laplace_rep_func = deepcopy(laplace_rep_func.state_dict())
        best_encoder = deepcopy(encoder.state_dict())
        waiting = 0
    elif waiting > 500:
        break
    else:
        waiting += 1


# %%
visualize(
    tp_to_predict.detach(),
    predictions.detach(),
    batch["data_to_predict"].detach(),
    "bofan-test1",
    epoch,
)
# %%
val_mse
# %%
