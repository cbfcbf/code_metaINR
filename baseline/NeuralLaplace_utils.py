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
extrap=True



# %% NL 的内容

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


def train_NL(trajectories,ti):

    samples = trajectories.shape[0]
    dim = trajectories.shape[2]
    # traj = (
    #     torch.reshape(trajectories, (-1, dim))
    #     - torch.reshape(trajectories, (-1, dim)).mean(0)
    # ) / torch.reshape(trajectories, (-1, dim)).std(0)
    # trajectories = torch.reshape(traj, (samples, -1, dim))
    train_split = int(0.8 * trajectories.shape[0])
    # test_split = int(0.9 * trajectories.shape[0])
    test_split = trajectories.shape[0]
    traj_index = torch.randperm(trajectories.shape[0])
    train_trajectories = trajectories[traj_index[:train_split], :, :]
    val_trajectories = trajectories[traj_index[train_split:test_split], :, :]
    # test_trajectories = trajectories[traj_index[test_split:], :, :]

    dltrain = DataLoader(
        train_trajectories,
        batch_size=128,
        shuffle=True,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            ti,
            data_type="train",
            extrap=extrap,
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
            extrap=extrap,
        ),
    )
    # dltest = DataLoader(
    #     test_trajectories,
    #     batch_size=128,
    #     shuffle=False,
    #     collate_fn=lambda batch: basic_collate_fn(
    #         batch,
    #         ti,
    #         data_type="test",
    #         extrap=True,
    #     ),
    # )


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
        cum_val_loss_all=0
        for batch in dlval:
            trajs_to_encode = batch[
                "observed_data"
            ]  # (batch_size, t_observed_dim, observed_dim)
            observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
            # print(trajs_to_encode.shape)
            # print(observed_tp.shape)
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

            # print(observed_tp.shape)
            # print(tp_to_predict.shape)
            all_tp = torch.concat((observed_tp , tp_to_predict),axis=1)
            # print(all_tp.shape)

            # print(batch["observed_data"].shape)
            # print(batch["data_to_predict"].shape)
            all_truth=torch.concat((batch["observed_data"], batch["data_to_predict"]),axis=1)
            # print(all_truth.shape)

            all_predictions= laplace_reconstruct(
                laplace_rep_func, p, all_tp, recon_dim=output_dim
            )

            cum_val_loss_all += loss_fn(
                torch.flatten(all_predictions), torch.flatten(all_truth)
            ).item()

        # if (epoch % 6 == 0):
        #     visualize(
        #         tp_to_predict.detach(),
        #         predictions.detach(),
        #         batch["data_to_predict"].detach(),
        #         "bofan-test1",
        #         epoch,
        #     )
        val_mse = cum_val_loss / cum_val_batches
        val_mse_all = cum_val_loss_all / cum_val_batches
        print(
            "[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_mse={}\t| val_mse_all={}\t|".format(
                epoch, epoch_duration, epoch_train_loss, val_mse,val_mse_all
            )
        )

        # Early stopping procedure
        if val_mse < best_loss:
            best_loss = val_mse
            best_laplace_rep_func = deepcopy(laplace_rep_func.state_dict())
            best_encoder = deepcopy(encoder.state_dict())
            waiting = 0
        elif waiting > 200:
            break
        else:
            waiting += 1

    visualize(
        all_tp.detach(),
        all_predictions.detach(),
        all_truth.detach(),
        "bofan-test1",
        epoch,
    )

    return best_laplace_rep_func,best_encoder

def sine(trajectories_to_sample, t_nsamples): #100个sample 20个observation
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
        # sample = np.arange(1000) 
        # np.random.shuffle(sample)  # For random sampling the characters we want.
        trajs.append(yi)
        trajs_truth.append(ground_truth)
    y = torch.stack(trajs)
    y_true=torch.stack(trajs_truth)
    trajectories = y.view(trajectories_to_sample, -1, 1)
    trajectories_true = y_true.view(trajectories_to_sample, -1, 1)

    return trajectories, ti, trajectories_true


# %%
if __name__=="__main__":
    Data_class=sine
    trajectories, ti ,trajectories_true = Data_class(100,21)
    best_laplace_rep_func,best_encoder=train_NL(trajectories, ti )
    def sine_test(t):
        y=0.4*torch.sin(t)+0.2*torch.sin(2*t)
        return y

    def sawtooth_test(t):
        x0=-1
        y= (t + x0) / (2 * torch.pi) - torch.floor((t + x0) / (2 * torch.pi))
        return y

    Test_fun=sine_test
    t=torch.arange(0,10,0.01)
    y=Test_fun(t)
    support_t=torch.arange(0,5,0.5)
    support_y=Test_fun(support_t)
    query_t=torch.arange(5,10.5,0.5)
    query_y_true=Test_fun(query_t)
    y_true = query_y_true.detach().numpy()
    t_plot = query_t.detach().numpy()

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
    y_NL = torch.squeeze(predictions).detach().numpy()

    # trajs_to_encode=trajs_to_encode[[0],:,:]
    # p2 = encoder(
    #     trajs_to_encode, observed_tp
    # )  # p is the latent tensor encoding the initial states
    # predictions2 = laplace_reconstruct(
    #     laplace_rep_func, p2, tp_to_predict, recon_dim=output_dim
    # )
    # # tp_to_predict = torch.squeeze(tp_to_predict).detach().numpy()
    # y_NL2 = torch.squeeze(predictions2).detach().numpy()

