# %% 
import data_loader as dl
import numpy as np
import matplotlib.pyplot as plt
import torch
import modules
import time
from torch.utils.data import DataLoader
import copy
from copy import deepcopy

device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

class TINR:
    def __init__(self, dataset,model):
        self.dataset = dataset
        self.model=model

        self.train_shape = []  # train set information
        self.train_representation_time = []
        self.train_representation_epoch = []
        self.train_time_per_epoch = []

        self.best_loss=None
        self.state_dict=None
        

    def Representation_Learning(
        self,
        batch_size=10000,
        epochs=10000,
        learning_rate=1e-4
    ):
        torch.cuda.empty_cache()
        data=self.dataset

        model=self.model

        # model = modules.Siren(
        #     in_features=1,
        #     out_features=1,
        #     hidden_features=hidden_dim,
        #     hidden_layers=hidden_layers,
        #     first_omega_0=first_omega_0,
        #     outermost_linear=True,
        # )
        # model = modules.Net()

        model.to(device)

        optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
        # early_stopping = utils.EarlyStopping(
        #     patience=earlystopping_patience, verbose=False
        # )
        start = time.time()

        train_data=modules.TimeData(data.x)
        
        train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=0,
        )

        epoch_time = []
        loss_fn=torch.nn.MSELoss()
        model_loss = []

        for step in range(epochs):
            epoch_start = time.time()
            for t,y in train_dataloader:

                batch_model_input = t.to(device).float()
                batch_ground_truth = y.to(device).float()
                
                batch_model_output, _ = model(batch_model_input)


                loss = loss_fn(batch_model_output, batch_ground_truth)
                optim.zero_grad()
                loss.backward()
                optim.step()
                model_loss.append(loss.item())
                batch_model_input = batch_model_input.detach().cpu()
                batch_ground_truth = batch_ground_truth.detach().cpu()
            epoch_time.append(time.time() - epoch_start)
            if step%100==0:
                print(step)
                print("loss:",str(loss))
                if step>10 and np.array(model_loss)[-10].mean()<1e-5:
                    break

        self.train_time_per_epoch.append(np.mean(epoch_time))

        time_required = copy.deepcopy(time.time() - start)

        self.train_representation_epoch.append(step)
        self.train_representation_time.append(time_required)

        self.best_loss = loss
        self.best_state_dict = deepcopy(model.state_dict())

    def visualize(self):

        t=torch.tensor([[i] for i in np.arange(0,10,0.1)]).float()

        # model=modules.Net()
        # model = modules.Siren(
        #     in_features=1,
        #     out_features=1,
        #     hidden_features=hidden_dim,
        #     hidden_layers=hidden_layers,
        #     first_omega_0=first_omega_0,
        #     outermost_linear=True,
        # )
        model=self.model
        model.load_state_dict(self.best_state_dict)
        # model.eval()

        y , _= model(t)

        # plt.plot(t,y)
        t1=np.array(t)[:,0]
        y1=y.detach().numpy()[:,0]
        print("loss:",str(self.best_loss))
        plt.plot(t1,y1)
        plt.plot(self.dataset.t,self.dataset.y,"r+")

        return t1,y1,self.best_loss

# %% Training

BATCH_SIZE = 10000
EPOCHS = 1000

#Siren parameters
HIDDEN_DIM = 256
FIRST_OMEGA_0 = 3 #与图像的频率越契合则越好学
HIDDEN_LAYERS = 3

w=10
dataset=dl.synthetic_data(w)

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
