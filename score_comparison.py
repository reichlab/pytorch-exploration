# adapted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# here we fit a linear regression model with one covariate, x
# the model is y = b0 + b1 * x + epsilon, where epsilon ~ Normal(0, 1)
# we estimate b0 and b1

import torch
from torch import nn
from torch import distributions as d
from torch.utils.data import Dataset, DataLoader
import scipy.stats as sps
import numpy as np

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class normModel(nn.Module):
    def __init__(self, init_loc=None, init_scale=None):
        """
        
        """
        super().__init__()
        if init_loc is not None:
            self.loc = nn.Parameter(init_loc)
        else: 
            self.loc = nn.Parameter(torch.randn(1))
        
        if init_scale is not None:
            self.scale = nn.Parameter(init_scale)
        else:
            self.scale = nn.Parameter(torch.abs(torch.randn(1)))
        # self.log_scale = nn.Parameter(torch.randn(1))
        # self.scale = torch.exp(self.log_scale)
        self.dist = d.Normal(self.loc, self.scale)


# create a Dataset with our training X's
class SimpleDataset(Dataset):
    def __init__(self, x):
        '''
        Initialize dataset
        '''
        self.x = x
    
    def __len__(self):
        '''
        How many instances in the dataset?
        '''
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        '''
        Return instance at specified index
        '''
        return self.x[idx, ...]


# set up a data set
n_train = 20
batch_size = n_train
dgp = d.Normal(loc=0.0, scale=1.0)
x_train = dgp.sample([n_train])
sample_mean = torch.mean(x_train).detach().numpy()
sample_sd = torch.std(x_train, dim=None, correction=0).detach().numpy()

train_loader = DataLoader(SimpleDataset(x_train), batch_size=batch_size)


# Function that does parameter estimation
def train_step(x_train, model, loss_fn, optimizer):
    # Compute prediction and loss
    loss = loss_fn(model, x_train)

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = loss.item()
    print(f"loss: {loss:>7f}")



# "hyperparameters"
# these describe how parameter optimization/estimation should work
learning_rate = 1e-3
epochs = 2000

# models
model1 = normModel(
    init_loc=torch.mean(x_train),
    init_scale=torch.std(x_train),
)
model2 = normModel(
    init_loc=torch.mean(x_train),
    init_scale=torch.std(x_train),
)


def nll_loss(model, x):
    return -torch.sum(model.dist.log_prob(x))


q_lvls = np.array([[0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99]])
norm_offsets = torch.Tensor(sps.norm.ppf(q_lvls))
q_lvls = torch.Tensor(q_lvls)

def quantile_loss(model, x):
    """
    adapted from https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/metrics/quantile.html#QuantileLoss
    """
    losses = []
    pred = model.loc + model.scale * norm_offsets
    errors = x - pred
    losses = 2 * torch.max((q_lvls - 1) * errors, q_lvls * errors)
    return torch.sum(losses)



# loss function and method for optimizing the loss
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# do estimation
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_step(x_train, model1, nll_loss, optimizer)


# inspect the parameter estimates
loss_nll = nll_loss(model1, x_train)
loss_nll.backward()

model1.loc.grad
model1.scale.grad

# estimates
loc_hat1 = model1.loc.detach().numpy()
scale_hat1 = model1.scale.detach().numpy()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_step(torch.unsqueeze(x_train, -1), model2, quantile_loss, optimizer)

loss_q = quantile_loss(model2, x_train)
loss_q.backward()
model2.loc.grad
model2.scale.grad


import pandas as pd

# Assuming loc_hat and scale_hat are for model1 from the provided code
# For model2, you would extract them similarly after its training loop
loc_hat2 = model2.loc.detach().numpy()
scale_hat2 = model2.scale.detach().numpy()

# Create a DataFrame
df = pd.DataFrame({
    'Parameter': ['loc', 'scale'],
    'sample': [mean(x_train), std(x_train)],
    'Model1': [loc_hat, scale_hat],
    'Model2': [loc_hat2, scale_hat2]
})

print(df) 