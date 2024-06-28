# adapted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# here we fit a linear regression model with one covariate, x
# the model is y = b0 + b1 * x + epsilon, where epsilon ~ Normal(0, 1)
# we estimate b0 and b1

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_model = nn.Linear(1, 1, device=device)

    def forward(self, x):
        y_hat = self.linear_model(x)
        return y_hat


model = RegressionModel().to(device)
print(model)

# set up a data set
n_train = 20
b0 = 10
b1 = 2
x_train = torch.rand(n_train, 1, device=device) * 10
y_train = torch.normal(b0 + b1 * x_train, torch.ones((n_train, 1), device=device))

plt.plot(torch.Tensor.cpu(x_train), torch.Tensor.cpu(y_train), 'o')
plt.show()

# before doing parameter estimation, what do we have?
def get_model_coefs(model):
    model_params = list(model.parameters())
    model_params
    model_b1 = model_params[0].data
    model_b0 = model_params[1].data
    return model_b0, model_b1

init_b0, init_b1 = get_model_coefs(model)
print(init_b0, init_b1)

y_hat_manual = init_b0 + init_b1 * x_train
y_hat_from_model = model(x_train)

torch.all(y_hat_manual == y_hat_from_model)

# create a Dataset with our training X's and Y's
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        '''
        Initialize dataset
        '''
        assert x.shape == y.shape
        self.x = x
        self.y = y
    
    def __len__(self):
        '''
        How many instances in the dataset?
        '''
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        '''
        Return instance at specified index
        '''
        return self.x[idx, ...], self.y[idx, 0]


train_loader = DataLoader(SimpleDataset(x_train, y_train))


# Function that does parameter estimation
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



# "hyperparameters"
# these describe how parameter optimization/estimation should work
learning_rate = 1e-2
batch_size=20
epochs = 3

# loss function and method for optimizing the loss
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# do estimation
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)

# inspect the parameter estimates
estimated_b0, estimated_b1 = get_model_coefs(model)
print(estimated_b0, estimated_b1)

# make a plot
x_test = torch.arange(0, 11, dtype=torch.float32, device=device)[:, None]
y_hat = model(x_test)

plt.plot(torch.Tensor.cpu(x_train), torch.Tensor.cpu(y_train), 'o')
plt.plot(torch.Tensor.cpu(x_test), y_hat.detach().numpy())
plt.show()

