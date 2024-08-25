import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import datagen

# hyperparameters for SGD
learning_rate = 0.05
momentum = 0  # bad - left unimplemented
weight_decay = 0 # bad - left unimplemented
nesterov = False # bad - left unimplemented
gamma = 0.95    # not formally tuned
batch_size = 16
epochs = 10
tau_weight = 0.01   # not formally tuned

# hyperparameters for Adam
# learning_rate = 0.005
# betas = (0.1, 0.999)
# weight_decay = 0 # unimplementeds
# amsgrad = False # unimplemented
# gamma = 0.9
# batch_size = 16
# epochs =  30
# tau_weight = 0.01

disc_level, nr = 3, 44890     # which train set to use

# ---------------------------------- loading data ----------------------------------
# loading saved training and testing datasets
train_dataset = torch.load(f'training sets/TRAIN_SET_{disc_level}x{disc_level}x{nr}.pt')
test_dataset = torch.load('hyperparameter tuning sets/HPT_200x200x10_test.pt')

# instantiating DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------------- neural network ----------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

# ---------------------------------- optimization loop ----------------------------------
def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode
    model.train()

    for sample_batched in dataloader:
        x, y = sample_batched['tseries'], sample_batched['params']
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y, torch.tensor([1, tau_weight]))
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for sample_batched in dataloader:
            x, y = sample_batched['tseries'], sample_batched['params']
            pred = model(x)
            test_loss += loss_fn(pred, y, torch.tensor([1, tau_weight])).item()

    test_loss /= num_batches
    print(f'Avg loss: {test_loss:>8f}')

# ---------------------------------- custom loss function ----------------------------------
def weighted_mse_loss(output, target, weight):
    return torch.mean(((weight * (output - target)) ** 2).to(torch.float32))

# ---------------------------------- training ----------------------------------
model = NeuralNetwork()
loss_fn = weighted_mse_loss

# SGD Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Adam Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# for t in range(epochs):
#     print(f"Epoch {t+1}")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
#     scheduler.step()
#     print('----------------------------')
# print("Done!")

# ---------------------------------- sanity check ----------------------------------
# f = datagen.monoexp(1.1, 250)
# ts1 = datagen.timeseries(f)
# out1 = model(ts1)
# print('answer: 1.1, 250; result: ' + str(out1))

# g = datagen.monoexp(0.9, 150)
# ts2 = datagen.timeseries(g)
# out2 = model(ts2)
# print('answer: 0.9, 150; result: ' + str(out2))

# ---------------------------------- saving model ----------------------------------
# torch.save(model.state_dict(), f'MODEL_{disc_level}x{disc_level}x{nr}.pt')