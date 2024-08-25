import torch
from torch import nn
from torch.utils.data import DataLoader
from ray import train, tune

# ---------------------------------- helper functions ----------------------------------
# custom loss function
def weighted_mse_loss(input, target, weight):
    return torch.mean(((weight * (input - target)) ** 2).to(torch.float32))

# ---------------------------------- configurable neural network ----------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, l1, l2):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, l1),
            nn.ReLU(),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.Linear(l2, 2),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

# ---------------------------------- ray tune trainable ----------------------------------
def trainable(config, train_dataset, val_dataset):
    # constants
    epochs = 30
    tau_weight = 0.01

    # setting up configurable neural network, optimizer, and lr scheduler
    model = NeuralNetwork(config['l1'], config['l2'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])

    # creating dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle = True)

    # optimization loop
    for i in range(epochs):
        # training loop
        for batch, sample_batched in enumerate(train_loader):
            x, y = sample_batched['tseries'], sample_batched['params']
            # Compute prediction and loss
            pred = model(x)
            loss = weighted_mse_loss(pred, y, torch.tensor([1, tau_weight]))
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # validation loop
        test_loss = 0
        with torch.no_grad():
            for sample_batched in val_loader:
                x, y = sample_batched['tseries'], sample_batched['params']
                pred = model(x)
                test_loss += weighted_mse_loss(pred, y, torch.tensor([1, tau_weight])).item()
        test_loss /= len(val_loader)
        
        # update lr
        scheduler.step()

        # report metric to ray
        train.report({"Avg loss": test_loss})

# ---------------------------------- hyperparameter search space ----------------------------------
config = {
    'l1': tune.grid_search([64]),
    'l2': tune.grid_search([8, 16, 32]),
    'lr': tune.grid_search([0.05]),
    'gamma': tune.grid_search([0.9]),
    'batch_size': tune.grid_search([16])
}

# ---------------------------------- hyperparameter tuning ----------------------------------
# loading training, validation, and testing datasets
train_dataset = torch.load('hyperparameter tuning sets/HPT_200x200x10_train.pt')
val_dataset = torch.load('hyperparameter tuning sets/HPT_200x200x10_val.pt')
test_dataset = torch.load('hyperparameter tuning sets/HPT_200x200x10_test.pt')

tuner = tune.Tuner(
    tune.with_parameters(trainable, train_dataset=train_dataset, val_dataset=val_dataset), 
    param_space=config
)
results = tuner.fit()