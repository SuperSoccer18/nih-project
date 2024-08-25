import torch
import nn_trainer
import matplotlib.pyplot as plt
import numpy as np

# constants
disc_level, nr = 51, 160
size = 0.5
test_disc = 1001

# load trained model
print('Loading model...')
model_state_dict = torch.load(f'trained models/MODEL_{disc_level}x{disc_level}x{nr}.pt')
model = nn_trainer.NeuralNetwork()
model.load_state_dict(model_state_dict)

# iterate through super fine testing set: (a, tau, 10 noise realizations, time series)
# test on each combo of a and tau; plug in 10x32 tensor into model --> 10x2 --> compute mse loss w/r to target
# record loss in new tensor: (a, tau, mse loss of model over 10 noise realizations)
print('Loading test set...')
test_signals = torch.load(f'test sets/TEST_SIGNALS_{test_disc}x{test_disc}x100.pt')
test_output = torch.load(f'test sets/TEST_OUTPUT_{test_disc}x{test_disc}x100.pt')
loss_tensor = torch.empty(test_disc, test_disc)

for i in range(test_disc):
    print(i)
    for j in range(test_disc):
        x = test_signals[i][j]
        pred = model(x)
        loss_tensor[i][j] = nn_trainer.weighted_mse_loss(pred, test_output[i][j], torch.tensor([1, nn_trainer.tau_weight])).item()

print(f'Average Loss: {loss_tensor.mean()}')

# plot line graphs: a held constant, x axis - tau, y axis - mse loss
# print('Plotting line graphs...')
# a_idx = 0   # a = 0.8
# x = np.linspace(datagen.tau_lower, datagen.tau_upper, test_disc)
# plt.plot(x, loss_tensor[a_idx])
# plt.vlines(np.linspace(datagen.tau_lower, datagen.tau_upper, disc_level), 0, 1, colors='r', linestyles='dashed')
# plt.show()

# a_idx = 500    # a = 1
# x = np.linspace(datagen.tau_lower, datagen.tau_upper, test_disc)
# plt.plot(x, loss_tensor[a_idx])
# plt.vlines(np.linspace(datagen.tau_lower, datagen.tau_upper, disc_level), 0, 1, colors='r', linestyles='dashed')
# plt.show()

# a_idx = 1000    # a = 1.2
# x = np.linspace(datagen.tau_lower, datagen.tau_upper, test_disc)
# plt.plot(x, loss_tensor[a_idx])
# plt.vlines(np.linspace(datagen.tau_lower, datagen.tau_upper, disc_level), 0, 1, colors='r', linestyles='dashed')
# plt.show()

# plot heat map: x axis - a, y axis - tau, color - mse loss
print('Plotting heat map...')
heatmap = plt.imshow(loss_tensor, cmap='hot', interpolation='nearest')
plt.title(f'{disc_level}x{disc_level} Discretized Model', fontsize=18, pad=10)
plt.xlabel('τ', fontsize=18)
plt.xticks(np.linspace(0, 1000, 6), np.linspace(100, 300, 6).astype(int))
plt.ylabel('C', fontsize=18)
plt.yticks(np.linspace(0, 1000, 6), np.linspace(0.8, 1.2, 6))
plt.colorbar(heatmap)
# tau = np.linspace(0, 1000, disc_level)
# a = np.linspace(0, 1000, disc_level)
# tau, a = np.meshgrid(tau, a)
# tau, a = tau.ravel(), a.ravel()
# plt.scatter(tau, a, c='lime', s=size)
plt.show()