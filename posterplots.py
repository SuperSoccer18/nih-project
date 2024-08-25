import torch
import numpy as np
import matplotlib.pyplot as plt
import datagen

# ---------------------------------- Signal-output example ----------------------------------
# f = datagen.monoexp(0.8, 100)
# ts = datagen.timeseries(f)
# noise = torch.from_numpy(np.random.normal(0, 1, 32))
# snr = 100
# sigma = 1/snr   # sigma = a/snr --> average a is 1
# noise *= sigma
# signal = ts + noise
# x = np.linspace(0, 3 * 300, 32)
# plt.title('Monoexponential Signal (C = 0.8, τ = 100)', fontsize=16, pad=10)
# plt.scatter(x, signal, color='red', s=10)
# plt.plot(x, f(x), color='blue')
# plt.xlabel('time (ms)', fontsize=14)
# plt.ylabel('signal', fontsize=14)
# # plt.grid()
# plt.show()

# ---------------------------------- Training Grid ----------------------------------
# tau = np.linspace(100, 300, 6)
# a = np.linspace(0.8, 1.2, 6)
# tau, a = np.meshgrid(tau, a)
# tau, a = tau.ravel(), a.ravel()
# plt.scatter(tau, a)
# plt.title('6x6 Training Set Discretization', fontsize=16, pad=10)
# plt.xlabel('τ', fontsize=16)
# plt.ylabel('C', fontsize=16)
# plt.show()

# ---------------------------------- Training vs. Test grid ----------------------------------
# fig, ax = plt.subplots()

# tau = np.linspace(100, 300, 21)
# a = np.linspace(0.8, 1.2, 21)
# tau, a = np.meshgrid(tau, a)
# tau, a = tau.ravel(), a.ravel()
# test = ax.scatter(tau, a, c='gold', label='test')

# tau = np.linspace(100, 300, 6)
# a = np.linspace(0.8, 1.2, 6)
# tau, a = np.meshgrid(tau, a)
# tau, a = tau.ravel(), a.ravel()
# train = ax.scatter(tau, a, c='r', label='train')

# plt.title('Input space', fontsize=18, pad=10)
# plt.xlabel('τ', fontsize=18)
# plt.ylabel('C', fontsize=18)
# ax.legend([train, test], ['train', 'test'], bbox_to_anchor=(1.04, 1), framealpha=1, edgecolor='inherit')
# plt.show()

# ---------------------------------- Discretization vs. Accuracy ----------------------------------
# x = np.array([3, 6, 11, 21, 51, 101, 201])
# y = np.array([0.722, 0.152, 0.0829, 0.0217, 0.00133, 0.000355, 0.000264])
# plt.scatter(x, y, s = 15)
# plt.title('Discretization vs. Accuracy', fontsize=18, pad=10)
# plt.xlabel('Training set discretization', fontsize=14)
# plt.ylabel('Test set error', fontsize=14)
# plt.grid()
# plt.show()