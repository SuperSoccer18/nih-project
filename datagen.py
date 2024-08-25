import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from math import e

# constants
a_lower, a_upper = 0.8, 1.2
tau_lower, tau_upper = 100, 300
time_series_steps = 32
a_disc_steps = 4
tau_disc_steps = 4
noise_realizations = 25250

# ---------------------------------- helper functions ----------------------------------
# return monoexponential function w given a and tau values
def monoexp(a, tau):
    def f(t):
        return a * e**(-t/tau)
    return f

# return timeseries of function f at time_series_steps points
# from 0 up to 3 times the largest time constant
def timeseries(f):
    x = np.linspace(0, 3 * tau_upper, time_series_steps)
    return torch.tensor(np.vectorize(f)(x)).float()

# create plot of monoexponential function and noisy signal data
def plotdata(tseries, params):
    f = monoexp(params[0], params[1])
    x = np.linspace(0, 3 * tau_upper, time_series_steps)
    plt.title("a = " + str(params[0].item()) + ', tau = ' + str(params[1].item()))
    plt.scatter(x, tseries, color='red', s=10)
    plt.plot(x, f(x), color='blue')
    plt.show()

# ---------------------------------- generating synthetic data set ----------------------------------
# generate noise realization
def generate_noise():
    noise_tensor = torch.empty(a_disc_steps, 
                            tau_disc_steps, 
                            noise_realizations, 
                            time_series_steps)
    for i in range(a_disc_steps):
        print(i)
        for j in range(tau_disc_steps):
            for k in range(noise_realizations):
                noise = np.random.normal(0, 1, time_series_steps)
                noise_tensor[i][j][k] = torch.from_numpy(noise)
    print('Done generating noise!')
    return noise_tensor

# generate clean curve timeseries
def generate_curves():
    print('Generating curves...')
    a = np.linspace(a_lower, a_upper, a_disc_steps, dtype='float32')
    tau = np.linspace(tau_lower, tau_upper, tau_disc_steps, dtype='float32')
    t = np.linspace(0, 3 * tau_upper, time_series_steps, dtype='float32')
    curve = a[:, np.newaxis, np.newaxis] * np.exp(-t[np.newaxis, np.newaxis, :] / tau[np.newaxis, :, np.newaxis])
    # replicate timeseries' for multiple noise realizations
    repeated_curve = np.tile(curve, (1, 1, noise_realizations))
    reshaped_curve = np.reshape(repeated_curve, (a_disc_steps, tau_disc_steps, noise_realizations, time_series_steps))
    print('Curve generation complete!')
    return torch.from_numpy(reshaped_curve)   # nn.linear only accepts float32

# generate target output: (a, tau)
def generate_output():
    print('Generating output...')
    a = np.linspace(a_lower, a_upper, a_disc_steps, dtype='float32')
    tau = np.linspace(tau_lower, tau_upper, tau_disc_steps, dtype='float32')
    TAU, A = np.meshgrid(tau, a)
    output = np.column_stack((A.ravel(), TAU.ravel()))
    repeated_output = np.repeat(output, noise_realizations, axis=0)
    reshaped_output = np.reshape(repeated_output, (a_disc_steps, tau_disc_steps, noise_realizations, 2))
    print('Output generation complete!')
    return torch.from_numpy(reshaped_output)

# ---------------------------------- custom Dataset class ----------------------------------
class CurveDataset(Dataset):
    def __init__(self, signal, output):
        self.signal = signal
        self.output = output
    
    def __len__(self):
        return len(self.signal)
    
    def __getitem__(self, idx):
        tseries, params = self.signal[idx], self.output[idx]
        sample = {'tseries': tseries, 'params': params}
        return sample