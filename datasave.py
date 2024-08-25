import torch
from torch.utils.data import random_split
import datagen
import matplotlib.pyplot as plt

# ---------------------------------- noise and curve data generation ----------------------------------
# generate and save noise tensor
noise = datagen.generate_noise()
print('Start downloading...')
# torch.save(noise, 'TRAIN_NOISE_51x51x160.pt')   # move into 'training sets' subdirectory after
print('Download complete!')

# load saved noise
print('Loading noise...')
# noise = torch.load('training sets/TRAIN_NOISE_51x51x160.pt')
print('Loading complete!')

# snr tuning - just multiply entire tensor by sigma
print('SNR tuning...')
snr = 100
sigma = 1/snr   # sigma = a/snr --> average a is 1
noise *= sigma
print('SNR tuning complete!')

# trimming noise tensor to fit curve data - for generating training sets w diff discretization levels
noise = noise[:datagen.a_disc_steps, :datagen.tau_disc_steps, :datagen.noise_realizations]

# generate signal and output data
curves = datagen.generate_curves()
print('Adding noise to curves...')
signals = curves + noise
print('Adding complete!')
output = datagen.generate_output()

# # save signals and output as tensors
# print('Saving signals...')
# torch.save(signals, 'TEST_SIGNALS_1001x1001x100.pt')
# print('Signals saved!')
# print('Saving outputs...')
# torch.save(output, 'TEST_OUTPUT_1001x1001x100.pt')
# print('Output saved!')

# ---------------------------------- implementing Dataset class ----------------------------------
# flatten signals and output
signals = torch.flatten(signals, end_dim=2)
output = torch.flatten(output, end_dim=2)

# split dataset into train, test, and validation, then save
# curvedata = datagen.CurveDataset(signals, output)
# train_dataset, val_dataset, test_dataset = random_split(curvedata, [0.6, 0.2, 0.2])  # 60% training, 20% validation, 20% test
# torch.save(train_dataset, 'train.pt')
# torch.save(val_dataset, 'val.pt')
# torch.save(test_dataset, 'test.pt')

# save whole dataset
print('Saving dataset...')
data = datagen.CurveDataset(signals, output)
torch.save(data, f'TRAIN_SET_{datagen.a_disc_steps}x{datagen.tau_disc_steps}x{datagen.noise_realizations}.pt')  # move into 'training sets' subdirectory after
print('Saving complete!')