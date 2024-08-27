# nih-project
Overview:
This repo contains code from my research project at the NIH STARS program. I worked with monoexponential parameter estimation neural networks and their application in myelin water fraction estimation in MRI relaxometry.

Training set generation:
The neural networks are trained on sets of synthetically generated monoexponential curves i.e., curves of the form y=ae^-t/tau. We range over a plausible range of a and tau values in a grid, generating a curve for each a, tau pairing, and adding Gaussian noise to create a simulated signal.

datagen.py contains functions for generating these sets, where the level of discretization (how many steps) of a and tau, as well as the number of noise realizations per pairing can be varied. datasave.py contains code for saving the generated curves to disc.

Neural network training:
Before training the parameter estimation neural networks, hyperparameters (layer size, learning rate, batch size, etc.) were first optimized using RayTune in nn_hpt.py. After fixing said hyperparameters, individual neural networks are trained on a chosen training set using nn_trainer.py.

Analysis:
After training neural networks to completion, we evaluate their performance on a test set (generated using datagen.py and datasave.py) in experiment.py. Results include overall error over the entire test set as well as a heat map depicting error at specific a, tau pairings.
