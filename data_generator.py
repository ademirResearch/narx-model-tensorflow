#%%
# Data generator script. Creates a csv file containing the experiments from the selected system

# 1. Instantiate desired system class
# 2. Select excitation signal type and parameters
# 3. Run experiment, creates a dataset file csv (inputs, outputs) as columns

from systems import FirstOrder, Lorenz, Nonlinear
from excitation_signals import FilteredGaussianWhiteNoise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

system = Nonlinear()
signal = FilteredGaussianWhiteNoise()
excitation_signal = signal.get_signal(samples=350000, mean=0.0, std=10.0)

# Plot a signal example
plt.plot(excitation_signal[:200])
plt.grid()
plt.show()

# Perform experiment
train_filename = "train_nonlinear"
test_filename = "test_nonlinear"
experiment_result = system.experiment(u=excitation_signal, x0=system.x0_vector, ts=0.02)

# Reshape into number of columns

data = np.hstack((excitation_signal.reshape(-1, 1), experiment_result.reshape(-1, system.num_states)))

df = pd.DataFrame(data, columns=system.column_names)
df.to_csv("datasets/{}.csv".format(train_filename), index=False)


# Test data
test_signal = np.ones(600)
test_result = system.experiment(u=test_signal, x0=system.x0_vector, ts=0.02)

data = np.hstack((test_signal.reshape(-1, 1), test_result.reshape(-1, system.num_states)))

df = pd.DataFrame(data, columns=system.column_names)
df.to_csv("datasets/{}.csv".format(test_filename), index=False)
# %%
