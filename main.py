#%%

from systems import FirstOrder, Lorenz, Nonlinear
from window_generator import WindowGenerator, tf, compile_and_fit
from autoregressive_model import FeedBack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


t = np.arange(0, 10, 0.02)
u = np.ones_like(t) * 10
tau = np.ones_like(u) * 0.63

system = Nonlinear()
response = system.experiment(u=u, ts=0.02, x0=system.x0_vector)

plt.plot(t, response)
# plt.plot(t, tau, "--")
plt.grid()
plt.show()

# Read dataset
df = pd.read_csv("datasets/train_nonlinear.csv")
df = df.drop(['x'], axis=1)

# Split dataset
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
print(train_df.head())

# Scale data

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Create window and examples
w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['y'], train_df=train_df, val_df=val_df, test_df=test_df)
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
# w2.example = example_inputs, example_labels

w2.plot(plot_col="y")

# Train dense model
single_step_window = WindowGenerator(
    input_width=20, label_width=1, shift=1,
    label_columns=['y'], train_df=train_df, val_df=val_df, test_df=test_df)
val_performance = {}
performance = {}

MAX_EPOCHS = 20

dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)



# Train autoregressive model
# 16 Units -> First order
feedback_model = FeedBack(units=64, out_steps=1)
history = compile_and_fit(feedback_model, single_step_window)


val_performance['AR LSTM'] = feedback_model.evaluate(single_step_window.val)
performance['AR LSTM'] = feedback_model.evaluate(single_step_window.test, verbose=0)
single_step_window.plot(feedback_model, plot_col="y")


# Performance plot
x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_name = 'mean_squared_error'
metric_index = dense.metrics_names.index('mean_squared_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error y, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()
plt.show()

# Test generative model
# Read test dataset
df = pd.read_csv("datasets/test_nonlinear.csv")
# df = df.drop(['x'], axis=1)

test_mean = df.mean()
test_std = df.std()

df = (df - train_mean) / train_std


u_ = (1.0 - train_mean["u"]) / train_std["u"]

nn_input = np.zeros((20, num_features))
nn_input[-1, 0] = u_

result = list()
for j in range(600):
    to_nn = tf.convert_to_tensor(nn_input.reshape(-1, 20, num_features))
    raw_y = feedback_model.call(to_nn)
    y = raw_y.numpy().flatten()
    nn_input = np.roll(nn_input, -1, axis=0)
    nn_input[-1, 0] = u_
    nn_input[-1, 1] = y[-1]
    result.append(y[-1])

df["y_hat"] = result
print(df.head())

df.plot()
plt.grid()
plt.show()


    # %%


# %%
