import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


# our model for the forward pass
def forward(x):
  return x * w


def loss(x, y):
  y_pred = forward(x)
  return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []


for w in np.arange(0.0, 4.1, 0.1):
  print("w = ", w)

  l_sum = 0

  for x_val, y_val in zip(x_data, y_data):
    # For each input and output, calculate y_hat
    # Compute the total loss and add to the total error
    y_pred_val = forward(x=x_val)
    l = loss(x=x_val, y=y_val)
    l_sum += l
    print("\t", x_val, y_val, y_pred_val, l)

  # Now compute the Mean squared error (mse) of each
  # Aggregate the weight/mse from this run
  print("MSE = ", l_sum / len(x_data))
  w_list.append(w)
  mse_list.append(l_sum / len(x_data))

# plot it all
plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()
