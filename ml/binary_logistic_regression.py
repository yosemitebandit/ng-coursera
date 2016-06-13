"""Logistic Regression -- binary classification."""

import matplotlib.pyplot as plt
import numpy as np


# A donut dataset centered around the origin.
x1 = [-3, -2, -1,  0, 1, 2, 3, -30, -20, -10, 0,  10,  20,  30, 15, -10, 0]
x2 = [-2, -4, -1, -1, 3, 1, 2, -25,  30,  10, 30, 40, -20, -10, 15, -20, -20]
r = [(x1[i]**2 + x2[i]**2) ** 0.5 for i in range(len(x1))]
ones_vector = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x1 = np.array([x1])
x2 = np.array([x2])
r = np.array([r])
ones_vector = np.array([ones_vector])
# Note x is (4 x 17).
x = np.concatenate((ones_vector, x1, x2, r))

y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Note y is (1 x 17).
y = np.array([y])
samples = x1.shape[1]


def sigmoid(x):
  return 1 / (1 + np.exp(-1 * x))


def h(weights_vector, x_matrix):
  """The hypothesis function.

  Args:
    weights_vector: [W0, W1, W2, W3, .. ]
    x_matrix: [1, 1, 1, ..
               x1, ..
               x2, ..
               ..          ]
  Returns:
    row vector, 1 x n
  """
  return sigmoid(np.dot(weights_vector, x_matrix))


def logistic_cost(weights_vector):
  summation = (np.dot(y, np.log(h(weights_vector, x)).T) +
               np.dot((1 - y), np.log(1 - h(weights_vector, x).T)))
  return -1 * summation.flatten()[0] / samples


def dJdW(weights_vector):
  delta = h(weights_vector, x) - y
  return np.dot(delta, x.T) / samples


# Create a vector of random weights.
weights = np.random.rand(1, x.shape[0])
initial_weights = np.copy(weights)


# Setup the learning rate and iterate on the weights.
learning_rate = 1e-5
iterations = 1e5
costs = []
while True:
  costs.append(logistic_cost(weights))
  # Improve the weights.
  partial_derivatives = dJdW(weights)
  weights -= learning_rate * partial_derivatives
  iterations -= 1
  if iterations == 0:
    break


# Plot cost v iteration.
plt.plot(costs)
plt.show()


# Plot the dataset.
for i in range(len(x1[0])):
  if y[0][i] == 1:
    marker = 'go'
  elif y[0][i] == 0:
    marker = 'rx'
  plt.plot(x1[0][i], x2[0][i], marker)
plt.show()


print initial_weights
print weights
