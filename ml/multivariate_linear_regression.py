"""Multivariate linear regression."""

import matplotlib.pyplot as plt
import numpy as np


# Create some sample data.
x1 = np.array([[100, 125, 300, 400, 500, 524, 423, 455, 788, 322, 221, 282]])
x2 = np.array([[323, 424, 444, 211, 100, 520, 393, 188, 887, 42,  100, 449]])
x3 = np.array([[100, 900, 800, 770, 232, 100, 400, 448, 22,  100, 300, 200]])
x4 = np.array([[400, 200, 999, 188, 377, 222, 111, 33,  100, 100, 23,  388]])
ones_vector = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
x = np.concatenate((ones_vector, x1, x2, x3, x4))
# Note: y is a row vector.
y = np.array([[10, 11, 31, 41, 53, 49, 44, 48, 70, 30, 22, 28]])


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
  return np.dot(weights_vector, x_matrix)


def cost(weights_vector):
  summation = ((h(weights_vector, x) - y) ** 2).sum()
  samples = x1.shape[1]
  return summation / (2 * samples)


def dJdW(weights_vector):
  delta = h(weights_vector, x) - y
  return np.dot(delta, x.T) / weights_vector.shape[1]


# Create a vector of random weights.
weights = np.random.rand(1, x.shape[0])


# Setup the learning rate and iterate on the weights.
learning_rate = 1e-6
costs = []
while True:
  current_cost = cost(weights)
  print 'cost:', current_cost
  # Improve the weights.
  partial_derivatives = dJdW(weights)
  weights -= learning_rate * partial_derivatives
  if costs and abs(current_cost - costs[-1]) < 1:
    break
  costs.append(current_cost)


# Plot iteration vs error.
plt.plot(np.log(costs), 'rx')
plt.show()
