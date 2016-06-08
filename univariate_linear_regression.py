"""Univariate linear regression."""

import random

import matplotlib.pyplot as plt


# Create some sample data.
x = [100, 125, 300, 400, 500, 524, 423, 455, 788, 322, 221, 282, 448, 988, 431]
y = [10, 11, 31, 41, 53, 49, 44, 48, 70, 30, 22, 28, 45, 90, 43]


def h(x_value, W, b):
  # h(x) = Wx + b
  return W * x_value + b


def cost(W, b):
  # J(W, b) = 1/2m * sum((h(x) - y) ** 2)
  cost = 0
  for i in range(len(x)):
    x_value = x[i]
    y_value = y[i]
    cost += (h(x_value, W, b) - y_value) ** 2
  return cost / (2 * len(x))


# Define partial derivatives.
def dJdW(W, b):
  cumulative_sum = 0
  for i in range(len(x)):
    x_value = x[i]
    y_value = y[i]
    cumulative_sum += (h(x_value, W, b) - y_value) * x_value
  return cumulative_sum / (len(x))


def dJdb(W, b):
  cumulative_sum = 0
  for i in range(len(x)):
    x_value = x[i]
    y_value = y[i]
    cumulative_sum += h(x_value, W, b) - y_value
  return cumulative_sum / (len(x))


# Take a random guess at W and b and setup the learning rate.
W = random.random() * 100 * random.choice([-1, 1])
b = random.random() * 10 * random.choice([-1, 1])
learning_rate = 1e-6
costs = []
while True:
  # Plot the guess and print the cost.
  x_sample = range(0, 1000, 10)
  y_sample = [h(x_val, W, b) for x_val in x_sample]
  plt.plot(x, y, 'rx', x_sample, y_sample, 'b-')
  plt.xlim(0, 1000)
  plt.ylim(0, 100)
  plt.show()
  current_cost = cost(W, b)
  print 'cost:', current_cost
  # Improve the guess.
  delta_W = learning_rate * dJdW(W, b)
  delta_b = learning_rate * dJdb(W, b)
  W -= delta_W
  b -= delta_b
  if costs and abs(current_cost - costs[-1]) < 1:
    break
  costs.append(current_cost)

plt.plot(costs)
plt.show()
