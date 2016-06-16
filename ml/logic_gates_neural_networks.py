"""A neural network for various logic gates."""

import numpy as np


x1 = np.array([[0, 0, 1, 1]])
x2 = np.array([[1, 0, 1, 0]])
ones_vector = np.array([[1, 1, 1, 1]])
x = np.concatenate((ones_vector, x1, x2))
print 'x1, x2:'
print x1
print x2
print '\n'


def sigmoid(x):
  return 1 / (1 + np.exp(-1 * x))

def hypothesis(weights_vector, x_matrix):
  return sigmoid(np.dot(weights_vector, x_matrix))


weights = np.array([[-30, 20, 20]])
print 'x1 and x2:'
print 'with weights:', weights
print hypothesis(weights, x)
print '\n'

weights = np.array([[-10, 20, 20]])
print 'x1 or x2'
print 'with weights:', weights
print hypothesis(weights, x)
print '\n'

weights = np.array([[10, -20, -20]])
print '(not x1) and (not x2):'
print 'with weights:', weights
print hypothesis(weights, x)
print '\n'
