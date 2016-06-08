"""Testing various modules."""

import numpy as np
from numpy.testing import assert_array_equal


def test_numpy_transpose():
  a = np.array([[1, 2, 3]])
  assert_array_equal(np.transpose(a), np.array([(1,),
                                                (2,),
                                                (3,)]))
