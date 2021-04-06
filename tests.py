from mining import *
from search import *
import numpy as np

x = np.array([[[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]]])

y = [[1,2],[3,4],[5,6]]

def test_1():

    mine = Mine(np.array(x))
    print(mine)
    print(mine.cumsum_mine)
    print(mine.initial)


test_1()
# print(x.cumsum(axis = 2))