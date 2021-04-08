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


def test_dig_tol(): #Michael
    mine = Mine(np.array([[1, 2], [3, 4]]))

    # 1D Array
    state1 = ([1, 1, 2, 2, 3, 4, 4])

    #2D Array
    state2 = ([1, 1, 2, 2, 3, 4, 4],
             [1, 1, 2, 3, 4, 5, 6])

    Dig_check = mine.is_dangerous(state1)
    print(Dig_check)

test_dig_tol()