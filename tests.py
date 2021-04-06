from mining import *
from search import *
import numpy as np


def test_1():
    mine = Mine(np.array([[1,2],[3,4]]))
    print(mine)


test_1()


def test_dig_tol():
    mine = Mine(np.array([[1,2],[3,4]]))
    state = [1, 1, 2, 2, 2, 3, 7]
    Dig_check = mine.is_dangerous(state)
    print(Dig_check)

test_dig_tol()