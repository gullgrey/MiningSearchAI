from mining import *
from search import *
import numpy as np

def test_dig_tol(): #Michael
    mine = Mine(np.array([[1, 2], [3, 4]]))



    Dig_check = mine.is_dangerous(state1)
    print(Dig_check)