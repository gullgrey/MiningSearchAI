from mining import *
from search import *
import numpy as np

#3D mine
x = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
              [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
#2D mine
y = np.array([[1, 2],
              [3, 4],
              [5, 6]])


def test_dig_tol(): #Michael
    mine = Mine(np.array([[1, 2], [3, 4]]))

    #1D Tuple
    state1 = (1, 1, 2, 2, 3, 2, 1)

    # 2D Tuple
    state2 = ((1, 1, 1, 1, 1, 1, 1),
              (1, 2, 2, 2, 2, 2, 1),
              (1, 2, 2, 3, 2, 2, 1),
              (1, 2, 2, 2, 2, 2, 1),
              (1, 1, 1, 1, 1, 1, 1))

    Dig_check = mine.is_dangerous(state1)
    print(Dig_check)

def test_payoff():
    mine = Mine(np.array(x))

    # 2D Tuple
    state2 = ((2, 2, 2),
              (2, 2, 2),
              (2, 2, 2),
              (2, 2, 2))

    payoff = mine.payoff(state2)
    print(payoff)

if __name__ == '__main__':
    separator_string = '\n------------------------------------------------------------\n '

    test_dig_tol()
    test_payoff()
    print("tests complete")


