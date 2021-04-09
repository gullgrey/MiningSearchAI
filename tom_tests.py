from mining import *
from search import *
import numpy as np


#3D mine
x = np.array(    [[[1, 2], [1, 2], [5, -205]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]]])

#2D mine
y = np.array([[1, 2],
              [3, 4],
              [5, 6]])

x_state = ((1,0,0),
           (0,2,0),
           (0,0,0),
           (0,0,0),)

y_state = ((0,0,0))

# 1D Tuple
state1 = (1, 1, 2, 2, 3, 4, 4)

#2D Tuple
state2 = ((1, 1, 2, 2, 3, 4, 4),
          (1, 1, 2, 3, 4, 5, 6))

state3 = (0, 0, 2, 1, 2, 2, 2)

state4 = ((0, 0, 2, 1, 2, 2, 2),
          (0, 0, 2, 1, 2, 2, 2))

def test_1():

    mine = Mine(np.array(x))
    print(mine)
    print(mine.cumsum_mine)
    print(mine.initial)


def test_dig_tol(): #Michael
    mine = Mine(np.array([[1, 2], [3, 4]]))



    Dig_check = mine.is_dangerous(state1)
    print(Dig_check)


def test_actions():
    mine = Mine(np.array(x), 2)
    mine.len_x = 2
    mine.len_y = 7
    a = mine.actions(state2)
    for value in a:
        print(value)
    # print(next(a), next(a))

def test_find_action_sequence():

    actions = find_action_sequence(state4, state2)
    print(actions)
    mine = Mine(x)
    state = state4
    for action in actions:
        state = mine.result(state, action)
    print(state)

if __name__ == '__main__':
    separator_string = '\n------------------------------------------------------------\n '

    test_1()
    print(separator_string)
    test_dig_tol()
    print(separator_string)
    test_actions()
    print(separator_string)
    test_find_action_sequence()

