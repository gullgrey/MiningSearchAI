from mining import *
from search import *
import numpy as np


#3D mine
x = np.array(    [[[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [1, 2]],
                  [[1, 2], [1, 2], [-7, 2]]])

#2D mine
y = np.array([[1, 2, -5, -5, -5, -5],
              [1, 2, -5, -5, -5, -5],
              [1, 2, -5, -5, -5, 100]])

z = np.array([[[-2, 2], [3, -4]],
            [[1, 2], [3, 4]],
              [[-2, 2], [3, -4]]])

x_state = ((1,0,0),
           (0,2,0),
           (0,0,0),
           (0,0,0))

x_state2 = ((1,0,0),
            (0,2,0),
            (0,1,0),
            (2,0,0))

x_state3 = ((1,0,2),
            (0,2,0),
            (0,1,0),
            (2,0,0))

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
    # print(mine)
    print(mine.cumsum_mine)
    print(mine.initial)


def test_dig_tol(): #Michael
    mine = Mine(np.array([[1, 2], [3, 4]]))

    Dig_check = mine.is_dangerous(state1)
    print(Dig_check)


def test_actions():
    mine = Mine(np.array(x), 2)
    # mine.len_x = 2
    # mine.len_y = 7
    a = mine.actions(((1.0, 1.0, 1.0), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
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


def test_dp_dig_plan():
    mine = Mine(np.array(x))
    result = search_dp_dig_plan(mine)
    print(result)
    print(mine)


# def test_bb_priority_queue():
#     mine = Mine(np.array(x))
#     bb_aux = BbAuxiliary(mine)
#     bb_aux.priority_queue.append(np.array(x_state3))
#     bb_aux.priority_queue.append(x_state2)
#     bb_aux.priority_queue.append(x_state)
#     while bb_aux.priority_queue:
#         print(bb_aux.priority_queue.pop())

def test_search_bb_dig_plan():
    mine = Mine(np.array(y))
    print(search_bb_dig_plan(mine))

if __name__ == '__main__':
    separator_string = '\n------------------------------------------------------------\n '

    test_1()
    print(separator_string)
    test_dig_tol()
    print(separator_string)
    test_actions()
    print(separator_string)
    test_find_action_sequence()
    print(separator_string)
    test_dp_dig_plan()
    # print(separator_string)
    # test_bb_priority_queue()
    print(separator_string)
    test_search_bb_dig_plan()

