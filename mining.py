#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from itertools import count

from numbers import Number

import search

def my_team():
    '''

    Returns
    -------
    A list of the team members of this assignment submission as a list of triplet
    of the form (student_number, first_name, last_name)

    '''
    return [(10895159, 'Thomas', 'Cleal'), (10583084, 'Michael', 'Solomon'), (10154337, 'Clancy', 'Haupt')]


def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the nparray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''
        # super().__init__() # call to parent class constructor not needed
        
        self.underground = underground 
        # self.underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2,3)

        self.len_x = None
        self.len_y = 1
        self.len_z = None

        self.cumsum_mine = None
        self.initial = None

        self.dimensions = None

        self._set_attributes()

    def _set_attributes(self):
        '''
        TODO add description

        Returns
        -------
        None.
        '''
        self.len_x = self.underground.shape[0]
        if self.underground.ndim == 2:
            self.len_z = self.underground.shape[1]

            initial_array = np.zeros(self.underground.shape[0], dtype=int)
        else:
            self.len_y = self.underground.shape[1]
            self.len_z = self.underground.shape[2]

            state_dimensions = (self.underground.shape[0], self.underground.shape[1])
            initial_array = np.zeros(state_dimensions, dtype=int)

        self.initial = convert_to_tuple(initial_array)

        cumsum = self.underground.cumsum(axis=self.underground.ndim - 1)

        # This inserts a zero at the first column of every row of the cumulative sum,
        # to represent a block before it has been mined.
        self.cumsum_mine = np.insert(cumsum, 0, 0, axis=self.underground.ndim - 1)

    def surface_neighbours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine

        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
     
    
    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''        
        state = np.array(state)
        assert state.ndim in (1, 2)

        valid_actions = []

        if state.ndim == 1:
            # generates every loc for a 1 dimensional state
            coordinates = ((x,) for x in range(self.len_x))
        else:
            # generates every loc for a 2 dimensional state
            coordinates = ((x, y) for x in range(self.len_x) for y in range(self.len_y))

        for coordinate in coordinates:
            neighbours = self.surface_neighbours(coordinate)
            within_tolerance = True
            for neighbour in neighbours:
                if (state[coordinate] - state[neighbour]) >= self.dig_tolerance:
                    within_tolerance = False
            if within_tolerance and (state[coordinate] < self.len_z):
                valid_actions.append(coordinate)

        return (action for action in valid_actions)
  
    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        # does this have to check validity
        action = tuple(action)
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)
                
    
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[...,z]) for z in range(self.len_z))

    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        
        No loops needed in the implementation!        
        '''
        # convert to np.array in order to use tuple addressing
        # state[loc]   where loc is a tuple

        state = np.array(state)
        if state.ndim == 1:
            x_coordinates = np.arange(0, self.len_x)

            return np.sum(self.cumsum_mine[x_coordinates, state])

        else:
            x_coordinates, y_coordinates = np.indices((self.len_x, self.len_y))
            x_coordinates = x_coordinates.flatten()
            y_coordinates = y_coordinates.flatten()

            return np.sum(self.cumsum_mine[x_coordinates, y_coordinates, state.flatten()])



        # state = np.array(state)
        #
        # total_payoff = 0                                               # initialise total payoff as 0
        #
        #
        # #2D Mine
        # if state.ndim == 1:                                            # 1D Array (x)
        #     rows = np.size(state, 0)                                   # length of rows
        #     for i in range(0, rows):                                   # get i & j index to do neighbour check on
        #         depth = 0                                              # starting z co-ordinate
        #         while depth != state[i]:                           # mine in single column until reaching "state depth"
        #             total_payoff += self.underground[i, depth]         # add each z value into total
        #             depth += 1                                         # mine down the column
        #     return total_payoff
        #
        # #3D Mine
        # elif state.ndim == 2:                                          # 2D Array (x,y)
        #     columns = np.size(state, 1)                                # length of columns
        #     rows = np.size(state, 0)                                   # length of rows
        #     for i in range(0, rows):                                   # get i & j index to do neighbour check on
        #         for j in range(0, columns):
        #             depth = 0                                          # starting z co-ordinate
        #             while depth != state[i, j]:                        # mine in single column until reaching "state depth"
        #                 total_payoff += self.underground[i, j, depth]  # add each z value into total
        #                 depth += 1                                     # mine down the column
        #     return total_payoff


    def _roll_compare(self, index, axis, diagonal, state):
        '''
        TODO add description
        Parameters
        ----------
        index
        axis
        diagonal
        state

        Returns
        -------
        TODO add returns
        '''
        if (index, axis) == (0, 1):
            roll_direction, roll_axis = 1, 1
        elif (index, axis) == (0, 0):
            roll_direction, roll_axis = 1, 0
        else:
            roll_direction, roll_axis = -1, 1

        if diagonal:
            compare_state = np.roll(np.roll(state, 1, axis=0), roll_direction, axis=roll_axis)
            trimmed_state = np.delete(np.delete(state, 0, 0), index, axis)
            trimmed_compare = np.delete(np.delete(compare_state, 0, 0), index, axis)
        else:
            compare_state = np.roll(state, roll_direction, axis=roll_axis)
            trimmed_state = np.delete(state, index, axis)
            trimmed_compare = np.delete(compare_state, index, axis)

        # These lines change the elements of the compared array to be the same as
        # the elements of the original array if they equal -1.
        # -1 represents an unassigned block of a state in the BB algorithm.
        unassigned_compare = (trimmed_compare < 0)
        trimmed_compare[unassigned_compare] = trimmed_state[unassigned_compare]
        unassigned_state = (trimmed_state < 0)
        trimmed_state[unassigned_state] = trimmed_compare[unassigned_state]

        # compares values in shifted array to original state
        # returns true if difference is greater then the dig tolerance.
        if (abs(trimmed_state - trimmed_compare) > self.dig_tolerance).any():
            return True
        return False

    def is_dangerous(self, state):
        '''
        Return True if the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''

        # convert to np.array in order to use numpy operators
        state = np.array(state)

        # check whether 2d or 3d array
        assert state.ndim in (1, 2)

        if state.ndim == 1:
            if self._roll_compare(0, 0, False, state):
                return True

        else:
            # Shift state right and compare values
            if (self._roll_compare(0, 1, False, state) or
                    # Shift state down right and compare values
                    self._roll_compare(0, 1, True, state) or
                    # Shift state down and compare values
                    self._roll_compare(0, 0, False, state) or
                    # Shift state down left compare values
                    self._roll_compare(-1, 1, True, state)):
                return True
        return False


        # if state.ndim == 1:                                                             #1D Array (x)
        #     for i, j in enumerate(state[:-1]):                                          #run through all values in list
        #         if abs(j - state[i + 1]) > self.dig_tolerance:                          #if absolute difference greater than dig tolerance
        #             return True
        #
        # elif state.ndim == 2:                                                           #2D Array (x,y)
        #     rows = np.size(state, 0)                                                    #length of rows
        #     columns = np.size(state, 1)                                                 #length of columns
        #     for i in range(0, rows):                                                    #get i & j index to do neighbour check on
        #         for j in range(0, columns):
        #             # print(str(i) + ", " + str(j) +  ": " + str(state[i,j]) +"\n")
        #             for r in range(i-1, i+2):                                           #cycle through range 1 above & below of row
        #                 for c in range(j-1, j+2):                                       #cycle through range 1 above & below of column
        #                     if 0 <= r < rows and 0 <= c < columns:                      #ensure not out of bounds
        #                         if abs(state[i, j] - state[r, c]) > self.dig_tolerance: #get value of current position (i,j) and minus it from current (r,c) neighbour
        #                             return True
        # return False

           


    
    # ========================  Class Mine  ==================================


class DpAuxiliary:
    """
    TODO add class description
    """

    def __init__(self, mine):
        """
        TODO add constructor description
        Parameters
        ----------
        mine
        """
        self.mine = mine

        # nodes in the search tree that have already been computed
        self.cached_nodes = {}

    def dp_recursive(self, best_payoff, best_action_list, best_final_state):
        """
        TODO add description
        Parameters
        ----------
        best_payoff
        best_action_list
        best_final_state

        Returns
        -------
        TODO add returns
        """
        best_dig = (best_payoff, best_action_list, best_final_state)

        actions = self.mine.actions(best_final_state)

        for action in actions:
            next_state = self.mine.result(best_final_state, action)

            # Checks to see if node has already been computed
            if next_state in self.cached_nodes:
                next_dig = self.cached_nodes[next_state]
            else:
                next_dig = self.dp_recursive(self.mine.payoff(next_state),
                                             best_action_list + [action],
                                             next_state)

                # adds computed node to a dictionary
                self.cached_nodes[next_state] = next_dig

            # compares the next dig on the frontier and updates if payoff is greater
            if next_dig[0] > best_dig[0]:
                best_dig = next_dig
        return best_dig


def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    initial_state = mine.initial
    initial_payoff = 0
    initial_action_list = []
    dp_auxiliary = DpAuxiliary(mine)
    return dp_auxiliary.dp_recursive(initial_payoff, initial_action_list, initial_state)


class BbAuxiliary:
    """
    TODO add class description
    """
    def __init__(self, mine):
        """
        TODO add constructor description
        Parameters
        ----------
        mine
        """
        self.mine = mine
        self.best_so_far = mine.initial
        self.priority_queue = search.PriorityQueue(order='max', f=lambda x: self.mine.payoff(x[1]))
        self.operations_counter = count()
        dimensions = self.mine.cumsum_mine.ndim - 1
        self.upper_bound = self.mine.cumsum_mine.argmax(dimensions)
        self.test_counter = count()

    def bb_search_tree(self, state):

        while np.any(state < 0):

            # coordinates of the next block that has not been assigned a dig.
            if self.mine.underground.ndim == 2:
                x = np.argmax(state < 0)
                coordinates = (x,)
            else:
                (x, y) = np.unravel_index(np.argmax(state < 0), state.shape)
                coordinates = (x, y)

            for z in range(self.mine.len_z + 1):
                frontier_state = np.copy(state)
                frontier_state[coordinates] = z
                if self.mine.is_dangerous(frontier_state):
                    continue

                unassigned_dig = (frontier_state < 0)
                compare_state = np.copy(frontier_state)
                compare_state[unassigned_dig] = self.upper_bound[unassigned_dig]

                if self.mine.payoff(compare_state) > self.mine.payoff(self.best_so_far):
                    self.priority_queue.append((next(self.operations_counter), compare_state, frontier_state))

            if not self.priority_queue.heap:
                print(state)
                return self.best_so_far

            state = self.priority_queue.pop()[2]
        return state

    def bb_solution_candidates(self, state):
        solution_candidate = self.bb_search_tree(state)

        new_queue = search.PriorityQueue(order='max', f=lambda x: self.mine.payoff(x[1]))
        while self.priority_queue:
            node = self.priority_queue.pop()
            if self.mine.payoff(node[1]) <= self.mine.payoff(solution_candidate):
                new_queue.append(node)
        if not new_queue.heap:
            return solution_candidate
        else:
            self.best_so_far = solution_candidate
            return self.bb_solution_candidates(solution_candidate)


def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    
    state = np.array(mine.initial) - 1
    bb_auxiliary = BbAuxiliary(mine)
    best_final_state = bb_auxiliary.bb_solution_candidates(state)

    best_final_state = convert_to_tuple(best_final_state)
    best_payoff = mine.payoff(best_final_state)
    best_action_list = find_action_sequence(mine.initial, best_final_state)

    return best_payoff, best_action_list, best_final_state


def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''    
    # approach: among all columns for which s0 < s1, pick the column loc
    # with the smallest s0[loc]
    s0 = np.array(s0)
    s1 = np.array(s1)

    assert s0.ndim == s1.ndim and s0.ndim in (1, 2)

    action_sequence = []
    while np.any(s1 - s0):

        flat_s0 = s0.flatten()
        flat_s0 = list(dict.fromkeys(flat_s0))
        flat_s0.sort()
        for depth_value in flat_s0:
            coordinate_list = np.where(s0 == depth_value)
            if s0.ndim == 2:
                min_coordinates = list(zip(coordinate_list[0], coordinate_list[1]))
            else:
                min_coordinates = coordinate_list[0]
                min_coordinates = ((coordinate,) for coordinate in min_coordinates)
            for coordinate in min_coordinates:
                if s1[coordinate] != s0[coordinate]:
                    action_sequence.append(coordinate)
                    s0[coordinate] += 1

    return action_sequence


