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
        self.two_dimensions: Boolean : True if the mine is two dimensional.
        self.x_coordinates, self.y_coordinates : numpy array : Used as the x and y
            coordinates when indexing the cumulative sum.
    
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
        self.cumsum_mine, self.initial,
        self.two_dimensions, self.x_coordinates, and self.y_coordinates
        
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

        # Default is 1 for a 2D mine.
        self.len_y = 1
        self.len_z = None

        self.cumsum_mine = None
        self.initial = None

        # Default is True for a 2D mine.
        self.two_dimensions = True
        self.x_coordinates = None
        self.y_coordinates = None

        self._set_attributes()

    def _set_attributes(self):
        '''
        Sets all of the class attributes in the constructor when a new mine class
            is first initialised.
        Based off the inputted underground and dig_tolerance.

        Returns
        -------
        None.
        '''
        x_index, y_index, z_index = 0, 1, -1
        two_dimensions = 2
        self.len_x = self.underground.shape[x_index]
        if self.underground.ndim == two_dimensions:

            # For a 2D mine.
            self.len_z = self.underground.shape[z_index]
            initial_array = np.zeros(self.len_x, dtype=int)

            # The x coordinates of the mine used in the payoff function.
            self.x_coordinates = np.arange(self.len_x)
        else:

            # For a 3D mine
            self.len_y = self.underground.shape[y_index]
            self.len_z = self.underground.shape[z_index]
            initial_array = np.zeros((self.len_x, self.len_y), dtype=int)
            self.two_dimensions = False

            # The x and y coordinates of the mine used in the payoff function.
            self.x_coordinates, self.y_coordinates = np.indices((self.len_x, self.len_y))
            self.x_coordinates = self.x_coordinates.flatten()
            self.y_coordinates = self.y_coordinates.flatten()

        self.initial = convert_to_tuple(initial_array)

        # This inserts a zero at the first column of every row of the cumulative sum,
        # to represent a block before it has been mined.
        cumsum = self.underground.cumsum(axis=self.underground.ndim - 1)
        position, value = 0, 0
        self.cumsum_mine = np.insert(cumsum, position, value, axis=self.underground.ndim - 1)

    def surface_neigbhours(self, loc):
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
            represented with nested lists, tuples or a numpy array
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions

        '''        
        state = np.array(state)
        assert state.ndim in (1, 2)

        def coordinates():
            '''
            Return a generator of tuple coordinates in the state named in the outer function.
            A coordinate is represented as a singleton (x,) in case of a 2D mine,
                and a pair (x,y) in case of a 3D mine.

            Returns
            -------
            A generator of every coordinate in the given state.
            '''
            if self.two_dimensions:

                # generates every coordinate for a 1 dimensional state
                return ((x,) for x in range(self.len_x))
            else:

                # generates every coordinate for a 2 dimensional state
                return ((x, y) for x in range(self.len_x) for y in range(self.len_y))

        def valid_action(coordinate):
            '''
            Determines whether the inputted coordinate would break dig tolerance compared
                to its neighbours if a dig took place there, based off the state named
                in the outer function.

            Parameters
            ----------
            coordinate:
                Tuple surface coordinates of a cell.
                A singleton (x,) in case of a 2D mine.
                A pair (x,y) in case of a 3D mine.

            Returns
            -------
            A Boolean: False if digging at the coordinate would exceed the dig tolerance
                compared to its neighbours.
            '''
            if state[coordinate] >= self.len_z:
                return False
            neighbours = self.surface_neigbhours(coordinate)

            # Compares each coordinate value with its neighbours.
            for neighbour in neighbours:
                if (state[coordinate] - state[neighbour]) >= self.dig_tolerance:
                    return False
            return True

        # The final generator of valid actions.
        return (coordinate for coordinate in coordinates() if valid_action(coordinate))
  
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
        assert state.ndim in (1, 2)

        # Both sum together every value in cumulative sum at the indexes of the given state.
        if self.two_dimensions:

            # 2D mine.
            return np.sum(self.cumsum_mine[self.x_coordinates, state])
        else:

            # 3D mine.
            return np.sum(self.cumsum_mine[self.x_coordinates, self.y_coordinates, state.flatten()])

    def _roll_compare(self, direction, state):
        '''
        Compares a state with a copy of the state rolled in one of four directions:
            RIGHT, DOWN, DOWN-RIGHT, DOWN-LEFT.
        If any of the compared values exceed the dig tolerance of the mine, then return True.
        Testing all four directions will indicate if a mine is dangerous.
        If the state has any unassigned digs (represented as -1) then those digs will be considered
            not breaking dig tolerance.

        Preconditions:
            state is a 1D or 2D numpy array
            direction is a string that is any of:
                'RIGHT', 'DOWN', 'DOWN-RIGHT', 'DOWN-LEFT'

        Parameters
        ----------
        direction:
            A string representing the direction the array is shifted before it is compared.
        state :
            Represented with a numpy array.
            State of the partially dug mine.

        Returns
        -------
        A Boolean: True if the mine breaks the dig tolerance in the rolled direction.
        '''
        assert state.ndim in (1, 2) and direction in ('RIGHT', 'DOWN', 'DOWN-RIGHT', 'DOWN-LEFT')
        shift_down = False

        # Setting up the rolling and deletion, indexes and axes.
        if direction == 'RIGHT':
            trim_index, trim_axis = 0, 1
            roll_direction, roll_axis = 1, 1
        elif direction == 'DOWN':
            trim_index, trim_axis = 0, 0
            roll_direction, roll_axis = 1, 0
        elif direction == 'DOWN-RIGHT':
            trim_index, trim_axis = 0, 1
            roll_direction, roll_axis = 1, 1
            shift_down = True
        else:  # direction == 'DOWN-LEFT'
            trim_index, trim_axis = -1, 1
            roll_direction, roll_axis = -1, 1
            shift_down = True

        if shift_down:
            down_trim_index, down_trim_axis = 0, 0
            roll_down, down_axis = 1, 0

            # First shifts the compared array down before shifting left or right.
            compare_state = np.roll(np.roll(state, roll_down, axis=down_axis), roll_direction, axis=roll_axis)

            # The two states are trimmed so that the edge values aren't compared to the
            # opposite edges.
            trimmed_state = np.delete(np.delete(state, down_trim_index, down_trim_axis), trim_index, trim_axis)
            trimmed_compare = np.delete(np.delete(compare_state, down_trim_index, down_trim_axis), trim_index, trim_axis)
        else:

            # First shifts the compared array down before shifting left or right.
            compare_state = np.roll(state, roll_direction, axis=roll_axis)

            # The two states are trimmed so that the edge values aren't compared to the
            # opposite edges.
            trimmed_state = np.delete(state, trim_index, trim_axis)
            trimmed_compare = np.delete(compare_state, trim_index, trim_axis)

        # These lines change the elements of the compared array to be the same as
        # the elements of the original array if they equal -1.
        # -1 represents an unassigned block of a state in the BB algorithm.
        unassigned_dig = -1
        unassigned_compare = (trimmed_compare == unassigned_dig)
        trimmed_compare[unassigned_compare] = trimmed_state[unassigned_compare]
        unassigned_state = (trimmed_state == unassigned_dig)
        trimmed_state[unassigned_state] = trimmed_compare[unassigned_state]

        # compares values in shifted array to original state
        # returns true if any difference is greater then the dig tolerance.
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

        # check if 1d or 2d array
        assert state.ndim in (1, 2)

        if self.two_dimensions:

            # Shift state across and compare values
            return self._roll_compare('DOWN', state)
        else:

            # Shift states in indicated directions and compare values
            if (self._roll_compare('RIGHT', state) or
                    self._roll_compare('DOWN-RIGHT', state) or
                    self._roll_compare('DOWN', state) or
                    self._roll_compare('DOWN-LEFT', state)):
                return True
        return False

    # ========================  Class Mine  ==================================


class DpAuxiliary:
    """
    A class to help compute the best payoff using Dynamic Programming.
    It uses a dictionary to cache nodes of a recursive search of a mine's
        underground, as a way of memoizing them.
    """

    def __init__(self, mine):
        """
        Constructor

        Initializes the attributes:
            self.mine:
                An instance of the mine class.
            self.cached_nodes:
                A dictionary cache used for memoization.

        Parameters
        ----------
        mine:
            An instance of the Mine class.

        Returns
        -------
        None.
        """
        self.mine = mine

        # nodes in the search tree that have already been computed
        self.cached_nodes = {}

    def dp_recursive(self, best_payoff, best_action_list, best_final_state):
        """
        Recursively checks every possible state within dig tolerance of the
            underground in self.mine.
        Each state is memoized in a dictionary "self.cashed_nodes".
        Each new state on the frontier is first checked to see if it in the dictionary
            and if so, that value is used rather then recursively computing its payoff
            again.
        Outputs the state that gives the best possible payoff, as well as the payoff itself
            and the list of actions it took to get there.

        Parameters
        ----------
        best_payoff:
            An integer representing the payoff of the current best_final_state.
        best_action_list:
            A list of actions represented as tuple coordinates to get from the initial
            state to the current best_final_state.
        best_final_state:
            A numpy array representing the current state being checked.

        Returns
        -------
        best_payoff, best_action_list, best_final_state
        """
        best_dig = (best_payoff, best_action_list, best_final_state)
        actions = self.mine.actions(best_final_state)

        # Iterates through every state that can be dug from the current
        # best_final_state
        for action in actions:
            next_state = self.mine.result(best_final_state, action)

            # Checks to see if node has already been computed
            if next_state in self.cached_nodes:
                next_dig = self.cached_nodes[next_state]
            else:
                next_dig = self.dp_recursive(self.mine.payoff(next_state),
                                             best_action_list + [action],
                                             next_state)

                # adds computed node to a dictionary with the state as the key.
                self.cached_nodes[next_state] = next_dig

            # compares the next dig on the frontier and updates if payoff is greater
            payoff_position = 0
            if next_dig[payoff_position] > best_dig[payoff_position]:
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


def bb_search_tree(mine):
    '''
    Computes the most profitable sequence of digging actions from the initial
        state of the mine using a Branch and Bound algorithm.
    Starts with an unassigned state of the mine and assigns every dig one at a
        time, adding every dig within tolerance to a priority queue.
    It then retrieves the dig with the most promising path based off the heuristic:
        The best state of the mine if there was no dig tolerance.
    It then repeats this process until a fully assigned, optimal state is found.

    Parameters
    ----------
    mine:
        An instance of the Mine class.

    Returns
    -------
    state :
        Represented with nested tuples.
        The state of the partially dug mine with the best payoff while within
        dig tolerance.

    '''
    best_so_far = mine.initial

    # The first state has every dig unassigned, represented as a -1.
    state = np.array(best_so_far) - 1
    optimistic_position, state_position = 1, 2
    priority_queue = search.PriorityQueue(order='max', f=lambda var: mine.payoff(var[optimistic_position]))
    operations_counter = count()
    dimensions = mine.cumsum_mine.ndim - 1

    # Represents the best state if there was no dig tolerance. Used as a heuristic.
    upper_bound = mine.cumsum_mine.argmax(dimensions)

    # Loops while any dig in the state is unassigned.
    unassigned_value = -1
    while np.any(state <= unassigned_value):

        # Generates coordinates of the next dig that has not been assigned a value.
        if mine.two_dimensions:
            x = np.argmax(state <= unassigned_value)
            coordinates = (x,)
        else:
            (x, y) = np.unravel_index(np.argmax(state <= unassigned_value), state.shape)
            coordinates = (x, y)

        for z in range(mine.len_z + 1):

            # The frontier_state is the actual frontier state with unassigned values.
            frontier_state = np.copy(state)
            frontier_state[coordinates] = z
            if mine.is_dangerous(frontier_state):
                continue

            unassigned_dig = (frontier_state <= unassigned_value)

            # The optimistic_state is used to determine order in the priority queue.
            # It consists of every assigned value in the state with every unassigned value being set
            # to its corresponding value in the upper bound.
            optimistic_state = np.copy(frontier_state)
            optimistic_state[unassigned_dig] = upper_bound[unassigned_dig]

            if mine.payoff(optimistic_state) > mine.payoff(best_so_far):

                # The operations_counter is used so that if optimistic_state arrays have the same payoff,
                # the heapq.heappush then compares a unique integer value instead.
                priority_queue.append((next(operations_counter), optimistic_state, frontier_state))

        # If the frontier is empty then the best state is the initial state.
        if not priority_queue.heap:
            return best_so_far

        # Pops the frontier state which has unassigned dig values.
        state = priority_queue.pop()[state_position]
    return convert_to_tuple(state)


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
    best_final_state = bb_search_tree(mine)
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

    # The difference between each of the two states' digs.
    state_difference = s1 - s0
    not_dug = 0
    assert np.all(state_difference >= not_dug)

    action_sequence = []

    # Adds all coordinates > 0 to the action sequence then shifts every value down 1.
    # Repeats process until every state difference is < 1.
    while np.any(state_difference > not_dug):

        # creates an array of coordinates.
        dig_layer = np.transpose((state_difference > not_dug).nonzero())
        dig_layer = convert_to_tuple(dig_layer)
        action_sequence.extend(dig_layer)
        state_difference -= 1
    return action_sequence
