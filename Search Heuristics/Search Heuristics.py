# Imported libraries.
import time as tm
import random
import itertools
import math
from abc import ABC

from simpleai.search.models import SearchProblem
from simpleai.search.traditional import astar
from os import system, name


# This class houses all the information, instructions, and actions that defines the code execution.
class info:
    message1 = {'Information': "\nUnlike ballistic missiles, cruise missiles do not follow a ballistic trajectory. "
                               "Rather, much like an airplane, "
                               "cruise missiles are equipped with a rocket or jet engine, "
                               "and wings to propel and guide them to a target.",
                'Information2': "The hallmark of modern-day cruise missiles is their incredible accuracy and ability "
                                "to avoid detection by flying at very low altitudes.\n"
                                "Such capabilities are made possible by GPS and several other sophisticated guidance "
                                "systems, which monitor the missile’s speed and direction and "
                                "compare radar measurements "
                                "and images of the terrain below to digital maps and photos stored on-board.\n",
                'Information3': "The code works by simulating the trajectory of the missile while flying "
                                "towards its intended target, which entails following multiple waypoints.\n",
                'Instructions': "\nWhen executing the code it will ask to enter the Altitude and the Distance "
                                "which is equal to height and width in a grid. A grid number higher then 6 x 6 will "
                                "take longer to calculate.\n",
                'Actions & States': "\nActions:\n"
                                    "U = Up\n"
                                    "D = Down\n"
                                    "L = Left\n"
                                    "R = Right\n"
                                    "RL = Roll\n"
                                    "Y = Yaw\n"
                                    "P = Pitch\n"
                                    "G = G_Forces\n"
                                    "A = Avoid\n"
                                    "\nStates:\n"
                                    "=== = Map or Ground\n"
                                    "R = Radar Zone\n"
                                    "D = Detection\n"
                                    "O = Low Altitude\n"
                                    "X = High Altitude\n"
                                    "C = Flight Under the Radar\n"}

    message2 = {'Title': "\nCruise Missile Trajectory Strategy Simulation\n",
                'Option1': "1. How Does it work?",
                'Option2': "2. Define the Waypoints",
                'Option3': "3. Instructions",
                'Option4': "4. Actions & States",
                'Option5': "5. Abort Mission\n"}


# The class labels houses all the labels that defines each interaction with the algorithm.
class labels:
    label1 = {'usr_inf1': "Tactical Land Attack Missile (TLAM)/A* Search\n",
              'usr_inf2': "Altitude: ",
              'usr_inf3': "Distance: ",
              'usr_inf4': "Number of Waypoints: "}

    label2 = {'Routes': "\nBest possible waypoint >> Proceed = y /Recalculate waypoint = n: ",
              'Trajectory': "Generating route......",
              'Keys': "Hit enter to proceed...\n"}

    sys1 = 'nt'
    sys2 = 'clear'
    sys3 = 'cls'


# This function clears the console terminal.
def clr_T():
    if name == labels.sys1:
        _ = system(labels.sys3)
    else:
        _ = system(labels.sys2)


# This function perform all the interactivity of the algorithm.
def deploy():
    while True:
        clr_T()
        print(info.message2['Title'])
        print(info.message2['Option1'])
        print(info.message2['Option2'])
        print(info.message2['Option3'])
        print(info.message2['Option4'])
        print(info.message2['Option5'])
        print("Enter Option: ", end=' ')
        userInput = input()

        if userInput == '1':
            print(info.message1['Information'])
            tm.sleep(1)
            print(info.message1['Information2'])
            print(info.message1['Information3'])
            print(labels.label2['Keys'], end='')
            input()

        elif userInput == '2':

            clr_T()
            print(labels.label1['usr_inf1'])
            print(labels.label1['usr_inf2'], end='')
            A = int(input())
            print(labels.label1['usr_inf3'], end='')
            D = int(input())
            print(labels.label1['usr_inf4'], end='')
            numWp = int(input())

            conf = False
            while not conf:
                func = f"Altitude = {A} Distance = {D}, number of Waypoints = {numWp} is this correct? (y/n): "
                print(func, end='')

                userInput = input()
                if userInput.lower() == 'y':
                    conf = True
                if userInput.lower() == 'n':
                    clr_T()
                    print(labels.label1['usr_inf1'])
                    print(labels.label1['usr_inf2'], end='')
                    A = int(input())
                    print(labels.label1['usr_inf3'], end='')
                    D = int(input())
                    print(labels.label1['usr_inf4'], end='')
                    numWp = int(input())

            conf = False
            while not conf:
                clr_T()
                print(labels.label1['usr_inf1'])
                mapping, startC, waypoints = map_gen(A, D, numWp)
                print(list_to_string(mapping))
                print(labels.label2['Routes'], end='')
                userInput = input()
                if userInput.lower() == 'y':
                    conf = True

            print(labels.label2['Trajectory'])
            result = astar(Traj_P(mapping, startC, waypoints))

            for action, state in result.path():
                print('\nAction:', action)
                print(state)

            print(labels.label2['Keys'], end='')
            input()

        elif userInput == '3':
            tm.sleep(1)
            print(info.message1['Instructions'])
            print(labels.label2['Keys'], end='')
            input()

        elif userInput == '4':
            tm.sleep(1)
            print(info.message1['Actions & States'])
            print(labels.label2['Keys'], end='')
            input()

        elif userInput == '5':
            quit()


# This function generates the state of range.
def range_gen(A, D):
    return random.randrange(0, A - 1), random.randrange(0, D - 1)


# This function generates a simple map
def map_gen(A, D, numWp):
    wP = []
    for i in range(numWp):
        x, y = range_gen(A, D)
        if (x, y) in wP:
            added = False
            while not added:
                x, y = range_gen(A, D)
                if not ((x, y) in wP):
                    added = True
        wP.append((x, y))

    # Generate start location
    x, y = range_gen(A, D)
    if (x, y) in wP:
        added = False
        while not added:
            x, y = range_gen(A, D)
            if not ((x, y) in wP):
                added = True
    start = (x, y)

    # Create Map List
    rand_map = []
    for i in range(D):
        row = []
        for j in range(A):
            if (j, i) in wP:
                row.append('D')
            elif (j, i) == start:
                row.append('R')
            else:
                row.append('==============================================')
        rand_map.append(row)
    return rand_map, start, wP


# This functions has been derived from the eight_puzzle.py.
def list_to_string(list_):
    return '\n'.join([' '.join(row) for row in list_])


# This functions has been derived from the eight_puzzle.py.
def string_to_list(string_):
    return [row.split(' ') for row in string_.split('\n')]


# This function returns a permuted list
def permutator(list_):
    return list(itertools.permutations(list_))


# This function calculates the distance between coordinates.
def flight_distance(coord1, coord2):
    return math.sqrt(((coord2[0] - coord1[0]) ** 2) + ((coord2[1] - coord1[1]) ** 2))


# This function compute the distance between two coordinates.
def compute_distance(coord1, coord2, straight_cost, diag_cost):
    dx = abs(coord1[0] - coord2[0])
    dy = abs(coord1[1] - coord2[1])
    return straight_cost * (dx + dy) + (diag_cost - 2 * straight_cost) * min(dx, dy)


# This function calculate the position of the AGM missile.
def AGM_location(state_list):
    for y, row in enumerate(state_list):
        for x, element in enumerate(row):
            if element == 'R' or element == 'X' or element == 'O':
                return x, y


# This function check the status of the trajectory in the map.
def check_map(map_list):
    for y, row in enumerate(map_list):
        for x, element in enumerate(row):
            if element == 'D' or element == 'X':
                return False
    return True


# This function determines the waypoint
def check_waypoints(map_list, waypoint_list):
    unclean = []
    for i, coord in enumerate(waypoint_list):
        if map_list[coord[1]][coord[0]] == 'D' or map_list[coord[1]][coord[0]] == 'X':
            unclean.append(coord)
    return unclean


# This function computes the possible waypoints.
def compute_waypoint(waypoint_list):
    perms = permutator(waypoint_list)
    waypoint_distance = {}
    for path in perms:
        distance = 0
        for i in range(len(path) - 1):
            distance += flight_distance(path[i], path[i + 1])
        waypoint_distance[path] = distance
    return waypoint_distance


# This class houses the trajectory problem tha uses A* Search heuristics.
class Traj_P(SearchProblem, ABC):

    def __init__(self, map_list, starting_coords, waypoint_list):

        self.COSTS = {'U': 1.0, 'D': 1.0, 'R': 1.0, 'L': 1.0, 'RL': 1.41, 'Y': 1.41, 'P': 1.41, 'G': 1.41, 'A': 2.0}
        initial_state = list_to_string(map_list)
        self.waypoints = waypoint_list
        self.starting_coords = starting_coords
        super().__init__(initial_state=initial_state)

    # this function returns a list of possible actions to take at a given location.
    def actions(self, state):
        map_list = string_to_list(state)

        rx, ry = AGM_location(map_list)
        A = len(map_list)
        D = len(map_list[0])

        actions = []

        if ry > 0:
            actions.append('U')
        if ry < A - 1:
            actions.append('D')
        if rx > 0:
            actions.append('L')
        if rx < D - 1:
            actions.append('R')
        if ry > 0 and rx > 0:
            actions.append('RL')
        if ry > 0 and rx < D - 1:
            actions.append('Y')
        if ry < A - 1 and rx > 0:
            actions.append('P')
        if ry < A - 1 and rx < D - 1:
            actions.append('G')

        actions.append('A')

        return actions

    # The function result returns the resulting state after performing the passed action.
    def result(self, state, action):
        map_list = string_to_list(state)
        rx, ry = AGM_location(map_list)

        if action.count('A'):
            if map_list[ry][rx] == 'X':
                map_list[ry][rx] = 'O'
        else:
            new_rx, new_ry = rx, ry

            if action.count('U'):
                new_ry -= 1
            if action.count('L'):
                new_rx -= 1
            if action.count('D'):
                new_ry += 1
            if action.count('R'):
                new_rx += 1
            if map_list[ry][rx] == 'R':
                map_list[ry][rx] = '=============================================='
            elif map_list[ry][rx] == 'X':
                map_list[ry][rx] = 'D'
            elif map_list[ry][rx] == 'O':
                map_list[ry][rx] = 'C'
            if map_list[new_ry][new_rx] == '==============================================':
                map_list[new_ry][new_rx] = 'R'
            elif map_list[new_ry][new_rx] == 'D':
                map_list[new_ry][new_rx] = 'X'
            if map_list[new_ry][new_rx] == 'C':
                map_list[new_ry][new_rx] = 'O'

        new_state = list_to_string(map_list)
        return new_state

    def is_goal(self, state):
        map_list = string_to_list(state)
        if check_map(map_list) and (self.starting_coords == AGM_location(map_list)):
            return True
        return False

    # This function calculates and returns the cost of performing an action.
    def cost(self, state1, action, state2):
        return self.COSTS[action]

    # The heuristic function estimation of the distance from a state to the goal.
    def heuristic(self, state):
        map_list = string_to_list(state)
        r_coords = AGM_location(map_list)

        h_vals = []

        rad_detector = check_waypoints(map_list, self.waypoints)
        if len(rad_detector) > 0:
            waypoint_distances = compute_waypoint(rad_detector)
            for i, waypoint in enumerate(rad_detector):
                corresponding_distances = {key: val for key, val in waypoint_distances.items() if key[0] == waypoint}
                min_key = min(corresponding_distances, key=corresponding_distances.get)
                h_vals.append(compute_distance(r_coords, waypoint, 1, 1.41) + corresponding_distances[min_key] + (
                        2 * len(rad_detector)) + compute_distance(min_key[0], self.starting_coords, 1, 1.41))

            return min(h_vals)

        else:
            return compute_distance(r_coords, self.starting_coords, 1, 1.41)


# Executes the entire algorithm.
if __name__ == "__main__":
    deploy()
