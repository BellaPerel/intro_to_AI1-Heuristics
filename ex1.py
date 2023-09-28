import search
import random
import math
import json
import itertools
import time

ids = ["316350651", "324620814"]


# convert dictionary nested dictionaries to json
def dict_to_json(dict):
    return json.dumps(dict)


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def create_graph(self, map):
        graph = {}
        for i in range(len(map)):
            for j in range(len(map[0])):
                graph[str((i, j))] = {}
                if map[i][j] != 'I':
                    if i > 0 and map[i - 1][j] != 'I':
                        graph[str((i, j))][str((i - 1, j))] = 1
                    if i < len(map) - 1 and map[i + 1][j] != 'I':
                        graph[str((i, j))][str((i + 1, j))] = 1
                    if j > 0 and map[i][j - 1] != 'I':
                        graph[str((i, j))][str((i, j - 1))] = 1
                    if j < len(map[0]) - 1 and map[i][j + 1] != 'I':
                        graph[str((i, j))][str((i, j + 1))] = 1
        return graph

    # apply floyd warshall algorithm to the graph
    def floyd_warshall(self, graph):
        distance_matrix = {}
        for i in graph.keys():
            distance_matrix[i] = {}
            for j in graph.keys():
                if i == j:
                    distance_matrix[i][j] = 0
                elif j in graph[i].keys():
                    distance_matrix[i][j] = graph[i][j]
                else:
                    distance_matrix[i][j] = math.inf
        for k in graph.keys():
            for i in graph.keys():
                for j in graph.keys():
                    distance_matrix[i][j] = min(distance_matrix[i][j], distance_matrix[i][k] + distance_matrix[k][j])
        return distance_matrix


    #given a distance, find all the edges that create a path from the source to the destination with that distance
    def find_edges_with_distance_recourse(self, distance, source, destination, graph):
        if distance == 0:
            return []
        if distance == 1:
            if destination in graph[source].keys():
                return [(source, destination)]
            else:
                return []
        else:
            for i in graph[source].keys():
                edges_to_add = self.find_edges_with_distance(distance - 1, i, destination, graph)
            if edges_to_add != []:
                edges_to_add.append((source, i))
            return edges_to_add

    def find_edges_with_distance(self, distance, source, destination, graph):
        #create an equivalent function to find_edges_with_distance_recourse, that doesn't se find_edges_with_distance
        if self.edges_of_shortest_path[destination][source]: ##NOT EXISTS
            return self.edges_of_shortest_path[destination][source]
        edges = []
        queue = [(source, distance)]
        while queue != []:
            source_and_dist = queue.pop(0)
            source = source_and_dist[0]
            distance = source_and_dist[1]
            if distance != 0 and distance != float("inf"):
                for neighbor in graph[source].keys():
                    if self.edges_of_shortest_path[destination][neighbor]:
                        edges += self.edges_of_shortest_path[destination][neighbor]
                    else:
                        if self.distance_matrix_fw[neighbor][destination] == distance - 1:
                            edges.append((source, neighbor))
                            queue.append((neighbor, distance - 1))
        return edges




    def create_edges_shortest_path_dict(self):
        #for each source and destination, create a list of all the edges that create a path with the distance between them
        #take the distance from the distance matrix dist_matrix_fw
        edges_dict = {}
        for i in self.distance_matrix_fw.keys():
            for j in self.distance_matrix_fw[i].keys():
                edges_dict[(i, j)] = set(self.find_edges_with_distance(self.distance_matrix_fw[i][j], i, j, self.graph))
        return edges_dict


    def paint_edges(self, edges_color_dict, list_of_edges):
        # paint the edges in the list of edges to black
        for edge in list_of_edges:
            edges_color_dict[edge] = 1


    def new_distance_matrix(self,graph):
        distance_matrix_fw = self.floyd_warshall(graph)
        return distance_matrix_fw

    def manhattan_distance_matrix(self, map):
        # create a distance matrix based on the map given between each two coordinates
        matrix = {}
        for i in range(len(map)):
            for j in range(len(map[0])):
                matrix[str((i, j))] = {}
                for k in range(len(map)):
                    for l in range(len(map[0])):
                        matrix[str((i, j))][str((k, l))] = abs(k - i) + abs(l - j)
        return matrix

    def create_edges_color_dict(self, graph):
        #create a dictionary of edges, where the key is the edge, and the value is the color of the edge
        #set all edges to white
        edges_dict = {}
        for i in graph.keys():
            for j in graph[i].keys():
                edges_dict[(i, j)] = 0
        return edges_dict



    def __init__(self, initial):
        self.currentprint = '0'
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""

        "state: " \
        "map" \
        "taxi: (location,curr_passengers,destinations,max_passengers,curr_fuel,max_fuel)" \
        "passengers: (name,location,destination)"
        self.map = initial["map"]
        self.statenum = 0
        state = {}
        state["taxies"] = {}
        state["passengers"] = {}
        state["passengers waiting"] = 0
        state["passengers picked up"] = 0
        for taxi in initial["taxis"].keys():
            taxi_info = initial["taxis"][taxi]
            state["taxies"][taxi] = {"location": taxi_info["location"],
                                     "curr_passengers": 0,
                                     "destinations": {},
                                     "max_passengers": taxi_info["capacity"],
                                     "curr_fuel": taxi_info["fuel"],
                                     "max_fuel": taxi_info["fuel"]}

        self.passengers_by_location_dict = {}
        self.operations = {"move left": 1,
                           "move right": 1,
                           "move up": 1,
                           "move down": 1,
                           "pick up": 1,
                           "drop off": 1,
                           "refuel": 1,
                           "wait": 1}
        self.gas_stations = []
        self.graph = self.create_graph(self.map)
        self.distance_matrix = self.manhattan_distance_matrix(self.map)
        self.distance_matrix_fw = self.new_distance_matrix(self.graph)
        self.edges_color_dict = self.create_edges_color_dict(self.graph)
        # self.edges_of_shortest_path = {}
        # self.edges_of_shortest_path = self.create_edges_shortest_path_dict()

        # in self.map, find all gas stations and add them to self.gas_stations
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == "G":
                    self.gas_stations.append((i, j))

        for passenger in initial["passengers"].keys():
            state["passengers waiting"] += 1
            passenger_info = initial["passengers"][passenger]
            state["passengers"][passenger] = {"location": passenger_info["location"],
                                              "destination": passenger_info["destination"]}
            # save in 'loc' the passenger's location
            loc = passenger_info["location"]
            # if the location is not in the dictionary, add it and add the passenger as value in tuple,
            # otherwise append it to the value
            if loc not in self.passengers_by_location_dict.keys():
                self.passengers_by_location_dict[str(loc)] = [passenger]
            else:
                self.passengers_by_location_dict[str(loc)].append(passenger)

        state["passengers_by_location"] = self.passengers_by_location_dict.copy()
        state = dict_to_json(state)
        initial = state
        search.Problem.__init__(self, initial)

    def check_if_gas_station(self, taxi_operations, state_as_dict):
        # check for all taxies, if each one's lcoation is 'G', if so, add 'refuel' to actions
        # taxi operations is a dict, gets actions from the conditions
        for taxi in state_as_dict["taxies"].keys():
            loc = state_as_dict["taxies"][taxi]["location"]
            if self.map[int(loc[0])][int(loc[1])] != "G" or state_as_dict["taxies"][taxi]["curr_fuel"] == \
                    state_as_dict["taxies"][taxi]["max_fuel"]:
                taxi_operations[taxi]["refuel"] = 0
        return taxi_operations

    def check_if_gas_station_iter(self, taxi_operations, taxi_dict, taxi_name):
        # check for all taxies, if each one's lcoation is 'G', if so, add 'refuel' to actions
        # taxi operations is a dict, gets actions from the conditions
        loc = taxi_dict["location"]
        if self.map[int(loc[0])][int(loc[1])] != "G" or taxi_dict["curr_fuel"] == \
                taxi_dict["max_fuel"]:
            taxi_operations["refuel"] = 0
        return taxi_operations

    def check_if_out_of_borders(self, taxi_operations, state_as_dict):
        # check for all taxies, if their location is an edge of the map, and add legal directions accordingly
        for taxi in state_as_dict["taxies"].keys():
            loc = state_as_dict["taxies"][taxi]["location"]
            if loc[0] == 0:
                taxi_operations[taxi]["move up"] = 0
            if loc[0] == len(self.map) - 1:
                taxi_operations[taxi]["move down"] = 0
            if loc[1] == 0:
                taxi_operations[taxi]["move left"] = 0
            if loc[1] == len(self.map[0]) - 1:
                taxi_operations[taxi]["move right"] = 0
        return taxi_operations

    def check_if_out_of_borders_iter(self, taxi_operations, taxi_dict, taxi_name):
        # check for all taxies, if their location is an edge of the map, and add legal directions accordingly
        loc = taxi_dict["location"]
        if loc[0] == 0:
            taxi_operations["move up"] = 0
        if loc[0] == len(self.map) - 1:
            taxi_operations["move down"] = 0
        if loc[1] == 0:
            taxi_operations["move left"] = 0
        if loc[1] == len(self.map[0]) - 1:
            taxi_operations["move right"] = 0
        return taxi_operations

    def check_if_near_impassable(self, taxi_operations, state_as_dict):
        # for all taxies, if they have a direction in their operation, check if the movement would result in them being in an 'I' location if the map, if so, remove the direction from their operations
        for taxi in state_as_dict["taxies"].keys():
            loc = state_as_dict["taxies"][taxi]["location"]
            loc = [int(loc[0]), int(loc[1])]
            if taxi_operations[taxi]["move left"]:
                if self.map[loc[0]][loc[1] - 1] == "I":
                    taxi_operations[taxi]["move left"] = 0
            if taxi_operations[taxi]["move right"]:
                if self.map[loc[0]][loc[1] + 1] == "I":
                    taxi_operations[taxi]["move right"] = 0
            if taxi_operations[taxi]["move up"]:
                if self.map[loc[0] - 1][loc[1]] == "I":
                    taxi_operations[taxi]["move up"] = 0
            if taxi_operations[taxi]["move down"]:
                if self.map[loc[0] + 1][loc[1]] == "I":
                    taxi_operations[taxi]["move down"] = 0
        return taxi_operations

    def check_if_near_impassable_iter(self, taxi_operations, taxi_dict, taxi_name):
        # for all taxies, if they have a direction in their operation, check if the movement would result in them being in an 'I' location if the map, if so, remove the direction from their operations
        loc = taxi_dict["location"]
        loc = [int(loc[0]), int(loc[1])]
        if taxi_operations["move left"]:
            if self.map[loc[0]][loc[1] - 1] == "I":
                taxi_operations["move left"] = 0
        if taxi_operations["move right"]:
            if self.map[loc[0]][loc[1] + 1] == "I":
                taxi_operations["move right"] = 0
        if taxi_operations["move up"]:
            if self.map[loc[0] - 1][loc[1]] == "I":
                taxi_operations["move up"] = 0
        if taxi_operations["move down"]:
            if self.map[loc[0] + 1][loc[1]] == "I":
                taxi_operations["move down"] = 0
        return taxi_operations

    def check_if_out_of_fuel(self, taxi_operations, state_as_dict):
        # check for all taxies, if each one's fuel is 0, if so, remove actions including directions
        # taxi operations is a dict, gets actions from the conditions
        for taxi in state_as_dict["taxies"].keys():
            if state_as_dict["taxies"][taxi]["curr_fuel"] == 0:
                # if the item exists it the list, remove it
                taxi_operations[taxi]["move left"] = 0
                taxi_operations[taxi]["move right"] = 0
                taxi_operations[taxi]["move up"] = 0
                taxi_operations[taxi]["move down"] = 0
        return taxi_operations

    def check_if_out_of_fuel_iter(self, taxi_operations, taxi_dict, taxi_name):
        # check for all taxies, if each one's fuel is 0, if so, remove actions including directions
        # taxi operations is a dict, gets actions from the conditions
        if taxi_dict["curr_fuel"] == 0:
            # if the item exists it the list, remove it
            taxi_operations["move left"] = 0
            taxi_operations["move right"] = 0
            taxi_operations["move up"] = 0
            taxi_operations["move down"] = 0
        return taxi_operations

    def check_if_passenger_in_location(self, taxi_operations, state_as_dict):
        # for each taxi, if there isn't a passenger in the location, or if the taxi is full, remove the 'pick up' action
        for taxi in state_as_dict["taxies"].keys():
            loc = tuple(state_as_dict["taxies"][taxi]["location"])
            if str(loc) not in state_as_dict["passengers_by_location"].keys() or state_as_dict["taxies"][taxi][
                "curr_passengers"] == state_as_dict["taxies"][taxi]["max_passengers"]:
                taxi_operations[taxi]["pick up"] = 0
        return taxi_operations

    def check_if_passenger_in_location_iter(self, taxi_operations, taxi_dict, taxi_name, state_as_dict):
        # for each taxi, if there isn't a passenger in the location, or if the taxi is full, remove the 'pick up' action
        loc = tuple(taxi_dict["location"])
        if str(loc) not in state_as_dict["passengers_by_location"].keys() or taxi_dict[
            "curr_passengers"] == taxi_dict["max_passengers"]:
            taxi_operations["pick up"] = 0
        return taxi_operations

    def check_if_potential_destination(self, taxi_operations, state_as_dict):
        # for each taxi, see if the location is not a destination, if so, remove the 'drop off' action
        for taxi in state_as_dict["taxies"].keys():
            loc = state_as_dict["taxies"][taxi]["location"]
            if loc not in state_as_dict["taxies"][taxi]["destinations"].values():
                taxi_operations[taxi]["drop off"] = 0
        return taxi_operations

    def check_if_potential_destination_iter(self, taxi_operations, taxi_dict, taxi_name):
        # for each taxi, see if the location is not a destination, if so, remove the 'drop off' action
        loc = taxi_dict["location"]
        if loc not in taxi_dict["destinations"].values():
            taxi_operations["drop off"] = 0
        return taxi_operations

    def reformat_actions(self, taxi_operations, state_as_dict):
        # change format to (action, taxi, relevant passenger if 'pick up' or 'drop off' or future location if 'move')
        actions = []
        for idx, taxi in enumerate(taxi_operations.keys()):
            actions.append([])
            if taxi_operations[taxi]["pick up"]:
                list_of_passengers_to_pick_up = state_as_dict["passengers_by_location"][
                    str(tuple(state_as_dict["taxies"][taxi]["location"]))]
                for passenger in list_of_passengers_to_pick_up:
                    actions[idx].append(("pick up", taxi, passenger))
            if taxi_operations[taxi]["drop off"]:
                for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                    if state_as_dict["taxies"][taxi]["destinations"][passenger] == state_as_dict["taxies"][taxi][
                        "location"]:
                        actions[idx].append(("drop off", taxi, passenger))
            if taxi_operations[taxi]["move right"]:
                actions[idx].append(("move", taxi, (
                    state_as_dict["taxies"][taxi]["location"][0], state_as_dict["taxies"][taxi]["location"][1] + 1)))
            if taxi_operations[taxi]["move left"]:
                actions[idx].append(("move", taxi, (
                    state_as_dict["taxies"][taxi]["location"][0], state_as_dict["taxies"][taxi]["location"][1] - 1)))
            if taxi_operations[taxi]["move up"]:
                actions[idx].append(("move", taxi, (
                    state_as_dict["taxies"][taxi]["location"][0] - 1, state_as_dict["taxies"][taxi]["location"][1])))
            if taxi_operations[taxi]["move down"]:
                actions[idx].append(("move", taxi, (
                    state_as_dict["taxies"][taxi]["location"][0] + 1, state_as_dict["taxies"][taxi]["location"][1])))
            if taxi_operations[taxi]["refuel"]:
                actions[idx].append(("refuel", taxi))
            if taxi_operations[taxi]["wait"]:
                actions[idx].append(("wait", taxi))
        return actions

    def create_all_available_actions(self, reformatted_actions):
        # return cartesian product of all actions
        return list(itertools.product(*reformatted_actions))

    def remove_illegal_actions(self, actions, state_as_dict):
        updated_actions = []
        for action in actions:
            list_of_locations = []
            list_of_actions = []
            for action_of_single_taxi in action:
                list_of_actions.append(action_of_single_taxi[0])
                if action_of_single_taxi[0] == "move":
                    list_of_locations.append(action_of_single_taxi[2])
                else:
                    # append the current location of the taxi
                    current_loc = state_as_dict["taxies"][action_of_single_taxi[1]]["location"]
                    list_of_locations.append(tuple(current_loc))

            # if all the actions are "wait", continue
            if list_of_actions == ["wait"] * len(list_of_actions):
                continue
            # if all the locations are unique, then the action is legal, add it to the list of updated_actions
            if len(list_of_locations) == len(set(list_of_locations)):
                updated_actions.append(action)
        return updated_actions


    def actions(self, state):
        # total_time_actions_start = time.time()
        #   self.statenum += 1
        """tests: 
        before: location, is it a gas station, is there a passenger, is it a destination
        1. is taxi overloaded
        2. if taxi is out of bounds
        3. if taxi is out of fuel
        4. if taxi is in a passenger location
        5. if taxi is in a destination location
        6. if taxi is in fuel station    
        "after tests: two taxies same location
        
        """"Returns all the actions that can be executed in the given"
        "state. The result should be a tuple (or other iterable) of actions"
        "as defined in the problem description file"""
        state_as_dict = json.loads(state)
        taxi_operations = {}

        for taxi in state_as_dict["taxies"].keys():
            #if there are passengers in the taxi
            if state_as_dict["taxies"][taxi]["curr_passengers"] > 0:
                #find the minimal distance between the taxi and the destinations
                min_dist = float('inf')
                for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                    dist = self.distance_matrix_fw[str(tuple(state_as_dict["taxies"][taxi]["location"]))][
                        str(tuple(state_as_dict["taxies"][taxi]["destinations"][passenger]))]
                    if dist < min_dist:
                        min_dist = dist
                if min_dist == float('inf'):
                    #print("illegal passenger")
                    return ((),(),())
                if min_dist > state_as_dict["taxies"][taxi]["curr_fuel"]:
                    #find the minimal distance between the taxi and the gas stations
                    min_dist_gas = float('inf')
                    for gas_station in self.gas_stations:
                        dist = self.distance_matrix_fw[str(tuple(state_as_dict["taxies"][taxi]["location"]))][str(gas_station)]
                        if dist < min_dist_gas:
                            min_dist_gas = dist
                    if min_dist_gas > state_as_dict["taxies"][taxi]["curr_fuel"]:
                        # taxi has no fuel for the task, game over
                        # print("game over: fuel:", state_as_dict["taxies"][taxi]["curr_fuel"], "min_dist:", min_dist, "min_dist_gas:", min_dist_gas)
                        return ((),(),())

        for taxi in state_as_dict["taxies"].keys():
            taxi_operations[taxi] = self.operations.copy()

        # check gas stations
        for taxi in state_as_dict["taxies"].keys():
            taxi_operations[taxi] = self.check_if_gas_station_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi], taxi)
            # check if out of borders
            taxi_operations[taxi] = self.check_if_out_of_borders_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi], taxi)
            # check if near impassable
            taxi_operations[taxi] = self.check_if_near_impassable_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi], taxi)
            # check if out of fuel
            taxi_operations[taxi] = self.check_if_out_of_fuel_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi], taxi)
            # check if passenger in location
            taxi_operations[taxi] = self.check_if_passenger_in_location_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi],
                                                                             taxi, state_as_dict)
            # check if potential destination
            taxi_operations[taxi] = self.check_if_potential_destination_iter(taxi_operations[taxi], state_as_dict["taxies"][taxi],
                                                                             taxi)

        actions_reformatted = self.reformat_actions(taxi_operations, state_as_dict)
        actions = self.create_all_available_actions(actions_reformatted)
        actions = self.remove_illegal_actions(actions, state_as_dict)
        # total_time_actions_end = time.time()
        # self.total_time_actions += total_time_actions_end - total_time_actions_start
        return tuple(actions)

    def result(self, state, action):
        # result_timer_results_start = time.time()
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        # state is given as a json file. So first, convert it to a dictionary
        state_as_dict = json.loads(state)
        # iterate on every action in the action tuple, and execute it
        # executing the action in the following manner:
        # the second element of the action tuple is the taxi key
        # the first element of the action tuple is the action to be executed
        # if the action is "move", then the third element of the action tuple is the location to move to
        # if the action is "pick up", then the third element of the action tuple is the passenger to pick up
        # if the action is "drop off", then the third element of the action tuple is the passenger to drop off
        # picking and dropping passenger updates the curr_passenger attribute of the taxi
        # refueling the taxi turns its current fuel value to its max fuel value
        for action_of_single_taxi in action:
            taxi_key = action_of_single_taxi[1]
            action_to_execute = action_of_single_taxi[0]
            loc = state_as_dict["taxies"][taxi_key]["location"]
            if action_to_execute == "move":
                state_as_dict["taxies"][taxi_key]["location"] = list(action_of_single_taxi[2])
                # remove one unit of fuel
                state_as_dict["taxies"][taxi_key]["curr_fuel"] -= 1
            elif action_to_execute == "pick up":
                state_as_dict["taxies"][taxi_key]["curr_passengers"] = state_as_dict["taxies"][taxi_key][
                                                                           "curr_passengers"] + 1
                # removing the passenger from the passengers_by_location_dict
                loc_as_tuple = tuple(loc)
                state_as_dict["passengers_by_location"][str(loc_as_tuple)].remove(action_of_single_taxi[2])
                # if the value in the dictionary is an empty list, remove the key
                if state_as_dict["passengers_by_location"][str(loc_as_tuple)] == []:
                    state_as_dict["passengers_by_location"].pop(str(loc_as_tuple))
                # add the passenger and its destination to the destinations dictionary of the taxi
                state_as_dict["taxies"][taxi_key]["destinations"][action_of_single_taxi[2]] = \
                    state_as_dict["passengers"][action_of_single_taxi[2]]["destination"]
                state_as_dict["passengers waiting"] -= 1
                state_as_dict["passengers picked up"] += 1
            elif action_to_execute == "drop off":
                state_as_dict["taxies"][taxi_key]["curr_passengers"] = state_as_dict["taxies"][taxi_key][
                                                                           "curr_passengers"] - 1
                # removing the passenger from the destinations dictionary of the taxi
                del state_as_dict["taxies"][taxi_key]["destinations"][action_of_single_taxi[2]]
                state_as_dict["passengers picked up"] -= 1
            elif action_to_execute == "refuel":
                state_as_dict["taxies"][taxi_key]["curr_fuel"] = state_as_dict["taxies"][taxi_key]["max_fuel"]
            else:
                continue

        # return the dictionary as a json file
        # result_timer_results_end = time.time()
        # self.total_time_results += result_timer_results_end - result_timer_results_start
        # print(action)
        # print(state_as_dict)
        return json.dumps(state_as_dict)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state_as_dict = json.loads(state)
        flag = state_as_dict["passengers waiting"] == 0 and state_as_dict["passengers picked up"] == 0
        return flag


    def control_print(self,node):
        if str(node) > self.currentprint:
            print(node)
            self.currentprint = str(node)


    def h(self, node):
        if self.goal_test(node.state):
            return 0
        to_add = self.h_3(node)
        state_as_dict_for_work = json.loads(node.state).copy()
        # create a dictionary of taxi and locations
        taxi_by_location = {}
        for taxi in state_as_dict_for_work["taxies"].keys():
            taxi_by_location[taxi] = [tuple(state_as_dict_for_work["taxies"][taxi]["location"])]

        # NEXT WE DEFINE TASKS AS TUPLES (PICK UP x, DROP OFF y ETC) AND ITERATE THROUGH THE TAXIS TO MAKE A HILL CLIMBING
        # PLAN FOR ALL THE TAXIS TO CALCUALTE THE COST OF THE PLAN
        # WE USE THE TAXI_BY_LOCATION DICTIONARY TO FIND THE CLOSEST TAXI TO THE PASSENGER

        while taxi_by_location and state_as_dict_for_work["passengers_by_location"]:
            # find the closest taxi to the closest passenger
            closest_taxi = None
            closest_passenger_location = None
            closest_distance = None
            for taxi in taxi_by_location.keys():
                for passenger_location in state_as_dict_for_work["passengers_by_location"].keys():
                    for location_of_taxi in taxi_by_location[taxi]:
                        distance = self.distance_matrix_fw[str(location_of_taxi)][str(passenger_location)]
                        if closest_distance is None or distance < closest_distance:
                            closest_distance = distance
                            closest_taxi = taxi
                            closest_passenger_location = passenger_location
            # add the distance to the sum
            to_add += closest_distance
            # remove the passenger from the dictionary
            state_as_dict_for_work["passengers_by_location"][closest_passenger_location].pop(0)
            # and if the list is now empty, remove the key
            if not state_as_dict_for_work["passengers_by_location"][closest_passenger_location]:
                del state_as_dict_for_work["passengers_by_location"][closest_passenger_location]
            # if the taxi is full, remove it from the dictionary
            # otherwise, add all of its destinations to the dictionary, including the passenger's destination
            if len(state_as_dict_for_work["taxies"][closest_taxi]["destinations"].keys()) == \
                    state_as_dict_for_work["taxies"][closest_taxi]["max_passengers"]:
                taxi_by_location.pop(closest_taxi)
            else:
                for destination in state_as_dict_for_work["taxies"][closest_taxi]["destinations"].keys():
                    taxi_by_location[closest_taxi].append(
                        tuple(state_as_dict_for_work["taxies"][closest_taxi]["destinations"][destination]))
                taxi_by_location[closest_taxi].append(closest_passenger_location)
        return to_add



    def h_1(self, node):
        """
        This is a simple heuristic
        """
        count_of_unpicked = 0 # number of passengers that are not picked up
        state = json.loads(node.state)
        for location in state["passengers_by_location"].keys():
            for passenger in state["passengers_by_location"][location]:
                count_of_unpicked += 1
        # count amount of destinations of all taxies
        count_of_destinations = 0
        for taxi in state["taxies"].keys():
            count_of_destinations += len(state["taxies"][taxi]["destinations"].keys())
        count_of_picked_but_not_dropped = count_of_destinations
        return (count_of_unpicked * 2 + count_of_picked_but_not_dropped) / len(state["taxies"].keys())

    def manhattan_distance(self, loc1, loc2):
        loc1 = str(loc1)
        loc2 = str(loc2)
        loc1 = loc1.replace("(", "")
        loc1 = loc1.replace(")", "")
        loc1 = loc1.replace(",", " ")
        loc2 = loc2.replace("(", "")
        loc2 = loc2.replace(")", "")
        loc2 = loc2.replace(",", " ")
        myList = loc1.split()
        myList = list(map(int, myList))
        myList2 = loc2.split()
        myList2 = list(map(int, myList2))
        loc1 = tuple(myList)
        loc2 = tuple(myList2)
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        # in passengers by location, return the manhattan distance of each passenger's location and his or her destination
        state_as_dict = json.loads(node.state)
        sum = 0
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                sum += self.distance_matrix[str(location)][
                    str(tuple(state_as_dict["passengers"][passenger]["destination"]))]
        for taxi in state_as_dict["taxies"].keys():
            # compute manhattan distancebetween taxi's location and all of its passengers' destinations
            # sum all distances
            # divide by number of taxies
            # return the result
            sum_of_distances = 0
            location = (state_as_dict["taxies"][taxi]["location"])
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                sum += self.distance_matrix[str(tuple((location)))][
                    str(tuple((state_as_dict["taxies"][taxi]["destinations"][passenger])))]
        # if self.currentprint != str(node):
        #     print(node)
        #     self.currentprint = str(node)
        return sum / len(state_as_dict["taxies"].keys())

    def h_3(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        # in passengers by location, return the manhattan distance of each passenger's location and his or her destination
        state_as_dict = json.loads(node.state)
        sum = 0
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                sum += self.distance_matrix_fw[str(location)][
                    str(tuple(state_as_dict["passengers"][passenger]["destination"]))]
        for taxi in state_as_dict["taxies"].keys():
            # compute manhattan distancebetween taxi's location and all of its passengers' destinations
            # sum all distances
            # divide by number of taxies
            # return the result
            sum_of_distances = 0
            location = (state_as_dict["taxies"][taxi]["location"])
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                sum += self.distance_matrix_fw[str(tuple((location)))][
                    str(tuple((state_as_dict["taxies"][taxi]["destinations"][passenger])))]
        # if self.currentprint != str(node):
        #     print(node)
        #     self.currentprint = str(node)
        return sum / len(state_as_dict["taxies"].keys())



    def h_4(self, node):
        to_add = self.h_3(node)
        state_as_dict_for_work = json.loads(node.state).copy()
        # create a dictionary of taxi and locations
        taxi_by_location = {}
        for taxi in state_as_dict_for_work["taxies"].keys():
            taxi_by_location[taxi] = [tuple(state_as_dict_for_work["taxies"][taxi]["location"])]
            ###LATE NIGHT WORKING HERE
            # if the taxi has a passenger, save in 'dist' the smallest distance between the taxi and the destinations of the passengers
            if len(state_as_dict_for_work["taxies"][taxi]["destinations"].keys()) > 0:
                dist = float('inf')
                for passenger in state_as_dict_for_work["taxies"][taxi]["destinations"].keys():
                    if self.distance_matrix_fw[str(tuple(state_as_dict_for_work["taxies"][taxi]["location"]))][
                        str(tuple(state_as_dict_for_work["taxies"][taxi]["destinations"][passenger]))] < dist:
                        dist = self.distance_matrix_fw[str(tuple(state_as_dict_for_work["taxies"][taxi]["location"]))][
                            str(tuple(state_as_dict_for_work["taxies"][taxi]["destinations"][passenger]))]
                    else:  # the spot should have been accessible, therefore the passenger is in an inaccessible location
                        return float('inf')

                # if the distance is bigger than the current fuel, find the distance between the taxi and the closest gas station
                # save the distance in 'dist_gas'
                # gas stations are saved in self.gas_stations
                # if the distance is smaller, or equal to the current fuel, add the distance to the sum, and replace the location of the taxi with the location of the gas station
                # if the distance is bigger, then the taxi can't reach the gas station
                # in this case, the taxi can't reach the passenger's destination at all
                # return infinity
                if dist > state_as_dict_for_work["taxies"][taxi]["curr_fuel"]:
                    dist_gas = float('inf')
                    for gas in self.gas_stations:
                        if self.distance_matrix_fw[str(tuple(state_as_dict_for_work["taxies"][taxi]["location"]))][
                            str(tuple(gas))] < dist_gas:
                            dist_gas = \
                                self.distance_matrix_fw[str(tuple(state_as_dict_for_work["taxies"][taxi]["location"]))][
                                    str(tuple(gas))]
                    if dist_gas <= state_as_dict_for_work["taxies"][taxi]["curr_fuel"]:
                        to_add += dist_gas
                        taxi_by_location[taxi] = [tuple(gas)]
                        # print("the distance is bigger than the current fuel", "dist:",dist,"c_fuel:", state_as_dict_for_work["taxies"][taxi]["curr_fuel"])
                    else:
                        # print("the distance is bigger than the current fuel and the taxi can't reach the gas station", "dist:",dist, "c_fuel:", state_as_dict_for_work["taxies"][taxi]["curr_fuel"], "dist_gas:", dist_gas)
                        # if dist == float('inf'):
                        #     print(self.map)
                        #     print("taxi_location:", state_as_dict_for_work["taxies"][taxi]["location"], "gas_location:", gas)
                        #     print("taxi_destination:", state_as_dict_for_work["taxies"][taxi]["destinations"][passenger])
                        return float('inf')

        # NEXT WE DEFINE TASKS AS TUPLES (PICK UP x, DROP OFF y ETC) AND ITERATE THROUGH THE TAXIS TO MAKE A HILL CLIMBING
        # PLAN FOR ALL THE TAXIS TO CALCUALTE THE COST OF THE PLAN
        # WE USE THE TAXI_BY_LOCATION DICTIONARY TO FIND THE CLOSEST TAXI TO THE PASSENGER

        while taxi_by_location and state_as_dict_for_work["passengers_by_location"]:
            # find the closest taxi to the closest passenger
            closest_taxi = None
            closest_passenger_location = None
            closest_distance = None
            for taxi in taxi_by_location.keys():
                for passenger_location in state_as_dict_for_work["passengers_by_location"].keys():
                    for location_of_taxi in taxi_by_location[taxi]:
                        distance = self.distance_matrix_fw[str(location_of_taxi)][str(passenger_location)]
                        if closest_distance is None or distance < closest_distance:
                            closest_distance = distance
                            closest_taxi = taxi
                            closest_passenger_location = passenger_location
            # add the distance to the sum
            to_add += closest_distance
            # remove the passenger from the dictionary
            state_as_dict_for_work["passengers_by_location"][closest_passenger_location].pop(0)
            # and if the list is now empty, remove the key
            if not state_as_dict_for_work["passengers_by_location"][closest_passenger_location]:
                del state_as_dict_for_work["passengers_by_location"][closest_passenger_location]
            # if the taxi is full, remove it from the dictionary
            # otherwise, add all of its destinations to the dictionary, including the passenger's destination
            if len(state_as_dict_for_work["taxies"][closest_taxi]["destinations"].keys()) == \
                    state_as_dict_for_work["taxies"][closest_taxi]["max_passengers"]:
                taxi_by_location.pop(closest_taxi)
            else:
                for destination in state_as_dict_for_work["taxies"][closest_taxi]["destinations"].keys():
                    taxi_by_location[closest_taxi].append(
                        tuple(state_as_dict_for_work["taxies"][closest_taxi]["destinations"][destination]))
                taxi_by_location[closest_taxi].append(closest_passenger_location)
        return to_add


    def h_5(self, node):
        state_as_dict = json.loads(node.state)
        pick_up_steps = 0
        drop_off_steps = 0
        for taxi in state_as_dict["taxies"].keys():
            drop_off_steps += len(state_as_dict["taxies"][taxi]["destinations"].keys())
        #divide drop_off_steps by the number of taxis
        drop_off_steps = drop_off_steps / len(state_as_dict["taxies"].keys())
        for passenger in state_as_dict["passengers_by_location"].keys():
            pick_up_steps += len(state_as_dict["passengers_by_location"][passenger]) * 2
        #divide pick_up_steps by the number of taxis
        pick_up_steps = pick_up_steps / len(state_as_dict["taxies"].keys())

        max_dist_unpicked= 0
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                max_dist_unpicked = max(max_dist_unpicked, self.distance_matrix[str(location)][
                    str(tuple(state_as_dict["passengers"][passenger]["destination"]))])

        max_dist_picked = 0
        for taxi in state_as_dict["taxies"].keys():
            location = (state_as_dict["taxies"][taxi]["location"])
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                max_dist_picked = max(max_dist_picked,self.distance_matrix[str(tuple((location)))][
                    str(tuple((state_as_dict["taxies"][taxi]["destinations"][passenger])))])

        max_dist = max(max_dist_picked, max_dist_unpicked)
        #    print(max_dist, pick_up_steps, drop_off_steps)
        return max_dist + pick_up_steps + drop_off_steps


    def h_6(self, node):
        tasks = 0
        points_of_interest_taxi = []
        points_of_interest_not_taxi = []
        state_as_dict = json.loads(node.state)
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                tasks += 2
                points_of_interest_not_taxi.append(location)
                points_of_interest_not_taxi.append(tuple(state_as_dict["passengers"][passenger]["destination"]))

        for taxi in state_as_dict["taxies"].keys():
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                tasks += 1
                points_of_interest_not_taxi.append(tuple(state_as_dict["taxies"][taxi]["destinations"][passenger]))
            points_of_interest_taxi.append(str(tuple(state_as_dict["taxies"][taxi]["location"])))

        #calculate distances between points of interest using distance_matrix_fw
        distances = []
        for i in range(len(points_of_interest_not_taxi)):
            for j in range(i+1, len(points_of_interest_not_taxi)):
                distances.append(self.distance_matrix_fw[str(points_of_interest_not_taxi[i])][str((points_of_interest_not_taxi)[j])])

        #for each taxi location, find the distance to the closest point of interest
        distances_taxi = []
        for taxi in points_of_interest_taxi:
            closest_distance = None
            for point in points_of_interest_not_taxi:
                distance = self.distance_matrix_fw[taxi][str(point)]
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
            distances_taxi.append(closest_distance)

        #sort distances and distances_taxi
        distances.sort()
        distances_taxi.sort()
        total_dists = distances_taxi + distances
        return (tasks+sum(total_dists[:tasks])) / len(state_as_dict["taxies"].keys())

    def h_7(self, node):
        if self.goal_test(node.state):
            return 0
        # self.control_print(node)
        return self.h_6(node)

    def h_7_2(self, node):
        if self.goal_test(node.state):
            return 0

        state_as_dict = json.loads(node.state)
        #make a list of all the points of unpicked passengers
        locations_unpicked = []
        taxi_distances = []
        tasks = 0
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                locations_unpicked.append(location)

        #for each taxi, make a points_of_interest list that includes its destinations,
        #and if it is not full, append locations_unpicked to it
        for taxi in state_as_dict["taxies"].keys():
            points_of_interest_taxi = []
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                tasks += 1
                points_of_interest_taxi.append(tuple(state_as_dict["taxies"][taxi]["destinations"][passenger]))
            if len(points_of_interest_taxi) < state_as_dict["taxies"][taxi]["max_passengers"]:
                points_of_interest_taxi += locations_unpicked
            #using distance_matrix_fw, calculate the minimal distance to a point of interest for each taxi
            closest_distance = None
            if points_of_interest_taxi == []:
                continue
            for point in points_of_interest_taxi:
                distance = self.distance_matrix_fw[str(tuple(state_as_dict["taxies"][taxi]["location"]))][str(point)]
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
            taxi_distances.append(closest_distance)
        taxi_distances.sort()

        #make a points_of_interest list, and tasks like h_6
        points_of_interest = []
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                tasks += 2
                points_of_interest.append(location)
                points_of_interest.append(tuple(state_as_dict["passengers"][passenger]["destination"]))
        #calculate distances between points of interest using distance_matrix_fw
        distances = []
        for i in range(len(points_of_interest)):
            for j in range(i+1, len(points_of_interest)):
                distances.append(self.distance_matrix_fw[str(points_of_interest[i])][str((points_of_interest)[j])])
        #sort distances and distances_taxi
        distances.sort()
        if distances != []:
            delta_distances = [distances[0]]
            if len(distances) > 1:
                for i in range(1, len(distances)):
                    delta_distances.append(distances[i] - distances[i-1])
                total_dist = taxi_distances + delta_distances
                return (tasks+sum(total_dist[:tasks])) / len(state_as_dict["taxies"].keys())
    #            els

    def h_8(self, node):
        edges_color_dict = self.edges_color_dict.copy()
        tasks = 0
        points_of_interest_taxi = []
        points_of_interest_not_taxi = []
        state_as_dict = json.loads(node.state)
        for location in state_as_dict["passengers_by_location"].keys():
            for passenger in state_as_dict["passengers_by_location"][location]:
                tasks += 2
                points_of_interest_not_taxi.append(location)
                points_of_interest_not_taxi.append(str(tuple(state_as_dict["passengers"][passenger]["destination"])))

        for taxi in state_as_dict["taxies"].keys():
            for passenger in state_as_dict["taxies"][taxi]["destinations"].keys():
                tasks += 1
                points_of_interest_not_taxi.append(tuple(state_as_dict["taxies"][taxi]["destinations"][passenger]))
            points_of_interest_taxi.append(str(tuple(state_as_dict["taxies"][taxi]["location"])))

        # calculate distances between points of interest using distance_matrix_fw
        distances = {}
        for i in range(len(points_of_interest_not_taxi)):
            distances[str(points_of_interest_not_taxi[i])] = {}
            for j in range(i + 1, len(points_of_interest_not_taxi)):
                distances[str(points_of_interest_not_taxi[i])][str(points_of_interest_not_taxi[j])] = self.distance_matrix_fw[str(points_of_interest_not_taxi[i])][str((points_of_interest_not_taxi)[j])]


        # for i in range(len(points_of_interest_not_taxi)):
        #     for j in range(i + 1, len(points_of_interest_not_taxi)):
        #         distances.append(
        #             self.distance_matrix_fw[str(points_of_interest_not_taxi[i])][str((points_of_interest_not_taxi)[j])])

        # for each taxi location, find the distance to the closest point of interest
        closest_to_taxi = {}
        for taxi in points_of_interest_taxi:
            #find the closest point of interest to the taxi and add it to the distances_taxi dict
            closest_distance = None
            for point in points_of_interest_not_taxi:
                distance = self.distance_matrix_fw[taxi][str(point)]
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_to_taxi[(taxi,point)] = distance

        #count the number of keys that have a value in closest_to_taxi
        num_of_taxies_distances = 0
        for taxi in closest_to_taxi.keys():
            if closest_to_taxi[taxi] is not None:
                num_of_taxies_distances += 1

        tasks -= num_of_taxies_distances
        #get the keys of the 'tasks' smallest distances in distances
        smallest_distances = []
        for key in distances.keys():
            for key2 in distances[key].keys():
                smallest_distances.append((key, key2, distances[key][key2]))
        #sort smallest distances by the third element of the tuple
        smallest_distances.sort(key=lambda x: x[2])

        smallest_distances = smallest_distances[:tasks]

        paths_to_paint = []
        for x in smallest_distances:
            self.paint_edges(edges_color_dict, self.edges_of_shortest_path[(x[0], x[1])])

        smallest_distances_taxi = [(k, v) for k, v in sorted(closest_to_taxi.items(), key=lambda item: item[1])]
        smallest_distances_taxi = smallest_distances_taxi[:num_of_taxies_distances]

        for x in smallest_distances_taxi:
            a = x[0][0]
            b = x[0][1]
            if type(a) != str:
                a = str(a)
            if type(b) != str:
                b = str(b)
            self.paint_edges(edges_color_dict, self.edges_of_shortest_path[(a,b)])

        #count the number of edges that are painted
        painted_edges = 0
        for edge in edges_color_dict.keys():
            if edges_color_dict[edge]:
                painted_edges += 1

        to_return = (painted_edges + tasks + num_of_taxies_distances) / len(state_as_dict["taxies"].keys())
        return to_return
    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_taxi_problem(game):
    return TaxiProblem(game)
