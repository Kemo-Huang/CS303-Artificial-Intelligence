from numpy import empty
from random import random


class Vertex:
    def __init__(self, _id):
        self.out_adjacent = {}
        self.in_adjacent = {}
        self.in_degree = 0
        self.id = _id

    def __str__(self):
        return str((self.out_adjacent, self.in_adjacent))

    def add_neighbor(self, neighbor, weight=0):
        self.out_adjacent[neighbor] = weight

    def add_incoming(self, incoming, weight=0):
        self.in_adjacent[incoming] = weight


class Graph:
    def __init__(self, n: int, data: list, is_ic: bool):
        self.vert_list = empty(n + 1, dtype=Vertex)
        for i in range(n + 1):
            self.vert_list[i] = Vertex(i)
        if is_ic:
            for e in data:
                self.vert_list[e[0]].add_neighbor(e[1], e[2])
                self.vert_list[e[1]].in_degree += 1
        else:
            for e in data:
                self.vert_list[e[0]].add_neighbor(e[1], e[2])
                self.vert_list[e[1]].add_incoming(e[0], e[2])
                self.vert_list[e[1]].in_degree += 1

    def __str__(self):
        result = ''
        for vertex in self.vert_list:
            result += str(vertex) + '\n'
        return result

    def get_total_weight_of_incoming_active(self, v: int, activated):
        neighbors = self.vert_list[v].in_adjacent
        total_weight = 0
        for vertex in neighbors.keys():
            if vertex in activated:
                total_weight += neighbors[vertex]
        return total_weight

    def get_inactive_neighbors_of(self, v: int, activated):
        """
        :return: a list with tuples of neighbor's id and weight
        """
        neighbors_dict = self.vert_list[v].out_adjacent
        res = []
        neighbors = neighbors_dict.keys()
        for vertex in neighbors:
            if vertex not in activated:
                res.append((vertex, neighbors_dict[vertex]))
        return res

    def independent_cascade(self, seed):
        activity_set = seed[:]
        activated = seed[:]
        while activity_set:
            new_activity_set = set()
            for active in activity_set:
                inactive_neighbors = self.get_inactive_neighbors_of(active, activated)
                for neighbor in inactive_neighbors:
                    if random() < neighbor[1]:
                        new_activity_set.add(neighbor[0])
            activated += list(new_activity_set)
            activity_set = new_activity_set
        return len(activated)

    def linear_threshold(self, seed, threshold):
        activity_set = seed[:]
        activated = seed[:]
        while activity_set:
            new_activity_set = set()
            for active in activity_set:
                inactive_neighbors = self.get_inactive_neighbors_of(active, activated)
                for inactive in inactive_neighbors:
                    inactive_id = inactive[0]
                    w_total = self.get_total_weight_of_incoming_active(inactive_id, activated)
                    if threshold[inactive_id] < w_total:
                        new_activity_set.add(inactive_id)
            activated += list(new_activity_set)
            activity_set = new_activity_set
        return len(activated)
