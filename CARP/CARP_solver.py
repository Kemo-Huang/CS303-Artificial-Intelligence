import numpy as np
from argparse import ArgumentParser
from time import time
from random import getrandbits, randrange, random, sample
from copy import deepcopy
import config
from classes import Route, Solution
from math import log

start_time = time()
TERMINATION = 0
SEED = None
NAME = ''
VERTICES = DEPOT = REQUIRED_EDGES = NON_REQUIRED_EDGES \
    = VEHICLES = CAPACITY = TOTAL_COST = 0
DIST = None
IDX_BEGIN = 0
IDX_END = 1
IDX_COST = 2
IDX_DEMAND = 3


def input_data():
    global TERMINATION, SEED, NAME, VERTICES, DEPOT, \
        REQUIRED_EDGES, NON_REQUIRED_EDGES, \
        VEHICLES, CAPACITY, TOTAL_COST
    parser = ArgumentParser()
    parser.add_argument('instance')
    parser.add_argument('-t', '--termination')
    parser.add_argument('-s', '--seed')
    args = parser.parse_args()
    try:
        fp = open(args.instance, 'r')
    except IOError:
        print("File does not exist.")
        return
    TERMINATION = args.termination
    if not TERMINATION:
        TERMINATION = 60
    else:
        TERMINATION = int(TERMINATION)
    SEED = args.seed
    line = fp.readline()
    NAME = line.split(' : ')[1][:-1]
    info = []
    for _ in range(7):
        line = fp.readline()
        info.append(int(line.split(' : ')[1][:-1]))
    VERTICES = info[0]
    DEPOT = info[1]
    REQUIRED_EDGES = info[2]
    NON_REQUIRED_EDGES = info[3]
    VEHICLES = info[4]
    CAPACITY = info[5]
    TOTAL_COST = info[6]
    fp.readline()
    required_edges = []
    non_required_edges = []
    for _ in range(REQUIRED_EDGES):
        line = fp.readline()
        required_edges.append(list(map(int, line.split())))
    for _ in range(NON_REQUIRED_EDGES):
        line = fp.readline()
        non_required_edges.append(list(map(int, line.split())))
    fp.close()
    return required_edges, non_required_edges


def floyd(data):
    global DIST
    DIST = np.full((VERTICES + 1, VERTICES + 1), np.inf)
    vertices_range = range(1, VERTICES + 1)
    for i in vertices_range:
        DIST[i, i] = 0
    for edge in data:
        DIST[edge[IDX_BEGIN], edge[IDX_END]] = DIST[edge[IDX_END], edge[IDX_BEGIN]] = edge[IDX_COST]
    for k in vertices_range:
        for i in vertices_range:
            for j in vertices_range:
                DIST[i, j] = min(DIST[i, j], DIST[i, k] + DIST[k, j])


# Note that deepcopy is needed for 'free' before path scanning
def path_scanning(free: list):
    route = Route()
    i = DEPOT
    discount = config.discount
    while free:
        _d = np.inf
        _u = -1
        reverse = False
        length = len(free)
        for u in range(length):
            if route.load + free[u][IDX_DEMAND] > discount * CAPACITY:
                continue
            if DIST[i, free[u][IDX_BEGIN]] < _d:
                _d = DIST[i, free[u][IDX_BEGIN]]
                _u = u
                reverse = False
            elif DIST[i, free[u][IDX_END]] < _d:
                _d = DIST[i, free[u][IDX_END]]
                _u = u
                reverse = True
            elif getrandbits(1):
                if DIST[i, free[u][IDX_BEGIN]] == _d:
                    _u = u
                    reverse = False
                elif DIST[i, free[u][IDX_END]] == _d:
                    _u = u
                    reverse = True
        if _u != -1:
            edge = free[_u]
            if not reverse:
                route.required_arc_list.append([edge, False])
                i = edge[IDX_END]
            else:
                route.required_arc_list.append([edge, True])
                i = edge[IDX_BEGIN]
            route.load += edge[IDX_DEMAND]
            route.cost += _d + edge[IDX_COST]
            del free[_u]
        else:
            break
    route.cost += DIST[i, DEPOT]
    return route


def single_insertion(solution: Solution):
    route_list = solution.route_list
    remove_route_idx = randrange(0, len(route_list))
    remove_route = route_list[remove_route_idx]
    remove_arc_list = remove_route.required_arc_list
    remove_arc_idx = randrange(0, len(remove_arc_list))
    remove_arc = remove_arc_list[remove_arc_idx]
    demand = remove_arc[0][IDX_DEMAND]
    insert_route_idx = None
    for i in range(len(route_list)):
        if i != remove_route_idx and route_list[i].load + demand < CAPACITY:
            insert_route_idx = i
            break
    if insert_route_idx is not None:
        insert_route = route_list[insert_route_idx]
        insert_arc_list = insert_route.required_arc_list
        insert_arc_idx = randrange(0, len(insert_arc_list) + 1)
        insert_arc = remove_arc
        new_insert_route = Route()
        new_insert_arc_list_1 = insert_arc_list[:]
        new_insert_arc_list_2 = insert_arc_list[:]
        new_insert_arc_list_1.insert(insert_arc_idx, insert_arc)
        new_insert_arc_list_2.insert(insert_arc_idx, [insert_arc[0], not insert_arc[1]])

        cost1 = get_route_cost(new_insert_arc_list_1)
        cost2 = get_route_cost(new_insert_arc_list_2)
        if cost1 < cost2:
            new_insert_route.required_arc_list = new_insert_arc_list_1
            new_insert_route.cost = cost1
        else:
            new_insert_route.required_arc_list = new_insert_arc_list_2
            new_insert_route.cost = cost2
        new_insert_route.load = insert_route.load + demand

        new_route_list = route_list[:]
        new_remove_arc_list = remove_arc_list[:]
        new_remove_route = None
        del new_remove_arc_list[remove_arc_idx]
        # update insert route first
        new_route_list[insert_route_idx] = new_insert_route
        if not new_remove_arc_list:
            del new_route_list[remove_route_idx]
        else:
            new_remove_route = Route()
            new_remove_route.required_arc_list = new_remove_arc_list
            new_remove_route.load = remove_route.load - demand
            new_remove_route.cost = get_route_cost(new_remove_arc_list)
            new_route_list[remove_route_idx] = new_remove_route
        diff1 = (0 if new_remove_route is None else new_remove_route.cost) - remove_route.cost
        diff2 = new_insert_route.cost - insert_route.cost
        # print('cost1', diff1 + diff2)
        new_solution = Solution()
        new_solution.route_list = new_route_list
        new_solution.quality = solution.quality + diff1 + diff2
        # print(new_solution.quality, get_quality(new_route_list))
        return new_solution
    else:
        length = len(remove_arc_list)
        if length < 2:
            return solution
        insert_arc_idx = randrange(0, length - 1)
        if insert_arc_idx >= remove_arc_idx:
            insert_arc_idx += 1
        new_route = Route()
        new_route.load = remove_route.load
        arc_list_1 = remove_arc_list[:]
        arc_list_2 = remove_arc_list[:]
        arc_list_1.insert(insert_arc_idx, arc_list_1.pop(remove_arc_idx))
        del arc_list_2[remove_arc_idx]
        arc_list_2.insert(insert_arc_idx, [remove_arc[0], not remove_arc[1]])
        cost1 = get_route_cost(arc_list_1)
        cost2 = get_route_cost(arc_list_2)
        if cost1 < cost2:
            new_route.required_arc_list = arc_list_1
            new_route.cost = cost1
        else:
            new_route.required_arc_list = arc_list_2
            new_route.cost = cost2
        # new_route.required_arc_list = arc_list
        new_route_list = route_list[:]
        new_route_list[remove_route_idx] = new_route
        new_solution = Solution()
        new_solution.route_list = new_route_list
        diff = new_route.cost - remove_route.cost
        # print('cost2', diff)
        new_solution.quality = solution.quality + diff
        # print(new_solution.quality, get_quality(new_route_list))
        return new_solution


def swap(solution):
    route_list = solution.route_list
    route_list_length = len(route_list)
    route_idx_1 = randrange(0, route_list_length)
    route_1 = route_list[route_idx_1]
    route_idx_2 = randrange(0, route_list_length - 1)
    if route_idx_2 >= route_idx_1:
        route_idx_2 += 1
    route_2 = route_list[route_idx_2]
    arc_list_1 = route_1.required_arc_list
    arc_list_2 = route_2.required_arc_list
    arc_idx_1 = randrange(0, len(arc_list_1))
    load_1 = route_1.load
    load_2 = route_2.load
    arc_1 = arc_list_1[arc_idx_1]
    demand_1 = arc_1[0][IDX_DEMAND]
    flag = False
    arc_2 = arc_idx_2 = new_load_1 = new_load_2 = None
    for arc_idx_2 in range(0, len(arc_list_2)):
        arc_2 = arc_list_2[arc_idx_2]
        demand_2 = arc_2[0][IDX_DEMAND]
        new_load_1 = load_1 + demand_2 - demand_1
        new_load_2 = load_2 + demand_1 - demand_2
        if new_load_1 <= CAPACITY and new_load_2 <= CAPACITY:
            flag = True
            break
    # swapping two routes is better than swapping one route
    if flag:
        new_arc_list_11 = arc_list_1[:]
        new_arc_list_12 = arc_list_1[:]
        new_arc_list_21 = arc_list_2[:]
        new_arc_list_22 = arc_list_2[:]
        new_arc_list_11[arc_idx_1] = arc_2
        new_arc_list_12[arc_idx_1] = [arc_2[0], not arc_2[1]]
        new_arc_list_21[arc_idx_2] = arc_1
        new_arc_list_22[arc_idx_2] = [arc_1[0], not arc_1[1]]
        cost11 = get_route_cost(new_arc_list_11)
        cost12 = get_route_cost(new_arc_list_12)
        cost21 = get_route_cost(new_arc_list_21)
        cost22 = get_route_cost(new_arc_list_22)
        new_route_1 = Route()
        new_route_2 = Route()
        new_route_1.load = new_load_1
        new_route_2.load = new_load_2
        if cost11 < cost12:
            new_route_1.required_arc_list = new_arc_list_11
            new_route_1.cost = cost11
        else:
            new_route_1.required_arc_list = new_arc_list_12
            new_route_1.cost = cost12
        if cost21 < cost22:
            new_route_2.required_arc_list = new_arc_list_21
            new_route_2.cost = cost21
        else:
            new_route_2.required_arc_list = new_arc_list_22
            new_route_2.cost = cost22
        new_route_list = route_list[:]
        new_route_list[route_idx_1] = new_route_1
        new_route_list[route_idx_2] = new_route_2
        new_solution = Solution()
        new_solution.route_list = new_route_list
        new_solution.quality = solution.quality + new_route_1.cost - route_1.cost + new_route_2.cost - route_2.cost
        return new_solution
    else:
        new_route = Route()
        new_route.load = route_1.load
        arc_list_1_length = len(arc_list_1)
        if arc_list_1_length < 2:
            return solution
        arc_idx_2 = randrange(0, arc_list_1_length - 1)
        if arc_idx_2 >= arc_idx_1:
            arc_idx_2 += 1
        arc_2 = arc_list_1[arc_idx_2]
        new_arc_list_1 = arc_list_1[:]
        new_arc_list_2 = arc_list_1[:]
        new_arc_list_3 = arc_list_1[:]
        new_arc_list_4 = arc_list_1[:]
        new_arc_list_1[arc_idx_1] = arc_2
        new_arc_list_1[arc_idx_2] = arc_1
        new_arc_list_2[arc_idx_1] = [arc_2[0], not arc_2[1]]
        new_arc_list_2[arc_idx_2] = arc_1
        new_arc_list_3[arc_idx_1] = arc_2
        new_arc_list_3[arc_idx_2] = [arc_1[0], not arc_1[1]]
        new_arc_list_4[arc_idx_1] = [arc_2[0], not arc_2[1]]
        new_arc_list_4[arc_idx_2] = [arc_1[0], not arc_1[1]]

        cost1 = get_route_cost(new_arc_list_1)
        cost2 = get_route_cost(new_arc_list_2)
        cost3 = get_route_cost(new_arc_list_3)
        cost4 = get_route_cost(new_arc_list_4)

        min_cost = min(cost1, cost2, cost3, cost4)
        new_route.cost = min_cost
        if cost1 == min_cost:
            new_route.required_arc_list = new_arc_list_1
        elif cost2 == min_cost:
            new_route.required_arc_list = new_arc_list_2
        elif cost3 == min_cost:
            new_route.required_arc_list = new_arc_list_3
        else:
            new_route.required_arc_list = new_arc_list_4
        new_solution = Solution()
        new_route_list = route_list[:]
        new_route_list[route_idx_1] = new_route
        new_solution.route_list = new_route_list
        new_solution.quality = solution.quality + new_route.cost - route_1.cost
        return new_solution


def two_opt(solution: Solution):
    route_list = solution.route_list
    new_route_list = route_list[:]
    new_solution = Solution()
    route_idx_1 = randrange(0, len(route_list))
    route_idx_2 = randrange(0, len(route_list) - 1)
    if route_idx_2 >= route_idx_1:
        route_idx_2 += 1
    old_route_1 = route_list[route_idx_1]
    old_route_2 = route_list[route_idx_2]
    new_route_1, new_route_2 = two_opt_double(old_route_1, old_route_2)
    if new_route_1 is not None:
        new_route_list[route_idx_1] = new_route_1
        new_route_list[route_idx_2] = new_route_2
        new_solution.route_list = new_route_list
        new_solution.quality = solution.quality + new_route_1.cost - old_route_1.cost \
                               + new_route_2.cost - old_route_2.cost
    else:
        route_idx = randrange(0, len(route_list))
        old_route = route_list[route_idx]
        new_route = two_opt_single(old_route)
        new_route_list[route_idx] = new_route
        new_solution.route_list = new_route_list
        new_solution.quality = solution.quality + new_route.cost - old_route.cost
    return new_solution


def two_opt_single(route: Route):
    arc_list = route.required_arc_list
    # extract sub-route indices
    idx = sample(range(len(arc_list)), 2)
    if idx[0] < idx[1]:
        start, end = idx
    else:
        end, start = idx
    # create a new route
    new_route = Route()
    new_sub_list = deepcopy(arc_list[start: end + 1])
    for arc in new_sub_list:
        arc[1] = not arc[1]
    new_sub_list.reverse()
    new_arc_list = arc_list[: start] + new_sub_list + arc_list[end + 1:]
    new_route.required_arc_list = new_arc_list
    new_route.cost = get_route_cost(new_arc_list)
    new_route.load = route.load
    return new_route


def two_opt_double(route1: Route, route2: Route):
    arc_list1 = route1.required_arc_list
    arc_list2 = route2.required_arc_list
    idx1 = (len(arc_list1) - 1) // 2
    idx2 = (len(arc_list2) - 1) // 2
    half11 = arc_list1[:idx1]
    half12 = arc_list1[idx1:]
    half21 = arc_list2[:idx2]
    half22 = arc_list2[idx2:]
    load11 = 0
    for arc in half11:
        load11 += arc[0][IDX_DEMAND]
    load12 = route1.load - load11
    load21 = 0
    for arc in half21:
        load21 += arc[0][IDX_DEMAND]
    load22 = route2.load - load21
    l1 = load11 + load22
    l2 = load12 + load21
    l3 = load11 + load21
    l4 = load12 + load22
    new_route1 = new_route2 = new_route3 = new_route4 = None
    if l1 < CAPACITY and l2 < CAPACITY:
        new_route1 = Route()
        new_route1.required_arc_list = half11 + half22
        new_route1.load = l1
        new_route1.cost = get_route_cost(new_route1.required_arc_list)
        new_route2 = Route()
        new_route2.required_arc_list = half21 + half12
        new_route2.load = l2
        new_route2.cost = get_route_cost(new_route2.required_arc_list)
    if l3 < CAPACITY and l4 < CAPACITY:
        new_route3 = Route()
        reverse21 = deepcopy(half21)
        for arc in reverse21:
            arc[1] = not arc[1]
        reverse21.reverse()
        new_route3.required_arc_list = half11 + reverse21
        new_route3.load = l3
        new_route3.cost = get_route_cost(new_route3.required_arc_list)
        new_route4 = Route()
        reverse12 = deepcopy(half12)
        for arc in reverse12:
            arc[1] = not arc[1]
        reverse12.reverse()
        new_route4.required_arc_list = reverse12 + half22
        new_route4.load = l4
        new_route4.cost = get_route_cost(new_route4.required_arc_list)
    if new_route1 is None and new_route3 is None:
        return None, None
    elif new_route1 is None:
        return new_route3, new_route4
    elif new_route3 is None:
        return new_route1, new_route2
    else:
        if new_route1.cost + new_route2.cost < new_route3.cost + new_route4.cost:
            return new_route1, new_route2
        else:
            return new_route3, new_route4


def get_route_cost(arc_list):
    cost = 0
    pre = DEPOT
    for arc in arc_list:
        if not arc[1]:
            begin = arc[0][IDX_BEGIN]
            end = arc[0][IDX_END]
        else:
            begin = arc[0][IDX_END]
            end = arc[0][IDX_BEGIN]
        cost += DIST[pre, begin] + arc[0][IDX_COST]
        pre = end
    cost += DIST[pre, DEPOT]
    return cost


def fix_m(time_cost, times, coe):
    n = -log(coe, config.alpha)
    remaining_time = TERMINATION - (time() - start_time)
    m0 = int(remaining_time * times / (time_cost * n)) + 5
    return m0, remaining_time


def annealing(required_edges, non_required_edges):
    current_s = best = get_initial_solution(required_edges, non_required_edges)
    initial_solution_time = time()
    initial_solution_cost = initial_solution_time - start_time
    print('initial quality:', best.quality, 'time', initial_solution_cost)
    alpha = config.alpha
    m0 = config.m0
    temperature = config.start_temperature
    end_t = config.end_temperature
    timeout = config.before_timeout
    coe = 1000000
    for _ in range(6):
        times = 0
        timer = time()
        if coe == 1:
            m0 += 10
        while temperature > coe * end_t:
            for _ in range(m0):
                new_s = get_new_solution(current_s)
                # print(new_s.quality)
                if new_s is not None:
                    cost_diff = new_s.quality - current_s.quality
                    if cost_diff < 0:
                        # print('find better')
                        current_s = new_s
                        if new_s.quality < best.quality:
                            best = new_s
                            # print('find best')
                    elif random() < np.exp(-cost_diff / temperature):
                        current_s = new_s
            temperature *= alpha
            times += m0
        time_cost = time() - timer
        m0, left_time = fix_m(time_cost, times, coe)
        coe /= 10
        if left_time < timeout:
            break
    print('temp', temperature)
    print('m0', m0)
    return best


def get_initial_solution(required_edges, non_required_edges):
    floyd(required_edges + non_required_edges)
    solution_list = []
    for i in range(config.scanning_times):
        route_list = []
        free = deepcopy(required_edges)
        while free:
            route = path_scanning(free)
            route_list.append(route)
        solution = Solution()
        solution.route_list = route_list
        solution.quality = get_quality(route_list)
        solution_list.append(solution)
    best = solution_list[0]
    for i in range(1, len(solution_list)):
        if solution_list[i].quality < best.quality:
            best = solution_list[i]
    return best


def get_quality(route_list):
    quality = 0
    for route in route_list:
        quality += route.cost
    return int(quality)


def get_new_solution(s: Solution):
    new_solution_list = [single_insertion(s), swap(s), two_opt(s)]
    best_new = new_solution_list[0]
    for solution in new_solution_list:
        if solution.quality < best_new.quality:
            best_new = solution
    return best_new


if __name__ == '__main__':
    required, non_required = input_data()
    answer = 's '
    routes = annealing(required, non_required)
    for r in routes.route_list:
        answer += '0,'
        for task in r.required_arc_list:
            a = task[0]
            if not task[1]:
                answer += '({},{}),'.format(a[IDX_BEGIN], a[IDX_END])
            else:
                answer += '({},{}),'.format(a[IDX_END], a[IDX_BEGIN])
        answer += '0,'
    answer = answer[:-1] + '\nq ' + str(routes.quality)
    print(answer)
    print('time', time() - start_time)
