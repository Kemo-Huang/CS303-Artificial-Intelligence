# influence maximization

import config
from numpy import zeros
from argparse import ArgumentParser
from time import time
from graph import Graph
from heapq import heappush
from numpy.random import rand


def evaluate(seeds, model, is_ic, n):
    _sum = 0
    rounds = 0
    cnt = 0
    max_cnt = config.ISE_MAX_COUNT
    threshold = config.IMP_RESULT_THRESHOLD

    def fix(s):
        return model(s, rand(n + 1))

    if is_ic:
        run = model
    else:
        run = fix
    result = run(seeds)
    for _ in range(config.IMP_MAX_ROUNDS):
        _sum += run(seeds)
        rounds += 1
        cur_res = _sum / rounds
        if abs(result - cur_res) < threshold * result:
            cnt += 1
        else:
            cnt = 0
        result = cur_res
        if cnt == max_cnt:
            break
    return result


def celf(n: int, k: int, model, allow_time: int, is_ic):
    stop_time = allow_time + time()
    s = []
    q = []
    flag = zeros(n + 1, dtype=int)
    for v in range(1, n + 1):
        avg = -evaluate([v], model, is_ic, n)  # for max heap
        heappush(q, (avg, v))
    eva_s = 0
    len_s = 0
    while len_s < k and q:
        u = q.pop(0)[1]
        if flag[u] == len_s:
            s.append(u)
            eva_s = evaluate(s, model, is_ic, n)
            len_s += 1
        else:
            avg = evaluate(s + [u], model, is_ic, n)
            new_mg = eva_s - avg  # for max heap
            flag[u] = len_s
            heappush(q, (new_mg, u))
        if time() >= stop_time:
            remain = q[:k - len_s]
            for r in remain:
                s.append(r[1])
            break
    return s


# def cmp(a, b):
#     if a.in_degree > b.in_degree:
#         return -1
#     elif a.in_degree < b.in_degree:
#         return 1
#     return 0


def main():
    start_time = time()
    parser = ArgumentParser()
    parser.add_argument('-i', '--instance')
    parser.add_argument('-k', '--size')
    parser.add_argument('-m', '--model')
    parser.add_argument('-t', '--time')
    args = parser.parse_args()
    with open(args.instance) as fp:
        first_line = fp.readline().split()
        n = int(first_line[0])
        m = int(first_line[1])
        edges = []
        for i in range(m):
            line = fp.readline().split()
            edges.append((int(line[0]), int(line[1]), float(line[2])))
    if args.model == 'IC':
        graph = Graph(n, edges, True)
        model = graph.independent_cascade
        remain_time = config.IMP_IC_REMAIN_TIME
        is_ic = True
    else:
        graph = Graph(n, edges, False)
        model = graph.linear_threshold
        remain_time = config.IMP_LT_REMAIN_TIME
        is_ic = False
    total_time = int(args.time or 60)  # default time limitation
    allow_time = total_time - time() + start_time - remain_time
    k = int(args.size)
    max_seeds = celf(n, k, model, allow_time, is_ic)
    for seed in max_seeds:
        print(seed)
    print(time()-start_time)


if __name__ == '__main__':
    main()
