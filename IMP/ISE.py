# influence spread computation

from argparse import ArgumentParser
from graph import Graph
from time import time
import config
from multiprocessing import Pool
from numpy.random import rand


def lt(arg):
    model, seeds, n = arg
    return model(seeds, rand(n + 1))


def evaluate(seeds, model, allow_time, is_ic, n):
    stop_time = allow_time + time()
    _sum = 0
    rounds = 0
    cnt = 0
    max_cnt = config.ISE_MAX_COUNT
    threshold = config.ISE_RESULT_THRESHOLD
    with Pool(config.N_PROCESSORS) as pool:
        if is_ic:
            activated_list = pool.imap_unordered(model, [seeds] * config.ISE_MAX_ROUNDS)
            result = model(seeds)
        else:
            activated_list = pool.imap_unordered(lt, [(model, seeds, n)] * config.ISE_MAX_ROUNDS)
            result = model(seeds, rand(n + 1))
        for activated in activated_list:
            _sum += activated
            rounds += 1
            cur_res = _sum / rounds
            if abs(result - cur_res) < threshold * result:
                cnt += 1
            else:
                cnt = 0
            result = cur_res
            if time() >= stop_time or cnt == max_cnt:
                break
    return result


def main():
    start_time = time()
    parser = ArgumentParser()
    parser.add_argument('-i', '--instance')
    parser.add_argument('-s', '--seed')
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
    with open(args.seed) as fp:
        seed = list(map(int, fp.read().split()))
    if args.model == 'IC':
        graph = Graph(n, edges, is_ic=True)
        model = graph.independent_cascade
        is_ic = True
    else:
        graph = Graph(n, edges, is_ic=False)
        model = graph.linear_threshold
        is_ic = False
    total_time = int(args.time or 60)  # default time limitation
    # start evaluation here
    allow_time = total_time - time() + start_time - config.ISE_REMAIN_TIME
    evaluation = evaluate(seed, model, allow_time, is_ic, n)
    print(evaluation)
    print(time() - start_time)


if __name__ == '__main__':
    main()
