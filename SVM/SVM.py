from numpy import dot, ndarray, c_, ones, sign, arange, shape, array
from numpy.random import shuffle, uniform
from multiprocessing import Pool
from time import time
from argparse import ArgumentParser
from sys import stdout
import os

epochs = 2500
learning_rate = 0.01
start_time = time()
parser = ArgumentParser()
parser.add_argument('train_data', type=str)
parser.add_argument('test_data', type=str)
parser.add_argument('-t', '--time_limit', type=int, default=10)
args = parser.parse_args()
time_limit = args.time_limit
train_x = []
train_y = []
with open(args.train_data) as file:
    line = file.readline()
    while line:
        data = line.split()
        train_x.append(list(map(float, data[:-1])))
        train_y.append(float(data[-1]))
        line = file.readline()
test_x = []
with open(args.test_data) as file:
    line = file.readline()
    while line:
        data = line.split()
        test_x.append(list(map(float, data[:-1])))
        line = file.readline()
arr_train_x, arr_train_y = array(train_x), array(train_y)
train_data_len = len(arr_train_x)
ordered_list = arange(train_data_len)
arr_train_x = c_[ones((arr_train_x.shape[0])), arr_train_x]


class SVM:
    def __init__(self):
        self.w = uniform(size=shape(arr_train_x)[1])

    @staticmethod
    def cal_sgd(x, y, w):
        return w - learning_rate * (-y * x) if y * dot(x, w) < 1 else w

    def fit(self, xy_list):
        for xy_zip in xy_list:
            for xi, yi in xy_zip:
                self.w = self.cal_sgd(xi, yi, self.w)

    def predict(self, x: ndarray):
        x_test = c_[ones((x.shape[0])), x]
        return sign(dot(x_test, self.w)).astype(int)


def mp_random_xy(_):
    randomize = ordered_list
    shuffle(randomize)
    return zip(arr_train_x[randomize], arr_train_y[randomize])


if __name__ == '__main__':
    p = Pool(8)
    svm = SVM()
    iter_start_time = time()
    svm.fit(p.map(mp_random_xy, range(epochs)))
    iter_time = time() - iter_start_time
    cost_time = time() - start_time
    while time_limit - cost_time > iter_time + 0.5:
        svm.fit(p.map(mp_random_xy, range(epochs)))
        cost_time = time() - start_time
    result = svm.predict(array(test_x))
    print("\n".join(result.astype(str)))
    # _sum = 0
    # for m in range(len(test_y)):
    #     _sum += abs(test_y[m] - result[m])
    # print(_sum / len(test_y))
    # print(time() - start_time)
    stdout.flush()
    os._exit(0)
