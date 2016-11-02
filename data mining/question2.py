import csv
import numpy as np
import matplotlib.pyplot as mplot
from fixed_lambda import fixed_lambda_curve
from random_generator import random_selector
from dm_1 import csv_reader

def q2():
    d_size=[]
    train_data=[]
    test_data=[]
    csv_path = "./data/train-1000-100.csv"
    csv_path_test = "./data/test-1000-100.csv"
    size = 50
    for k in range(50, 1001, 50):
        sum_test1 = 0
        sum_train1 = 0
        sum_test46 = 0
        sum_train46 = 0
        sum_test150 = 0
        sum_train150 = 0
        for count1 in range(0, 10):
            train_x, train_y = random_selector(csv_path, k)
            x_train_data = train_x
            y_train_data = train_y
            test_x, test_y = csv_reader(csv_path_test)
            test_x_data = np.matrix(test_x, dtype=float)
            test_y_data = np.matrix(test_y, dtype=float)

            train_val1, test_val1 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 1)
            sum_train1 = train_val1 + sum_train1
            sum_test1 = test_val1 + sum_test1

            train_val46, test_val46 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 46)
            sum_train46 = train_val46 + sum_train46
            sum_test46 = test_val46 + sum_test46

            train_val150, test_val150 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 150)
            sum_train150 = train_val150 + sum_train150
            sum_test150 = test_val150 + sum_test150

        avg_1_test = sum_test1 / 10
        avg_46_test = sum_test46 / 10
        avg_150_test = sum_test150 / 10
        avg_1_train = sum_train1 / 10
        avg_46_train = sum_train46 / 10
        avg_150_train = sum_train150 / 10
        test_data.append(avg_1_test)
        train_data.append(avg_1_train)
        d_size.append(k)
    mplot.plot(d_size,train_data)
    mplot.plot(d_size,test_data)
    mplot.savefig("1test.jpg")
    mplot.close()