import csv
import numpy as np
import matplotlib.pyplot as matplot
from multiply_method import multiply

from fixed_lambda import *
from random_generator import random_selector


def csv_reader(file_path):
    x = []
    y = []

    with open(file_path, "r") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            if reader.line_num != 1:
                x.append(row[:-1])
                y.append([row[-1]])
    x = [[1] + x1 for x1 in x]


    return x,y


def convert_to_matrix(train_x, train_y,test_x, test_y):

    x_train_data = np.matrix(train_x,dtype=float)
    y_train_data = np.matrix(train_y,dtype=float)
    test_x_data = np.matrix(test_x, dtype=float)
    test_y_data = np.matrix(test_y, dtype=float)

    return x_train_data, y_train_data,test_x_data, test_y_data


def plot_graph(lambda_array, train_array, test_array):
    matplot.plot(lambda_array, test_array)
    matplot.plot(lambda_array, train_array)
#    matplot.show()


def cross_validation(train_file):
    train_x,train_y=csv_reader(train_file)
    train_x_matrix=np.matrix(train_x,dtype=float)
    train_y_matrix=np.matrix(train_y,dtype=float)
    num_records,y=train_x_matrix.shape

    fold_size=num_records//10

    test_mse_array =[]
    train_mse_array =[]
    for k in range(0, 151):
        train_avg=0
        test_avg=0
        for i in range(0 ,10):
            train_x_data=np.matrix(list(train_x[0: (i*fold_size)]) +list(train_x[((i+1)*fold_size):]),dtype=float)
            test_x_data=np.matrix(list(train_x[(i*fold_size) : ((i+1)*fold_size)]),dtype=float)
            train_y_data=np.matrix(list(train_y[0: (i*fold_size)]) +list(train_y[((i+1)*fold_size):]),dtype=float)
            test_y_data=np.matrix(list(train_y[(i*fold_size) : ((i+1)*fold_size)]),dtype=float)
            train_mse,w=calculate_w_for_test(train_x_data,train_y_data,k)
            test_mse=MSE_calc(test_x_data, test_y_data, w)
            train_avg=train_avg+train_mse
            test_avg=test_avg+test_mse

        test_mse_array.append(test_avg/10)
        train_mse_array.append(train_avg/10)
    print ("Minimum Value By Cross validation:::",test_mse_array.index(min(test_mse_array)))



def l2_regression(train_csv, test_csv, plot_name):
    csv_path = train_csv
    csv_path_test=test_csv

    train_x, train_y = csv_reader(csv_path)
    test_x, test_y = csv_reader(csv_path_test)
    x_train_data, y_train_data, test_x_data, test_y_data=convert_to_matrix(train_x, train_y,test_x, test_y)
    lambda_array, train_array, test_array = multiply(x_train_data, y_train_data, test_x_data, test_y_data)
    print("Minimum Lambda Value for" +test_csv+":::", test_array.index(min(test_array)))
    print("Minimum Lambda Value for" + train_csv+":::", train_array.index(min(train_array)))
    matplot.axis([0, 152, 0, 18])
    matplot.plot(lambda_array, train_array)
    matplot.plot(lambda_array, test_array)
    matplot.savefig(plot_name)
    matplot.close()


# -----------------------------------------------MAIN------------------------------------------------------------------
if __name__ == "__main__":

    d_size = []
    train_data = []
    test_data = []
    train_data46 = []
    test_data46 = []
    train_data150 = []
    test_data150 = []


    l2_regression("./data/train-100-10.csv","./data/test-100-10.csv","100-101.png")
    l2_regression("./data/train-100-100.csv","./data/test-100-100.csv","100-1001.png")
    # l2_regression("./data/50(1000)_100_train.csv", "./data/test-1000-100.csv","50(1000)-100.png")
    # l2_regression("./data/100(1000)_100_train.csv", "./data/test-1000-100.csv","100(1000)-100.png")
    # l2_regression("./data/150(1000)_100_train.csv", "./data/test-1000-100.csv","150(1000)-100.png")
    # l2_regression("./data/train-1000-100.csv", "./data/test-1000-100.csv","1000-100.png")
    # l2_regression("./data/train-wine.csv", "./data/test-wine.csv","wine.png")

    csv_path = "./data/train-1000-100.csv"
    csv_path_test = "./data/test-1000-100.csv"
    size = 50
    for k in range(50, 1001, 50):
        #sum_test1 = 0
        #sum_train1 = 0
        sum_test46 = 0
        sum_train46 = 0
        #sum_test150 = 0
        #sum_train150 = 0
        for count1 in range(0, 10):
            train_x, train_y = random_selector(csv_path, k)
            x_train_data = train_x
            y_train_data = train_y
            test_x, test_y = csv_reader(csv_path_test)
            test_x_data = np.matrix(test_x, dtype=float)
            test_y_data = np.matrix(test_y, dtype=float)

            #train_val1, test_val1 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 1)
            #sum_train1 = train_val1 + sum_train1
            #sum_test1 = test_val1 + sum_test1

            train_val46, test_val46 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 46)
            sum_train46 = train_val46 + sum_train46
            sum_test46 = test_val46 + sum_test46

            #train_val150, test_val150 = fixed_lambda_curve(x_train_data, y_train_data, test_x_data, test_y_data, 150)
            #sum_train150 = train_val150 + sum_train150
            #sum_test150 = test_val150 + sum_test150

        avg_46_test = sum_test46 / 10
        avg_46_train = sum_train46 / 10
        test_data46.append(avg_46_test)
        train_data46.append(avg_46_train)
        d_size.append(k)
        #avg_1_test = sum_test1 / 10
        #avg_150_test = sum_test150 / 10
        #avg_1_train = sum_train1 / 10
        #avg_150_train = sum_train150 / 10
        #test_data.append(avg_1_test)
        #train_data.append(avg_1_train)
        #test_data150.append(avg_150_test)
        #train_data150.append(avg_150_train)


    matplot.plot(d_size, train_data46)
    matplot.plot(d_size, test_data46)
    matplot.savefig("test_46.jpg")
    matplot.close()
    #matplot.plot(d_size, train_data)
    #matplot.plot(d_size, test_data)
    #matplot.savefig("1test.jpg")
    #matplot.close()
    #matplot.plot(d_size, train_data150)
    #matplot.plot(d_size, test_data150)
    #matplot.savefig("test_150.jpg")
    #matplot.close()
# ---------------------------------END PROBLEM 2-----------------------------------------------------------------------

    cross_validation("./data/train-100-10.csv" )               #cross-validation method function call for each dataset
    # cross_validation("./data/train-100-100.csv" )
    # cross_validation("./data/train-1000-100.csv")
    # cross_validation("./data/50(1000)_100_train.csv" )
    # cross_validation("./data/100(1000)_100_train.csv" )
    # cross_validation("./data/150(1000)_100_train.csv" )
    # cross_validation("./data/train-wine.csv" )

# ---------------------------------------------------------------------------------------------------------------------
