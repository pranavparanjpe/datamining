import pandas
import random
import csv
import numpy as np

def random_selector(filename,k):
    matrice = []
    x = []
    y = []

    with open(filename,"r") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            if reader.line_num != 1:
                matrice.append(row)
    new_matrix = np.matrix(matrice)
    #print ("\n" ,new_matrix)


    csv_matrix=new_matrix[np.random.choice(new_matrix.shape[0],k,replace=False),:]
    #print("\n",csv_matrix)

    for row in csv_matrix:
        row_s=list(np.squeeze(np.asarray(row)))
        #print(type(row_s))
        x.append(['1'] + row_s[:-1])
        y.append([row_s[-1]])
    #x = [[1] + x1 for x1 in x]


    return np.matrix(x, dtype=float),np.matrix(y, dtype=float)

