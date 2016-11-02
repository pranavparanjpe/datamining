import numpy as np

def MSE_calc(x,y,w,):
    rows, columns=x.shape
    mse=0
    count=0
    wtrans = w.transpose()                              #Gtranspose of given w for finding the MSE
    for i in x:                                         #iterate over each row in given dataset
        x_transpose=i.transpose()                       #MSE calculation
        error = np.dot(wtrans,x_transpose)
        error_val = error.item(0,0)
        y_val = y.item(count,0)
        count=count+1
        sqr=error_val - y_val
        sqr=sqr ** 2
        mse=mse +sqr
        #mse = mse+error_val-y_val

    mse=mse/rows

    return mse
