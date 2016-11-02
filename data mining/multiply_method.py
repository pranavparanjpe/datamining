from MSE import *

def multiply(x,y,xtest,ytest):

    xtrans = x.transpose()                                #matrix transpose
    c = np.dot(xtrans , x)                                #matrix multiplication of X and its transpose
    lam = 0                                               #initial value of lambda assumed to start from 0
    rows,id_size=x.shape                                  #number of columns of the X matrix to find identity matrix size
    train_array=[]
    test_array=[]
    lambda_array=[]

    for i in range(0, 151):                                #Run loop for 150 values of lambda

        id= np.identity(id_size,dtype=float)
        id_mat= lam*id                                      #identity matrix X lambda value
        w_inter = np.add(c ,id_mat)                         #intermediate value of w
        inverse = np.linalg.inv(w_inter)
        w_val = np.dot(inverse , xtrans)                    #matrix multiplication of (XT.X +lam.I)-1 . XT
        w = np.dot(w_val , y)                               #w value final---matrix
        mean_square = MSE_calc(x, y, w)                     # Objective Function for for Train data
        train_array.append(mean_square)
        MSTest = MSE_calc(xtest,ytest,w)                    #Predicting Test Function value for given w
        print("lambda: "+str(i)+" "+str(MSTest))
        test_array.append(MSTest)
        lambda_array.append(i)                              #Plot purpose
        lam=lam+1
    return lambda_array,train_array,test_array
