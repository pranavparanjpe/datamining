from MSE import *

def fixed_lambda_curve(x,y,xtest,ytest,lam):
    xtrans = x.transpose()
    c = np.dot(xtrans, x)
    rows, id_size = x.shape

    id = np.identity(id_size, dtype=float)
    id_mat = lam * id                                                # identity matrix X lambda value
    w_inter = np.add(c, id_mat)                                     # intermediate value of w
    inverse = np.linalg.inv(w_inter)
    w_val = np.dot(inverse, xtrans)                              # matrix multiplication of (XT.X +lam.I)-1 . XT
    w = np.dot(w_val, y)                                        # w value final---matrix
    mean_square = MSE_calc(x, y, w)
    MSTest = MSE_calc(xtest, ytest, w)

    return mean_square,MSTest

def calculate_w_for_test(x,y,lam):
    xtrans = x.transpose()
    c = np.dot(xtrans, x)
    rows, id_size = x.shape

    id = np.identity(id_size, dtype=float)
    id_mat = lam * id                                       # identity matrix X lambda value
    w_inter = np.add(c, id_mat)                             # intermediate value of w
    inverse = np.linalg.inv(w_inter)
    w_val = np.dot(inverse, xtrans)                         # matrix multiplication of (XT.X +lam.I)-1 . XT
    w = np.dot(w_val, y)                                    # w value final---matrix
    mean_square = MSE_calc(x, y, w)
    return mean_square,w