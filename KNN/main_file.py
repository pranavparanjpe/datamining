import csv
import math
import operator as op
import numpy as np

def csv_reader(file_path):                                      #CSV file Reader
    x=[]
    y=[]
    with open(file_path,"r") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            if reader.line_num != 1:
                x.append(row[1:])
                y.append([row[-1]])
    #x_data = np.matrix(x,dtype=float)
    #y_data = np.matrix(y,dtype=float)
    x_data = [list(map(float, row)) for row in x]
    y_data = [list(map(float, row)) for row in y]
    #print(" ",x_data)
    return(x_data,y_data)

#---------------------FUNCTION TO CALCULATE DISTANCE----------------------------------------

def euc_dist(train_row,test_row,k_val):
    dist = 0
    for i in range(len(train_row)-1):
        #dist=dist+pow((test_row[i] - train_row[i]),2)
        sum=test_row[i] - train_row[i]
        dist=dist+(sum*sum)
    fin_dist=math.sqrt(dist)
    return(fin_dist)


#---------------------FUNCTION TO CALCULATE DISTANCE FOR EACH ROW---------------------------
def calc_dist(train_data,test_row,k_val):
    rows=(len(train_data))
    #print(" ",rows)
    dist_values=[]
    for row in range(rows):
        dist=euc_dist(train_data[row],test_row,k_val)
        temp = []
        temp.extend(train_data[row])
        temp.append(dist)
        dist_values.append(temp)
    dist_values.sort(key=op.itemgetter(len(dist_values[0])-1))
    val=get_prediction(dist_values,k_val)
    return(val)


#-----------------------------------------------PREDICTION------------------------------------------------------------
def get_prediction(dist_val,k_val):
    yes=0
    no=0
    temp = dist_val[:k_val]
    for row in temp:
        val=row[-2]
        if(val==1):
            yes=yes+1
        else:
            no=no+1
    if(yes>no):
        return (1)
    else:
        return (0)


def z_normalization(x_data):
    numpy_data=np.asarray(x_data,dtype=float)
    col_len=len(x_data[0])-1
    mean_array=[]
    std_array=[]
    normalized_x_data=[]
    print (" ",col_len)
    for i in range(col_len):
        mean=numpy_data[:,i].mean()
        mean_array.append(mean)
        #print(" ",numpy_data[:,i].mean())
        std=numpy_data[:, i].std()
        std_array.append(std)
        #print(" ",numpy_data[:, i].std())
    mean_array.append(0)
    std_array.append(1)
    #print(" ", mean_array)
    final_data = []
    for row in numpy_data:
        #print(" ")
        normalized_x_data=(row-mean_array)/std_array
        final_data.append(normalized_x_data.tolist())
    #print(" ",final_data)
    return(mean_array,std_array,final_data)


#--------------------------MAIN METHOD------------------------------------------------------

if __name__ == "__main__":
    csv_path = "./spam_train.csv"
    x_data,y_data=csv_reader(csv_path)                        #input train datafile
    csv_test_path="./spam_test.csv"                           #input test datafile
    x_test_data,y_test_data=csv_reader(csv_test_path)

    k_arr=[1,5,11,21,41,61,81,101,201,401]                    #Random K values for how many nearest neighbors to choose
    #k_arr = [1]
#---------------------------------------------------------------------------------------------------------------------
    for k in k_arr:
        count = 0
        for row in x_test_data:
            #print(" ",row[:,[0,1]])
            #print(" ", row.shape)
            prediction_count=calc_dist(x_data,row,k)
            if (row[-1] == prediction_count):
                count = count + 1
        #print(" ",x_data)
        accuracy=count/len(x_test_data) *100
        print(" " ,accuracy)
#--------------------------------------------------------------------------------------------------------------------
    mean_array,std_array,normalized_x_data=z_normalization(x_data)
    numpy_test_data = np.asarray(x_test_data, dtype=float)
    for k in k_arr:
        count = 0
        for row in numpy_test_data:
            row1 = (row - mean_array) / std_array
            #print(" ",row[:,[0,1]])
            #print(" ", row.shape)
            prediction_count=calc_dist(normalized_x_data,row1,k)
            if (row[-1] == prediction_count):
                count = count + 1
        #print(" ",x_data)
        accuracy=count/len(x_test_data) *100
        print(" " ,accuracy)
#---------------------------------------------------------------------------------------------------------------------
    x1=[]
    with open("./spam_test.csv","r") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            if reader.line_num != 1:
                x1.append(row[:1])
    x1=[item for sublist in x1 for item in sublist]
    instances=50
    prediction_val=[]
    mean_array,std_array,normalized_x_data=z_normalization(x_data)
    numpy_test_data = np.asarray(x_test_data, dtype=float)

    for i in range (instances):
        row1 = (numpy_test_data[i] - mean_array) / std_array
        del prediction_val[:]
        prediction_val.append(x1[i])
        for k in k_arr:
            #print(" ",x_test_data[i])
            output_set = calc_dist(normalized_x_data, row1, k)
            if (output_set== 1):
                prediction_val.append("spam")
            else:
                prediction_val.append("no")
        print(', '.join(prediction_val))



