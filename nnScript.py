import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from math import log
import pickle
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    #size= (z.shape[0], z.shape[1])
    #m= np.ones(size)

    return  1/(1+np.exp(-z))#your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('/home/mohit/Downloads/ML/basecode/mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    
    #Your code here
    
    #file.write('\n preprocess Started')
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    for i in range(10):
        m = mat.get('train'+str(i))
        if i==0:
            a = range(m.shape[0])
            aperm = np.random.permutation(a)
            A1 = m[aperm[0:1000],:]
            A2 = m[aperm[1000:],:]
            train_data= A2
            t2= np.zeros((A2.shape[0],10))
            t2[:,i]=1
            train_label= t2
            validation_data= A1
            v2= np.zeros((A1.shape[0],10))
            v2[:,i]=1
            validation_label= v2
        else:
            a = range(m.shape[0])
            aperm = np.random.permutation(a)
            A1 = m[aperm[0:1000],:]
            A2 = m[aperm[1000:],:]
            train_data= np.concatenate((train_data,A2),0)
            t2= np.zeros((A2.shape[0],10))
            t2[:,i]=1
            train_label= np.concatenate((train_label,t2 ),0) 
            validation_data= np.concatenate((validation_data,A1),0)
            v2= np.zeros((A1.shape[0],10))
            v2[:,i]=1
            validation_label= np.concatenate((validation_label,v2),0)
        n= mat.get('test'+str(i))
        if i==0:
            test_data= n
            t2= np.zeros((n.shape[0],10))
            t2[:,i]=1
            test_label= t2
        else:
            test_data= np.concatenate((test_data,n),0)
            t2= np.zeros((n.shape[0],10))
            t2[:,i]=1
            test_label= np.concatenate((test_label,t2 ),0) 
    train_data= np.double(train_data)
    train_data= train_data/255
    validation_data= np.double(validation_data)
    validation_data= validation_data/255
    test_data= np.double(test_data)
    test_data= test_data/255
    #file.write('\n preProcess finished')
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    
i =0
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    
    #Your code here
    #
    #
    #
    #
    #
   
    output= np.array([])
    n= (training_data.shape[0],1)
    columnSize= np.ones(n)
    training_data= np.concatenate((training_data,columnSize),1)
    s= (1,w1.shape[1])
    rowSizeForHidden= np.ones(s)
    w1= np.concatenate((w1,rowSizeForHidden),0)
    hidden_output_wts= w2.transpose()
    
    w1Transpose= w1.transpose()
    value3= np.dot(training_data,w1Transpose)
    hidden_layer_output= sigmoid(value3)
    hidden_layer_output[:,-1]=1
   
    outputValue2= np.dot(hidden_layer_output,hidden_output_wts)
    output= sigmoid(outputValue2)
               
    size_err_output= (hidden_layer_output.shape[1],10)
    temp_err_output= np.zeros(size_err_output)
    size_err_hidden= (training_data.shape[1],hidden_layer_output.shape[1]-1)
    temp_err_hidden= np.zeros(size_err_hidden)
    
    output_vector1= output- training_label
    temp_err_output= (np.dot((output_vector1.transpose()), hidden_layer_output)) + (lambdaval*w2)
    final_error_at_output= temp_err_output/(training_data.shape[0])
    
    grad2_wts_add= np.dot(output_vector1, hidden_output_wts.transpose())
    hidden_output_wts_update= np.multiply((1-hidden_layer_output), hidden_layer_output)
    hidden_err_half= np.multiply(hidden_output_wts_update, grad2_wts_add)
    temp_err_hidden= (np.dot((hidden_err_half.transpose()), training_data)) + (lambdaval* w1)
    err_hidden= temp_err_hidden/(training_data.shape[0])
    final_error_at_hidden= np.delete(err_hidden, (err_hidden.shape[0]-1), 0)
    
    temp_sum_function1= np.multiply(training_label, np.log(output))
    temp_sum_function2= np.multiply((1-training_label), np.log((1-output)))
    reg_temp1= np.multiply(w1, w1)
    reg_temp2= np.multiply(w2, w2)
    reg_sem1= (lambdaval/2)*(np.ndarray.sum(reg_temp1)+ np.ndarray.sum(reg_temp2))
    reg_sem2= np.ndarray.sum(temp_sum_function1) + np.ndarray.sum(temp_sum_function2)
    
    obj_val= (reg_sem1- reg_sem2)/(training_data.shape[0])
       
    #print obj_val        
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad= np.concatenate((final_error_at_hidden.flatten(), final_error_at_output.flatten()),0)
    #print obj_grad
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    n= (data.shape[0],1)
    columnSize= np.ones(n)
    data= np.concatenate((data,columnSize),1)
    s= (1,785)
    rowSizeForHidden= np.ones(s)
    w1= np.concatenate((w1,rowSizeForHidden),0)
    hidden_output_wts= w2.transpose()
    w1Transpose= w1.transpose()
    value3= np.dot(data,w1Transpose)
    hidden_layer_output= sigmoid(value3)
    hidden_layer_output[:,-1]=1
    outputValue2= np.dot(hidden_layer_output,hidden_output_wts)
    output= sigmoid(outputValue2)
    
    labels= np.argmax(output, axis=1)
    
    return labels
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 1;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)
training_label= np.argmax(train_label, axis=1)
#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((1.0*predicted_label == 1.0*training_label))) + '%')
#file.write('\n Training set Accuracy:' + str(100*np.mean(1.0*(predicted_label == training_label))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset
validating_label= np.argmax(validation_label, axis=1)
print('\n Validation set Accuracy:' + str(100*np.mean(1.0*(predicted_label == validating_label))) + '%')
#file.write('\n Validation set Accuracy:' + str(100*np.mean(1.0*(predicted_label == validating_label))) + '%')

predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset
testing_label= np.argmax(test_label, axis=1)
print('\n Test set Accuracy:' + str(100*np.mean(1.0*(predicted_label == testing_label))) + '%')
#file.write('\n Test set Accuracy:' + str(100*np.mean(1.0*(predicted_label == testing_label))) + '%')
#file.close()

data= {"n_hidden" : n_hidden, "w1": w1, "w2": w2, "lambda" : lambdaval}
pickle.dump( data, open( "params.pickle", "wb" ) )
