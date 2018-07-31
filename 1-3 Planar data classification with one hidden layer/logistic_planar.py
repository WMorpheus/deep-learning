import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


X, Y = load_planar_dataset()

def initialize_parameters(n_x,n_h,n_y): 
    np.random.seed(2)
    
    W1=np.random.randn(n_h,n_x)
    b1=np.zeros((n_h,1))
    
    W2=np.random.randn(n_y,n_h)
    b2=np.zeros((n_y,1))
    
    parameters={
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters
	
def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2 
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
	
def compute_cost(A2, Y, parameters):


    m = Y.shape[1] 

    cost = -(1.0/m)*np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))
 
    cost = np.squeeze(cost)    

    return cost
	
def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2=A2-Y
    dW2=1/m*np.dot(dZ2,A1.T)
    db2=1/m*np.sum(dZ2)
    
    dgZ1=1-np.power(A1,2)
    dZ1=np.dot(W2.T,dZ2)*dgZ1
    dW1=1/m*np.dot(dZ1,X.T)
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads
	
def update_parameters(parameters, grads, learning_rate = 1.2):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
	
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    costs=[]

    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = compute_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        
        if i%1000==0:
            costs.append(cost)
            
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters,costs
	
def predict(parameters,X):
    
    A2,cache=forward_propagation(X,parameters)
    
    predictions=(A2>0.5)
    
    return predictions

if __name__=='__main__':
	parameters,costs=nn_model(X,Y,n_h=4,print_cost=True)

	plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
	plt.title("Decision Boundary for hidden layer size " + str(4))