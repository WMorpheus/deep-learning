import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#处理训练集和测试集
train_num=train_set_x_orig.shape[0]   #训练集数目
train_set_x_flatten=train_set_x_orig.reshape(train_num,-1).T
train_set_x=train_set_x_flatten/255   #像素值为0-255
test_num=test_set_x_orig.shape[0]     #测试集数目
test_set_x_flatten=test_set_x_orig.reshape(test_num,-1).T
test_set_x=test_set_x_flatten/255

#定义sigmoid函数
def sigmoid(z):
    a=1.0/(1+1/np.exp(z))
    return a

#单次forward and backward propagation
def propagation(w,b,X,Y):
    m=X.shape[1]   #训练集数目
    
    Z=np.dot(w.T,X)+b
    A=sigmoid(Z)
    
    cost=-1.0/m*np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))
    
    dw=1/m*np.dot(X,(A-Y).T)
    db=1/m*np.sum(A-Y)

    return cost,dw,db

#logistic回归，训练w，b
def optimize(w,b,X,Y,iteration_num,learning_rate,print_cost=False):
    
    costs=[]
    
    for i in range(iteration_num):
        cost,dw,db=propagation(w,b,X,Y)
        
        w=w-learning_rate*dw
        b=b-learning_rate*db
        
        if i%100==0:
            costs.append(cost)
            
        if i%100==0 and print_cost==True:
            
            print(str(i)+' iteration num get cost '+str(cost))
            
    return costs,w,b,dw,db,cost

#测试参数w，b的预测结果
def predict(w,b,X):
    
    num=X.shape[1]
    Y_prediction=np.zeros((1,num))
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    for i in range(num):
        if A[0][i]<0.5:
            Y_prediction[0][i]=0
        else:
            Y_prediction[0][i]=1
    
    return Y_prediction  
	
#模型主函数	
def model(X_train,Y_train,X_test,Y_test,iteration_num=2000,learning_rate=0.5,print_cost=False):
    
    dim=X_train.shape[0]       #维度
    train_num=X_train.shape[1]     #训练集数目
    
    w=np.zeros((dim,1))     #初始化w，b
    b=0
    
    costs,w,b,dw,db,cost=optimize(w,b,X_train,Y_train,iteration_num,learning_rate,print_cost)       #logistic回归，训练w，b
        #测试参数w,b的效果
    train_Y_prediction=predict(w,b,X_train)
    train_correction=1-np.sum(np.abs(Y_train-train_Y_prediction))/train_num
    test_num=X_test.shape[1]
    test_Y_prediction=predict(w,b,X_test)
    test_correction=1-np.sum(np.abs(Y_test-test_Y_prediction))/test_num
    
    print('train set correction rate is '+str(train_correction))
    print('test set correction rate is '+str(test_correction))
    
    d={
        'costs':costs,
        'train_Y_prediction':train_Y_prediction,
        'test_Y_prediction':test_Y_prediction,
        'w':w,
        'b':b,
        'learning_rate':learning_rate,
        'iteration_num':iteration_num
    }
    return d

if __name__=='__main__':
	d=model(train_set_x,train_set_y,test_set_x,test_set_y,learning_rate=0.005,print_cost=True)
	#画图
	costs=d['costs']
	plt.plot(costs)
	plt.show()