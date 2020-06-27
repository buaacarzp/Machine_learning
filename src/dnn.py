# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:52:47 2019
 
@author: peng.zhou
"""
from numpy import exp,array,random,dot
 
class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        layer2=5# neural counts
        layer3=4
        #init the w,b
        self.w1=2*random.random((3,layer2))-1
        self.w2=2*random.random((layer2,layer3))-1
        self.w3=2*random.random((layer3,1))-1
        print("start")
     
    def sigmoid(self,x):
        return (1/(1+exp(-x)))
    def sigmoid_dv(self,x):
        return(x*(1-x))
    # train set
    def train(self,inputs,y,epoch):
        #num=0
        for iteration in range(epoch):
           
           # num+=1
            #forward transform
            z1=dot(inputs,self.w1)#4,3x3,5=4,5
            a1=self.sigmoid(z1)#4,5
            z2=dot(a1,self.w2)#4,5x5,4=4,4
            a2=self.sigmoid(z2)#4,4
            z3=dot(a2,self.w3)#4,4x4,1=4,1
            a3=self.sigmoid(z3)#4,1
            #Back transform
            #d_z 和d_a矩阵形状是一样的,并且a取决于z的矩阵形状，且z和d_z的形状是一样的，否则在之后的权值更新时做不了加减法
            #所以当我们确定矩阵形状的时候：先在正向传播的时候确定z的形状，继而就确定了a的形状，之后反向传播的时候也就确定了dz的形状
            d_z3=(a3-y)*self.sigmoid_dv(a3)#4,1
            d_w3=dot(a2.T,d_z3)#4,4x4,1=4,1
            d_a2=dot(d_z3,(self.w3.T))#4,1x1,4=4,4
            d_z2=d_a2*(self.sigmoid_dv(a2))#4,4x4,4=4,4
            d_w2=dot((a1.T),d_z2)#5,4x4,4=5,4
            d_a1=dot(d_z2,self.w2.T)#dot((d_z2,self.w2.T)) #5,4x4,4=5,4### 4,4x4,5=4x5
            d_z1=(d_a1*(self.sigmoid_dv(a1)))#5,4x4,5=5,5###4,5x4,5 
            d_w1=dot((inputs.T),d_z1)#3,4x5,5#3,4X4,5=3,5
            #update the w
            self.w1 -=d_w1
            self.w2 -=d_w2
            self.w3 -=d_w3
            #print("num is\n"+str(num))
    def predict(self,test_input):
            z1=dot(test_input,self.w1)
            a1=self.sigmoid(z1)
            z2=dot(a1,self.w2)
            a2=self.sigmoid(z2)
            z3=dot(a2,self.w3)
            a3=self.sigmoid(z3)
            return (a3)
if __name__=="__main__":
    train_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    train_outputs=array([[0,1,1,0]]).T
    print('in:\n'+str(train_inputs))
    print('out:\n'+str(train_outputs))
    print("start train the module!\n")
    neural_network=NeuralNetwork()
    #neural_network.train(train_inputs,train_outputs,10000)
    print("the random weight of w1 is:\n"+str(neural_network.w1))
    print("the random weight of w2 is:\n"+str(neural_network.w2))
    print("the random weight of w3 is:\n"+str(neural_network.w3))
    neural_network.train(train_inputs,train_outputs,10000)
    #test the module
    print("the situation[1,0,0] is ?:\n"+str(neural_network.predict(array([1,1,1]))))