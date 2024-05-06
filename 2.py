import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class neural:
    def __init__(self,inp,hid,out):
        self.inp=inp
        self.hid=hid
        self.out=out
        
        self.w1=np.random.randn(hid,inp)*0.01
        self.b1=np.zeros((hid,1))
        self.w2=np.random.randn(out,hid)*0.01
        self.b2=np.zeros((out,1))
        
    def forward(self,x):
        self.z1=np.dot(self.w1,x)+self.b1
        self.a1=np.tanh(self.z1)
        self.z2=np.dot(self.w2,self.a1)+self.b2
        self.a2=self.sigmoid(self.z2)
        
        return self.a2
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def cost(self,a,y):
        m=y.shape[1]
        return -(1/m)*np.sum(y*np.log(a)+(1-y)*np.log(1-a))
        
    def backward(self,x,y,lr=0.01):
        m=x.shape[1]
        
        dz2=self.a2-y
        dw2=(1/m)*np.dot(dz2,self.a1.T)
        db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
        dz1=np.dot(self.w2.T,dz2)*(1-np.power(self.a1,2))
        dw1=(1/m)*np.dot(dz1,x.T)
        db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)
        
        self.w1-=dw1
        self.b1-=db1
        self.w2-=dw2
        self.b2-=db2
        
    def train(self,x,y,iter=10000,lr=0.01):
        for i in range(iter+1):
            a2=self.forward(x)
            loss=self.cost(a2,y)
            self.backward(x,y)
            if i%1000==0:
                print(loss)

                
x,y=make_moons(n_samples=1000,noise=0.2,random_state=42)
x=x.T
y=y.reshape(1,-1)
plt.scatter(x[0,:],x[1,:],c=y.ravel())
plt.show()
inp=2
hid=4
out=1
poi=neural(inp,hid,out)
poi.train(x,y,iter=10000,lr=0.01)
                
