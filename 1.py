import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize(inp_size,hid_size,out_size):
    np.random.seed(42)
    w1=np.random.randn(hid_size,inp_size)*0.01
    b1=np.zeros((hid_size,1))
    w2=np.random.randn(out_size,hid_size)*0.01
    b2=np.zeros((out_size,1))
    parameters={"w1":w1,"b1":b1,"w2":w2,"b2":b2}
    return parameters

def forward(parameters,x):
    w1,b1,w2,b2=parameters.values()
    z1=np.dot(w1,x)+b1
    a1=np.tanh(z1)
    z2=np.dot(w2,a1)+b2
    a2=sigmoid(z2)
    cache={"z1":z1,"a1":a1,"z2":z2,"a2":a2}
    return a2,cache

def cost_calc(a2,y):
    m=y.shape[1]
    cost=-(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    return cost

def backward(parameters,cache,x,y):
    m=x.shape[1]
    w1,w2=parameters['w1'],parameters['w2']
    z1,a1,z2,a2=cache.values()
    dz2=a2-y
    dw2=(1/m)*np.dot(dz2,a1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1=np.dot(w2.T,dz2)*(1-np.power(a1,2))
    dw1=(1/m)*np.dot(dz1,x.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)
    grads={"dw1":dw1,"db1":db1,"dw2":dw2,"db2":db2}
    return grads

def update(grads,parameters):
    w1,b1,w2,b2=parameters.values()
    dw1,db1,dw2,db2=grads.values()
    w1-=dw1
    b1-=db1
    w2-=dw2
    b2-=db2
    parameters={"w1":w1,"b1":b1,"w2":w2,"b2":b2}
    return parameters

def model(inp_size,hid_size,out_size,x,y,iterations=10000):
    parameters=initialize(inp_size,hid_size,out_size)
    for i in range(iterations+1):
        a2,cache=forward(parameters,x)
        cost=cost_calc(a2,y)
        grads=backward(parameters,cache,x,y)
        parameters=update(grads,parameters)
        if i%1000==0:
            print(f"cost:{cost}")
            
    return parameters

inp_size=2
hid_size=4
out_size=1
x=np.array([[0,0],[0,1],[1,0],[1,1]]).T
y=np.array([[0,1,1,0]])
parameters=model(inp_size,hid_size,out_size,x,y)
