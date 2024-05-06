import tensorflow as tf
from tensorflow.keras import models,layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_img,train_lab),(test_img,test_lab)=mnist.load_data()
train_img=train_img.reshape((60000,28,28,1)).astype('float32')/255
test_img=test_img.reshape((10000,28,28,1)).astype('float32')/255
train_lab=to_categorical(train_lab)
test_lab=to_categorical(test_lab)

class neural:
    def __init__(self,num_classes,inp_shape):
        self.model=self.build(num_classes,inp_shape)
        
    def build(self,num_classes,inp_shape):   
        model=models.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(32,activation='relu'))
        model.add(layers.Dense(128,activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model
    
    def train(self,train_img,train_lab,epochs=5,batch_size=64,validation_split=0.1):
        self.model.fit(train_img,train_lab,epochs=epochs,batch_size=batch_size,validation_split=validation_split)
        
    def evaluate(self,test_img,test_lab):
        return self.model.evaluate(test_img,test_lab)
    
    def predict(self,img):
        return self.model.predict(img)
    
num_classes=10
inp_shape=(28,28,1)
nn=neural(num_classes,inp_shape)
nn.train(train_img,train_lab,epochs=5)

loss,acc=nn.evaluate(test_img,test_lab)
acc

predictions=[test_img[:5],nn.predict(test_img[:5])]
plt.figure(figsize=(25,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    img=predictions[0][i]
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted:{tf.argmax(predictions[1][i])}')
plt.show()
