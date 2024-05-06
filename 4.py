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


model=models.Sequential()
model.add(layers.Conv2D(32,(2,2),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(2,2),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_img,train_lab,epochs=5,batch_size=64,validation_split=0.1)

loss,acc=model.evaluate(test_img,test_lab)
predictions=[test_img[:5],model.predict(test_img[:5])]

plt.figure(figsize=(25,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    img=predictions[0][i]
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted:{tf.argmax(predictions[1][i])}')
plt.show()
