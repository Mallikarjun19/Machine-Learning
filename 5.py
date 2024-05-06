import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def create_convolutional_layer(filters, kernel_size, activation='relu', input_shape=None):
    if input_shape:
        return layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape)
    else:
        return layers.Conv2D(filters, kernel_size, activation=activation)

def create_maxpooling_layer(pool_size=(2, 2)):
    return layers.MaxPooling2D(pool_size)

def create_dense_layer(units, activation='relu'):
    return layers.Dense(units, activation=activation)

def build_convnet(input_shape, num_classes):
    model = models.Sequential()
    model.add(create_convolutional_layer(32, (3, 3), input_shape=input_shape))
    model.add(create_maxpooling_layer())
    model.add(create_convolutional_layer(64, (3, 3)))
    model.add(create_maxpooling_layer())
    model.add(layers.Flatten())
    model.add(create_dense_layer(128))
    model.add(create_dense_layer(num_classes, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

input_shape = (28, 28, 1)
num_classes = 10
model = build_convnet(input_shape, num_classes)
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
predictions2 = [test_images[:5],model.predict(test_images[:+5])]
from matplotlib import pyplot as plt

plt.figure(figsize=(25, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img_data = predictions2[0][i].reshape(28, 28)
    plt.imshow(img_data, cmap='gray')
    plt.title(f'Predicted: {tf.argmax(predictions2[1][i])}')
    plt.axis('off')
plt.show()
