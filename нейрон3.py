import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = tensorflow.keras.datasets.mnist.load_data()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis = 3)
test_images = np.expand_dims(test_images, axis = 3)

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
                   Conv2D(num_filters, filter_size, input_shape=(28,28, 1)),
                   MaxPooling2D(pool_size = pool_size),
                   Flatten(),
                   Dense(10, activation = 'softmax'),
                   ])
model.compile('adam', loss = 'categorical_crossentropy', metrics=['accuracy'],)
model.fit(train_images, to_categorical(train_labels), epochs = 3, validation_data = (test_images, to_categorical(test_labels)))
predictions = model.predict(test_images[:5])

print(np.argmax(predictions, axis = 1)) #7, 2, 1, 0, 4

print(test_labels[:5])