import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


def read_data_array(csv_file_paths):
    image_path_collection = []
    image_class_collection = []

    for csv_file_path in csv_file_paths:
        annotations = pd.read_csv(csv_file_path)
        rows, cols = annotations.shape

        for i in range(rows):
            image_path = annotations.at[i, 'file']
            image_class = annotations.at[i, 'class']
            image_path_collection.append(image_path)
            class_array = np.zeros(43)
            class_array[int(image_class)] = 1
            image_class_collection.append(class_array)

    print("Input image size: " + str(len(image_path_collection)))
    print("Input label size: " + str(len(image_class_collection)))

    image_collection = []
    for i in range(len(image_path_collection)):
        image = Image.open(image_path_collection[i])
        image_array = np.array(image)
        image_collection.append(image_array)

    if len(image_collection) == len(image_class_collection):
        return np.array(image_collection), np.array(image_class_collection)
    else:
        raise ImportError


def continue_train():

    num_classes = 43
    input_shape = (48, 48, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, strides=1,
                               padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1,
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1,
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.0002, schedule_decay=1e-4),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # import data
    train_xs, train_ys = read_data_array(['./Train_processed.csv', './Train_processed_extend.csv'])
    test_xs, test_ys = read_data_array(['./Test_processed.csv'])
    train_xs = np.expand_dims(train_xs, axis=3)
    test_xs = np.expand_dims(test_xs, axis=3)

    # run model
    model.fit(train_xs, train_ys, batch_size=500, epochs=10, verbose=1)
    loss, acc = model.evaluate(test_xs, test_ys)
    print("\n\n==============================================================\n\n")
    print("Loss {}, Accuracy {}".format(loss, acc))
    print("\n\n==============================================================\n\n")


if __name__ == '__main__':
    continue_train()
