import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time


def read_data(csv_file_path):
    # open .csv file
    annotations = pd.read_csv(csv_file_path)
    rows, cols = annotations.shape

    image_path_collection = []
    image_class_collection = []

    # read data in row
    for i in range(rows):
        image_path = annotations.at[i, 'file']
        image_class = annotations.at[i, 'class']
        image_path_collection.append(image_path)
        class_array = np.zeros(43)
        class_array[int(image_class)] = 1
        image_class_collection.append(class_array)

    print("Input image size: " + str(len(image_path_collection)))
    print("Input label size: " + str(len(image_class_collection)))

    # load image in memory
    image_collection = []
    for i in range(len(image_path_collection)):
        image = cv2.imread(image_path_collection[i])
        image_array = np.array(image[:, :, 0])
        image_array_flatten = image_array.flatten()
        image_collection.append(image_array_flatten)

    if len(image_collection) == len(image_class_collection):
        return np.array(image_collection), np.array(image_class_collection)
    else:
        raise ImportError


def next_batch(num, data, labels):
    # random shuffle data and labels
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def init_train():
    tf.set_random_seed(time.time())
    np.random.seed(int(time.time()))

    BATCH_SIZE = 500
    LR = 0.001

    tf_x = tf.placeholder(tf.float32, [None, 48 * 48]) / 255.
    image = tf.reshape(tf_x, [-1, 48, 48, 1])
    tf_y = tf.placeholder(tf.int32, [None, 43])

    # CNN
    conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, strides=1,
                             padding='same', activation=tf.nn.relu)
    # -> (48*48*16)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)
    pool1_drop = tf.nn.dropout(pool1, rate=0.1)
    # -> (24*24*16)
    conv2 = tf.layers.conv2d(inputs=pool1_drop, filters=32, kernel_size=5, strides=1,
                             padding='same', activation=tf.nn.relu)
    # -> (24*24*32)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
    pool2_drop = tf.nn.dropout(pool2, rate=0.2)
    # -> (12*12*32)
    conv3 = tf.layers.conv2d(pool2_drop, 64, 5, 1, 'same', activation=tf.nn.relu)
    # -> (12*12*64)
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
    pool3_drop = tf.nn.dropout(pool3, rate=0.3)

    # -> (6*6*64)
    flat = tf.reshape(pool3_drop, [-1, 6 * 6 * 64])
    output = tf.layers.dense(flat, 43)

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial session
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4

    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    xs_i, ys_i = read_data('./Train_processed.csv')
    test_xs, test_ys = read_data('./Test_processed.csv')

    for step in range(2001):
        batch_xs, batch_ys = next_batch(BATCH_SIZE, xs_i, ys_i)
        _, loss_ = sess.run([train_op, loss], {tf_x: batch_xs, tf_y: batch_ys})
        if step % 50 == 0:
            accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_xs, tf_y: test_ys})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_)
            saver.save(sess, './checkpoint_dir/MyModel')
        time.sleep(1)


if __name__ == '__main__':
    init_train()