import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import init_model


def read_data(csv_file_paths):
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
        image = cv2.imread(image_path_collection[i])
        image_array = np.array(image[:, :, 0])
        image_array_flatten = image_array.flatten()
        image_collection.append(image_array_flatten)

    if len(image_collection) == len(image_class_collection):
        return np.array(image_collection), np.array(image_class_collection)
    else:
        raise ImportError


def continue_train():
    tf.set_random_seed(time.time())
    np.random.seed(int(time.time()))

    BATCH_SIZE = 500

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 4
    config.inter_op_parallelism_threads = 4

    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))
    graph = tf.get_default_graph()

    train_op = graph.get_operation_by_name("Adam")
    loss = graph.get_tensor_by_name("softmax_cross_entropy_loss/value:0")
    tf_x = graph.get_tensor_by_name("truediv:0")
    tf_y = graph.get_tensor_by_name("Placeholder_1:0")
    flat = graph.get_tensor_by_name("Reshape_1:0")
    accuracy = graph.get_tensor_by_name("Mean:0")

    # xs_i, ys_i = read_data(['./Train_processed.csv', './Train_processed_extend.csv'])
    xs_i, ys_i = read_data(['./Train_processed.csv'])
    # xs_i, ys_i = read_data(['./Train_processed_extend.csv'])
    test_xs, test_ys = read_data(['./Test_processed.csv'])

    for step in range(100001):
        batch_xs, batch_ys = init_model.next_batch(BATCH_SIZE, xs_i, ys_i)
        _, loss_ = sess.run([train_op, loss], {tf_x: batch_xs, tf_y: batch_ys})
        if step % 100 == 0:
            accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_xs, tf_y: test_ys})
            print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_)
            saver.save(sess, './checkpoint_dir/MyModel')
        time.sleep(1)


if __name__ == '__main__':
    continue_train()