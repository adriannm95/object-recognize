# -*- coding: utf-8 -*-

# Sample code to use string producer.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h

num_classes = 3
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10 #retrasa el inicio del entreno ya que tensor flow debe procesarlos antes del entreno.
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch) 
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y

#estuve por aqui, falta la ruta de las imagenes y cambie el casteo tf.cast(train_batch_label, dtype=tf.float32)))

train_batch, train_batch_label = dataSource(["Signos/Dataset/0/*.JPG", "Signos/Dataset/1/*.JPG","Signos/Dataset/2/*.JPG"], batch_size=batch_size)
valid_batch, valid_batch_label = dataSource(["Signos/Dataset/0_Validacion/*.JPG", "Signos/Dataset/1_Validacion/*.JPG","Signos/Dataset/2_Validacion/*.JPG"], batch_size=batch_size)
test_batch, test_batch_label = dataSource(["Signos/Dataset/0_Test/*.JPG", "Signos/Dataset/1_Test/*.JPG","Signos/Dataset/2_Test/*.JPG"], batch_size=batch_size)

train_batch_predicted = myModel(train_batch, reuse=False)
valid_batch_predicted = myModel(valid_batch, reuse=True)
test_batch_predicted = myModel(test_batch, reuse=True)

cost = tf.reduce_sum(tf.square(train_batch_predicted - tf.cast(train_batch_label, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(valid_batch_predicted - tf.cast(valid_batch_label, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(test_batch_predicted - tf.cast(test_batch_label, dtype=tf.float32)))

# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()
train_Gra = []
train_Valid = []
errors =0

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(430):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            #print(sess.run(valid_batch_label))
            #print(sess.run(valid_batch_predicted))

            print("Error:", sess.run(cost_valid))

            train_Gra.append(sess.run(cost)) # metemos el coste
            train_Valid.append(sess.run(cost_valid)) # metemos el coste de validacion




    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    # join de lo hilos.
    coord.request_stop()
    coord.join(threads)

    # imprimimos los resultados guardados en el array
    plt.title("Entrenamiento")
    plt.plot(train_Gra)
    plt.show()

    #imprimimos los resultados guardados en el array
    plt.title("Validacion")
    plt.plot(train_Valid)
    plt.show()
