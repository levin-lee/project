import numpy as np
import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1.contrib.layers import xavier_initializer as init
# from tensorflow.compat.v1.contrib.layers import flatten

# import tensorflow._api.v2.compat.v1 as tf
# from tensorflow._api.v2.compat.v1.contrib.layers import xavier_initializer as init
# from tensorflow._api.v2.compat.v1.contrib.layers import flatten
init = tf.keras.initializers.glorot_uniform()
# Flatten function
flatten = tf.keras.layers.Flatten()


def generator(inputs):
    '''
    generator (2D)

    inputs: [N, 64, 64, 1]
    '''
    skip_1 = inputs
    output_1 = tf.layers.conv2d(inputs, 32, (3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='conv1')
    output_1 = tf.nn.relu(output_1)
    output_1 = tf.layers.conv2d(output_1, 32, (3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='conv2')
    output_1 = tf.nn.relu(output_1)
    skip_2 = output_1
    output_2 = tf.layers.conv2d(output_1, 32, (1, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='conv3')
    output_2 = tf.nn.relu(output_2)
    output_2 = tf.layers.conv2d(output_2, 32, (1, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='conv4')
    output_2 = tf.nn.relu(output_2)
    skip_3 = output_2
    output_3 = tf.layers.conv2d(output_2, 32, (1, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='conv5')
    output_3 = tf.nn.relu(output_3)
    output_3 = tf.layers.conv2d_transpose(output_3, 32, (1, 3), 1, 'same', use_bias=False,
                                          kernel_initializer=init(), name='deconv1')
    output_3 += skip_3
    output_3 = tf.nn.relu(output_3)
    output_4 = tf.layers.conv2d_transpose(output_3, 32, (1, 3), 1, 'same', use_bias=False,
                                          kernel_initializer=init(), name='deconv2')
    output_4 = tf.nn.relu(output_4)
    output_4 = tf.layers.conv2d_transpose(output_4, 32, (1, 3), 1, 'same', use_bias=False,
                                          kernel_initializer=init(), name='deconv3')
    output_4 += skip_2
    output_4 = tf.nn.relu(output_4)
    output_5 = tf.layers.conv2d_transpose(output_4, 32, (3, 3), 1, 'same', use_bias=False,
                                          kernel_initializer=init(), name='deconv4')
    output_5 = tf.nn.relu(output_5)
    output_5 = tf.layers.conv2d_transpose(output_5, 1, (3, 3), 1, 'same', use_bias=False,
                                          kernel_initializer=init(), name='deconv5')
    output_5 += skip_1
    output_5 = tf.nn.relu(output_5)
    return output_5


def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)


def discriminator(inputs):
    '''
    discriminator (2D)

    inputs: [N, 64, 64, 1]
    '''
    outputs = tf.layers.conv2d(inputs, 64, 3, strides=(2, 2), padding='valid', kernel_initializer=init(),
                               name='conv1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 128, 3, strides=(2, 2), padding='valid', kernel_initializer=init(),
                               name='conv2')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 256, 3, strides=(2, 2), padding='valid', kernel_initializer=init(),
                               name='conv3')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = tf.layers.conv2d(outputs, 512, 3, strides=(2, 2), padding='valid', kernel_initializer=init(),
                               name='conv4')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = flatten(outputs)
    outputs = tf.layers.dense(outputs, units=1024, name='dense1')
    outputs = leaky_relu(outputs, alpha=0.2)
    outputs = tf.layers.dense(outputs, units=1, name='dense2')
    return outputs
