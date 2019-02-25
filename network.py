'''
Name: Jiwung Hyun
Date: 2018-02-25

Currently ms_cnn function is not used.
'''

import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import batch_norm


def _conv2d(input_layer, kernel_size, num_o, stride, name, biased=False):
    num_x = input_layer.shape[3].value
    # print(num_x)
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
        s = [1, stride, stride, 1]
        o = tf.nn.conv2d(input_layer, W, s, padding='SAME')
        if biased:
            b = tf.get_variable('biases', shape=[num_o])
            o = tf.nn.bias_add(o, b)
        return o

def _fully_connected(x, size, name):
    input_size = int(x.get_shape()[1])
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('weights', shape=[input_size, size])
        b = tf.get_variable('biases', shape=[size])
        o = tf.matmul(x, W) + b
        return o


def _batch_norm(x, name, is_training, activation_fn, trainable=False):
    with tf.variable_scope(name) as scope:
        o = batch_norm(x, decay=0.9997, scale=True, activation_fn=activation_fn, is_training=is_training,
                       trainable=trainable, scope=scope)
        return o

def _conv_block(x, num_o, name, is_training):
    o_a = _conv2d(x, 1, num_o/4, 1, name=name+'a')
    o_a = _batch_norm(o_a, name=name+'a', is_training=is_training, activation_fn=tf.nn.relu)
    o_b = _conv2d(o_a, 3, num_o/4, 1, name=name+'b')
    o_b = _batch_norm(o_b, name=name + 'b', is_training=is_training, activation_fn=tf.nn.relu)
    o_c = _conv2d(o_b, 1, num_o, 1, name=name+'c')
    o_c= _batch_norm(o_c, name=name + 'c', is_training=is_training, activation_fn=None)
    return o_c


def _conv_fusion_block(x, num_o, name, is_training, two=False):
    if two:
        o_a = _conv2d(x, 3, num_o/4, 2, name=name+'a')
        o_a = _batch_norm(o_a, name=name + 'a', is_training=is_training, activation_fn=tf.nn.relu)
    else:
        o_a = _conv2d(x, 3, num_o/2, 2, name=name+'a')
        o_a = _batch_norm(o_a, name=name + 'a', is_training=is_training, activation_fn=tf.nn.relu)
    o_b = _conv2d(o_a, 1, num_o, 1, name=name+'b')
    o_b = _batch_norm(o_b, name=name + 'b', is_training=is_training, activation_fn=None)
    return o_b

def ms_cnn(x, is_training):

    # conv1
    conv1 = _conv2d(x, 7, 64, 2, 'conv1', is_training)
    print("after conv1:", conv1.shape)

    # conv2
    outputs = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    for i in range(3):
        outputs = _conv_block(outputs, 256, 'conv2_'+str(i+1), is_training)
    print("after conv2:", outputs.shape)

    # fusion2
    fusion2 = _conv_fusion_block(conv1, 256, 'fusion2_', is_training, True)

    # conv2 + fusion2
    fusion2 = tf.add(outputs, fusion2)
    print("after fusion2 (add)", fusion2.shape)

    # conv3
    outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    for i in range(4):
        outputs = _conv_block(outputs, 512, 'conv3_'+str(i+1), is_training)
    print("after conv3:", outputs.shape)

    # fusion3
    fusion3 = _conv_fusion_block(fusion2, 512, 'fusion3_', is_training)

    # conv3 + fusion3
    fusion3 = tf.add(outputs, fusion3)
    print("after fusion3 (add)", fusion3.shape)

    # conv4
    outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    for i in range(23):
        outputs = _conv_block(outputs, 1024, 'conv4_'+str(i+1), is_training)
    print("after conv4:", outputs.shape)

    # fusion4
    fusion4 = _conv_fusion_block(fusion3, 1024, 'fusion4_', is_training)

    # conv4 + fusion4
    fusion4 = tf.add(outputs, fusion4)
    print("after fusion4 (add)", fusion4.shape)

    # conv5
    outputs = tf.nn.max_pool(outputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    for i in range(3):
        outputs = _conv_block(outputs, 2048, 'conv5_'+str(i+1), is_training)
    print("after conv5:", outputs.shape)

    # fusion5
    fusion5 = _conv_fusion_block(fusion4, 2048, 'fusion5_', is_training)

    # conv5 + fusion5
    fusion5 = tf.add(outputs, fusion5)
    print("after fusion5 (add)", fusion5.shape)

    # fusion5 average pooling
    avg_pool = tf.nn.avg_pool(fusion5, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
    print("avg_pool:", avg_pool.shape)
    flatten = tf.reshape(avg_pool, [-1, 2048])
    print("flatten:", flatten.shape)

    # fc, sigmoid
    fc1 = _fully_connected(flatten, 2048, 'fc1')
    output = _batch_norm(fc1, 'fc1_bn', is_training=is_training, activation_fn=None)
    print("ms_cnn:", output.shape)

    return output


def mlp(x, num_classes, is_training):
    with tf.name_scope('MLPwithBN'):
        layer_1 = _fully_connected(x, 2048, 'layer_1')
        print("layer_1:",layer_1.shape)
        layer_1 = _batch_norm(layer_1, 'layer_1_bn', is_training=is_training, activation_fn=tf.nn.relu)
        layer_2 = _fully_connected(layer_1, 2048, 'layer_2')
        print("layer_2:", layer_2.shape)
        layer_2 = _batch_norm(layer_2, 'layer_2_bn', is_training=is_training, activation_fn=tf.nn.relu)

        fc1 = _fully_connected(layer_2, num_classes, 'layer_logit')
        logit = _batch_norm(fc1, 'layer_logit_bn', is_training=is_training, activation_fn=None)
        return layer_2, logit

def multi_class_classification_model(x, num_classes):
    with tf.name_scope('MLP_MCC'):
        fc1 = _fully_connected(x, num_classes, 'MCC_fc1')
        print("MCC_fc1:", fc1.shape)

        return fc1

def label_quantity_prediction_model(x, keep_prob):
    with tf.name_scope('MLP_LQP'):
        fc1 = _fully_connected(x, 512, 'LQP_fc1')
        fc1 = tf.nn.dropout(fc1, keep_prob)
        fc2 = _fully_connected(fc1, 256, 'LQP_fc2')
        fc2 = tf.nn.dropout(fc2, keep_prob)
        fc3 = _fully_connected(fc2, 1, 'LQP_fc3')
        output = 6 * tf.nn.sigmoid(fc3) + 1

        return output
