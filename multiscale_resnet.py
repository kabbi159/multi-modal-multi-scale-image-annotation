'''
Name: Jiwung Hyun
Date: 2019-02-25
'''

import tensorflow as tf
from network import _conv_fusion_block, _fully_connected, _batch_norm


def multiscale_resnet101(end_points, num_classes, is_training):
    fusion2 = _conv_fusion_block(end_points['resnet_v1_101/conv1'], 256, 'fusion2_', True)
    print(fusion2.shape)
    fusion2 = tf.add(end_points['resnet_v1_101/block1/unit_2/bottleneck_v1'], fusion2)
    print("after fusion2 (add)", fusion2.shape)

    fusion3 = _conv_fusion_block(fusion2, 512, 'fusion3_', is_training)
    fusion3 = tf.add(end_points['resnet_v1_101/block2/unit_3/bottleneck_v1'], fusion3)
    print("after fusion3 (add)", fusion3.shape)

    fusion4 = _conv_fusion_block(fusion3, 1024, 'fusion4_', is_training)
    fusion4 = tf.add(end_points['resnet_v1_101/block3/unit_22/bottleneck_v1'], fusion4)
    print("after fusion4 (add)", fusion4.shape)

    fusion5 = _conv_fusion_block(fusion4, 2048, 'fusion5_', is_training)
    fusion5 = tf.add(end_points['resnet_v1_101/block4/unit_2/bottleneck_v1'], fusion5)
    print("after fusion5 (add)", fusion5.shape)

    # fusion5 average pooling
    avg_pool = tf.nn.avg_pool(fusion5, ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding='SAME')
    print("avg_pool:", avg_pool.shape)
    flatten = tf.reshape(avg_pool, [-1, 2048])
    print("flatten:", flatten.shape)

    # fc, sigmoid
    fc1 = _fully_connected(flatten, 2048, 'fc1')
    output = _batch_norm(fc1, 'fc1_bn', is_training=is_training, activation_fn=tf.nn.sigmoid)
    print("ms_cnn:", output.shape)

    logit = _fully_connected(output, num_classes, 'fc2')
    logit = _batch_norm(logit, 'fc2_bn', is_training=is_training, activation_fn=None)

    return output, logit
