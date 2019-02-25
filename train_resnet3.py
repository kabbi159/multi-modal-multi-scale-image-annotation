import tensorflow as tf
import numpy as np
from datetime import datetime
from network import mlp, multi_class_classification_model, label_quantity_prediction_model
from nuswide_datagenerator import ImageDataGenerator
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.framework import get_variables_to_restore, assign_from_checkpoint_fn
from tensorflow.contrib import slim
from multiscale_resnet import multiscale_resnet101
import os

# To use on multi GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

checkpoint_path = '/home/jiwunghyun/PycharmProjects/multi-modal-multi-scale-image-annotation/checkpoints'

# Learning parameters
learning_rate0 = 0.0005
learning_rate1 = 0.0005
learning_rate2 = 0.05
num_epochs = 30
batch_size = 32

# Network parameters
dropout_rate = 0.5
num_classes = 81
num_noisy_tags = 925 # 1000 - 75 (duplicated)

train_layers0 = ['res']  # resnet101 finetuning
train_layers1_1 = ['fus', 'fc1', 'fc2']  # multi-scale layer train
train_layers1_2 = ['lay']  # textual model train
train_layers2_1 = ['MCC']  # multi-class classification model train
train_layers2_2 = ['LQP']  # label quantity prediction model train

# Dataset loader + initializer
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(mode='training', batch_size=batch_size, num_classes=num_classes, shuffle=True)
    test_data = ImageDataGenerator(mode='inference', batch_size=batch_size, num_classes=num_classes, shuffle=False)

    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()

training_init_op = iterator.make_initializer(tr_data.data)
test_init_op = iterator.make_initializer(test_data.data)

# # make tag_list to show result (what is the label)
# fp = open("/media/jiwunghyun/DATA/nus-wide/Concepts81.txt", 'r')
# tag_list = []
# for i in range(81):
#     a = fp.readline().split('\n')[0]
#     tag_list.append(a)
# fp.close()


# placeholder for input and output
img = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3])  # image
tag = tf.placeholder(tf.float32, shape=[batch_size, num_noisy_tags])  # noisy tags ex.) [1 0 0 1 0]
y = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
q = tf.reduce_sum(y, 1)  # quantity
keep_prob = tf.placeholder(tf.float32)

# model
# resnet_v1 101
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_101(img, num_classes, is_training=False)
net_logit = tf.squeeze(net)

# tensorflow operation for load pretrained weights
variables_to_restore = get_variables_to_restore(exclude=['resnet_v1_101/logits', 'resnet_v1_101/AuxLogits'])
init_fn = assign_from_checkpoint_fn('resnet_v1_101.ckpt', variables_to_restore)

# multiscale resnet_v1 101
visual_features, fusion_logit = multiscale_resnet101(end_points, num_classes)
textual_features, textual_logit = mlp(tag, num_classes)
refined_features = tf.concat([visual_features, textual_features], 1)

# score is prediction score, and k is label quantity
score = multi_class_classification_model(refined_features, num_classes)
k = label_quantity_prediction_model(refined_features, keep_prob)
k = tf.reshape(k, shape=[batch_size])


# make trainable variable list
var_list0 = [v for v in tf.trainable_variables() if v.name.split('/')[0][0:3] in train_layers0]
var_list1_1 = [v for v in tf.trainable_variables() if v.name.split('/')[0][0:3] in train_layers1_1]
var_list1_2 = [v for v in tf.trainable_variables() if v.name.split('/')[0][0:3] in train_layers1_2]
var_list2_1 = [v for v in tf.trainable_variables() if v.name.split('/')[0][0:3] in train_layers2_1]
var_list2_2 = [v for v in tf.trainable_variables() if v.name.split('/')[0][0:3] in train_layers2_2]

# sigmoid_cross_entropy_loss : resnet, fusion, mlp, prediction fc + mean squared error loss : label quantity fc
with tf.name_scope('loss'):
    resnet_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_logit, labels=y), 1))
    fusion_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=fusion_logit, labels=y), 1))
    mlp_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=textual_logit, labels=y), 1))
    cross_entropy_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y), 1))
    mean_squared_error_loss = tf.reduce_mean(tf.square(q-k))


# train operation
with tf.name_scope("train"):
    global_step0 = tf.Variable(0, trainable=False)
    global_step1_1 = tf.Variable(0, trainable=False)
    global_step1_2 = tf.Variable(0, trainable=False)
    global_step2_1 = tf.Variable(0, trainable=False)
    global_step2_2 = tf.Variable(0, trainable=False)

    decayed_learning_rate0 = tf.train.exponential_decay(learning_rate0, global_step0, 80000, 0.1)
    decayed_learning_rate1_1 = tf.train.exponential_decay(learning_rate1, global_step1_1, 80000, 0.1)
    decayed_learning_rate1_2 = tf.train.exponential_decay(learning_rate2, global_step1_2, 80000, 0.1)
    decayed_learning_rate2_1 = tf.train.exponential_decay(learning_rate1, global_step2_1, 80000, 0.1)
    decayed_learning_rate2_2 = tf.train.exponential_decay(learning_rate2, global_step2_2, 80000, 0.1)

    gradients0 = list(zip(tf.gradients(resnet_loss, var_list0), var_list0))
    gradients1_1 = list(zip(tf.gradients(fusion_loss, var_list1_1), var_list1_1))
    gradients1_2 = list(zip(tf.gradients(mlp_loss, var_list1_2), var_list1_2))
    gradients2_1 = list(zip(tf.gradients(cross_entropy_loss, var_list2_1), var_list2_1))
    gradients2_2 = list(zip(tf.gradients(mean_squared_error_loss, var_list2_2), var_list2_2))

    train_op0 = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate0, momentum=0.9).apply_gradients(
        gradients0)
    train_op1_1 = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate1_1, momentum=0.9).apply_gradients(
        gradients1_1)
    train_op1_2 = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate1_2, momentum=0.9).apply_gradients(
        gradients1_2)
    train_op2_1 = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate2_1, momentum=0.9).apply_gradients(
        gradients2_1)
    train_op2_2 = tf.train.MomentumOptimizer(learning_rate=decayed_learning_rate2_2, momentum=0.9).apply_gradients(
        gradients2_2)

# Evaluation op: Accuracy of the model
with tf.name_scope("evaluation"):

    k = tf.cast(tf.floor(k), tf.int32)
    y_int32 = tf.cast(y, tf.int32)  # [0 1 1 0 0]

    for i in range(k.shape[0]):
        tmp = tf.nn.top_k(score[i], k[i])
        one_hot = tf.one_hot(tmp.indices, score.shape[1])
        topk_tmp = tf.reduce_sum(one_hot, 0)
        if i == 0:
            topk = topk_tmp
        else:
            topk = tf.concat([topk, topk_tmp], 0)

    topk = tf.reshape(topk, shape=score.shape)
    topk = tf.cast(topk, tf.int32)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
test_batches_per_epoch = int(np.floor(test_data.data_size / batch_size))

# Start session
with tf.Session() as sess:

    # Initialize all variables

    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    print("{} Start training...".format(datetime.now()))

    # Fintuning resnet 101
    for epoch in range(5):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):
            img_batch, label_batch, tag_batch = sess.run(next_batch)
            _tr0, _resnet_loss = sess.run([train_op0, resnet_loss],
                                         feed_dict={img: img_batch, tag: tag_batch, y: label_batch,
                                                    keep_prob: dropout_rate})

            if (step % 200) == 0:
                print('Step {} training loss (resnet loss):'.format(step), _resnet_loss)

        # print("{} Saving checkpoint of model...".format(datetime.now()))
        # # save checkpoint of the model
        # checkpoint_name = os.path.join(checkpoint_path,
        #                                'model_epoch' + str(epoch + 1) + '.ckpt')
        #
        # save_path = saver.save(sess, checkpoint_name)
        #
        # print("{} Model checkpoint saved at {}".format(datetime.now(),
        #                                                checkpoint_name))



    # fusion layer + textual feature MLP training
    for epoch in range(15):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):
            img_batch, label_batch, tag_batch = sess.run(next_batch)

            _tr1_1, _tr1_2, _fusion_loss, _mlp_loss = sess.run([train_op1_1, train_op1_2, fusion_loss, mlp_loss], feed_dict={img: img_batch, tag: tag_batch, y: label_batch, keep_prob: dropout_rate})

            if(step % 200) == 0:
                print('Step {} training loss (fusion loss):'.format(step), _fusion_loss, '(textual loss):', _mlp_loss)

    # FC layer for multi-class classification and label quantity training
    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch, tag_batch = sess.run(next_batch)

            _tr2_1, _tr2_2, tr_ce_loss, tr_mse_loss = sess.run([train_op2_1, train_op2_2, cross_entropy_loss, mean_squared_error_loss],
                                                           feed_dict={img: img_batch, tag: tag_batch, y: label_batch, keep_prob: dropout_rate})

            if (step % 200) == 0:
                print('Step {} training loss (sigmoid-cross-entropy):'.format(step), tr_ce_loss, '(mean-squared-error):', tr_mse_loss)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        sess.run(test_init_op)

        num_correct, num_gt, num_pr = [0 for i in range(num_classes)], [0 for i in range(num_classes)], [0 for i in range(num_classes)]

        for _ in range(test_batches_per_epoch):
            # test_index = random.randrange(2, 15)

            img_batch, label_batch, tag_batch = sess.run(next_batch)
            _topk = sess.run(topk, feed_dict={img: img_batch, tag: tag_batch, y: label_batch, keep_prob: 1.})

            num_gt += np.sum(label_batch, 0)
            num_pr += np.sum(_topk, 0)

            for i in range(batch_size):
                for j in range(num_classes):
                    if _topk[i][j] * label_batch[i][j]:
                        num_correct[j] += 1

        per_class_precision = np.mean(num_correct / num_pr)
        per_class_recall = np.mean(num_correct / num_gt)
        overall_precision = np.sum(num_correct) / np.sum(num_pr)
        overall_recall = np.sum(num_correct) / np.sum(num_gt)

        print("{} Test per class Precision = {:.4f}".format(datetime.now(), per_class_precision))
        print("{} Test per class Recall = {:.4f}".format(datetime.now(), per_class_recall))
        print("{} Test overall Precision = {:.4f}".format(datetime.now(), overall_precision))
        print("{} Test overall Recall = {:.4f}".format(datetime.now(), overall_recall))

        # save checkpoint of the model
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch' + str(epoch + 1) + '.ckpt')

        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))

