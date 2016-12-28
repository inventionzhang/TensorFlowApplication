# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on traffic prediction.

"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
from fileDirectory import Dir_Structure as dir


# Parameters

root_Dir = dir.RootDir.getRootDir()
# file_path_train = root_Dir + "/data/link603_wednesday_pro_train.txt"
# file_path_train = root_Dir + "/data/603-1_15minuteAllTimeSpan.txt"
# file_path_train = root_Dir + "/data/603-1_15minuteAllTimeSpan_101_11101.txt"
file_path_train = root_Dir + "/data/603-1_15minuteAllTimeSpan_0_100.txt"
file_path_test = root_Dir + "/data/603-1_15minuteAllTimeSpan_0_100.txt"

learning_rate = 0.01
# training_epochs = 30000 # 训练次数
training_epochs = 300 # 训练次数
# training_epochs = 1000

batch_size = 20
display_step = 1
examples_to_show = 10

# Network Parameters
# n_hidden_1 = 256 # 1st layer num features
# n_hidden_2 = 128 # 2nd layer num features
# n_input = 784 # MNIST data input (img shape: 28*28)

n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_hidden_3 = 128 # 3nd layer num features
n_hidden_4 = 128 # 4nd layer num features
n_input = 28 # serial data input (img shape: 4*7)

bach_xs_train = []
bach_ys_train = []
bach_xs_test = []
bach_ys_test = []
input_next_row = 4
input_next_col = 7
column_num = 7 * 4 #输入数据列数
# output_col_index = 8 #目标输出列 所在的列

def read_txt_file_train(file_path_train):
    train_data = []
    file = open(file_path_train, "r")
    train_lines = file.readlines()  # 读取全部内容
    i = 0
    # 去掉第一行字段标题
    # 获得train data
    for line in train_lines:
        line = line.strip('\r\n')
        i += 1
        if i == 1:
            continue
        else:
            lineArray = line.split(",")
            subStr = lineArray[5:]
            train_data.append(subStr)
    file.closed
    return train_data


# obtain test data
def read_txt_file_test(file_path_test):
    test_data = []
    file = open(file_path_test, "r")
    test_lines = file.readlines()
    i = 0
    for line in test_lines:
        line = line.strip('\r\n')
        i += 1
        if i == 1:
            continue
        else:
            lineArray = line.split(",")
            subStr = lineArray[5:]
            test_data.append(subStr)
    file.closed
    return test_data

# 获得下一批次数据
# inputData:list输入数据，输入数据的下面n行数据作为同一个时间特征数据
# next_n：当前行的以下行数作为输入向量
# output_col：输出数据所在的列，列数从1算起
# return ndarray


def obtain_next_batch(inputData, next_n, output_col):
    row_num = len(inputData)
    last_num = row_num - next_n

    result_input_data = np.zeros((last_num, column_num), dtype=np.float64) #定义数组
    result_output_data = np.zeros((last_num, 1), dtype=np.float64)

    pro_data = []
    try:
        for i in range(0, last_num):
            t_data = inputData[i]

            for j in range(1, next_n + 1):
                tt_data = inputData[i + j]
                t_data.extend(tt_data)

            pro_data.append(t_data)
        input_data = pro_data
        # output_data = pro_data[:,2]
        temp_input_data = []
        temp_output_data = []
        for r in pro_data:
            temp_input_row = [] #输入行
            temp_output_row = [] #输出行
            t_column = 0
            for c in r:
                t_column +=1
                t = np.float64('%.6f' %np.float64(c)) #保留小数点后6位
                temp_input_row.append(t)
                if (t_column == output_col):
                    temp_output_row.append(t)
            temp_input_data.append(temp_input_row)
            temp_output_data.append(temp_output_row)
        result_input_data = np.array(temp_input_data)
        result_output_data = np.array(temp_output_data)

    except Exception,ex:
        traceback.print_exc()
        print (Exception, ":", ex)

    return result_input_data, result_output_data


# 数据归一化
def MaxMinNormalization(data):
    try:
        max_val = np.max(data)
        min_val = np.min(data)
        row_len = len(data)
        col_len = len(data[0])
        result_data = np.zeros((row_len, col_len), dtype=np.float64)  # 定义数组
        temp_input_data = []
        for r in range(row_len):
            temp_input_row = []
            for c in  range(col_len):
                x_val = data[r][c]
                temp = (x_val - min_val)/(max_val - min_val)
                temp = np.float64('%.6f' % np.float64(temp))  # 保留小数点后6位
                temp_input_row.append(temp)
            temp_input_data.append(temp_input_row)
        result_data = np.array(temp_input_data)

    except Exception, ex:
        traceback.print_exc()
        print (Exception, ":", ex)
    return result_data, max_val, min_val

# 最大最小反归一化
def antinormalization_max_min(data, max_val, min_val):
    try:
        row_len = len(data)
        col_len = len(data[0])
        result_data = np.zeros((row_len, col_len), dtype=np.float64)  # 定义数组
        temp_input_data = [] # 列表
        for r in range(row_len):
            temp_input_row = []
            for c in  range(col_len):
                x_val = data[r][c]
                temp = x_val * (max_val - min_val) + min_val
                temp = np.float64('%.6f' % np.float64(temp))  # 保留小数点后6位
                temp_input_row.append(temp)
            temp_input_data.append(temp_input_row)
        result_data = np.array(temp_input_data)

    except Exception, ex:
        traceback.print_exc()
        print (Exception, ":", ex)
    return  result_data

train_data = read_txt_file_train(file_path_train)

test_data = read_txt_file_test(file_path_test)

train_input_data, train_output_data = obtain_next_batch(train_data, 3, 3)

test_input_data, test_output_data = obtain_next_batch(test_data, 3, 3)

train_normalize_input, max_val, min_val = MaxMinNormalization(train_input_data)

train_normalize_output, _, _ = MaxMinNormalization(train_output_data)

test_normalize_input,max_val_test, min_val_test = MaxMinNormalization(test_input_data)

test_normalize_output, _, _ = MaxMinNormalization(test_output_data)

#
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))

    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   biases['encoder_b4']))
    return layer_4


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))

    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

output_col_index = 2 #目标输出列 所在的列

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # total_batch = int(len(train_data[0])/batch_size)
    total_batch = 1
    row_count = len(train_data) # row number

    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # batch_xs = train_input_data

            batch_xs = train_normalize_input
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode_train = sess.run(
        y_pred, feed_dict={X: train_normalize_input})

    inpu_count = len(train_normalize_input)
    output_normal_data_pre = np.zeros((inpu_count, 1), dtype=np.float64)
    temp_output_pre = []
    for i in range(inpu_count):
        temp_output_row = []  # 输出行
        output_val = encode_decode_train[i][output_col_index]
        temp_output_row.append(output_val)
        temp_output_pre.append(temp_output_row)

    output_normal_data_pre = np.array(temp_output_pre)

    figure = plt.figure()
    # plt.plot(train_normalize_output, output_normal_data_pre) #折线
    # plt.plot(train_normalize_output, 'b')
    # plt.plot(output_normal_data_pre, 'r')

    # 测试数据预测
    encode_decode_test = sess.run(y_pred, feed_dict={X: test_normalize_input})
    test_input_count = len(test_normalize_input)
    test_output_normal_data_pre = np.zeros((test_input_count, 1), dtype=np.float64)
    temp_output_pre = []
    for i in range(test_input_count):
        temp_output_row = []  # 输出行
        output_val = encode_decode_test[i][output_col_index]
        temp_output_row.append(output_val)
        temp_output_pre.append(temp_output_row)
    test_output_normal_data_pre = np.array(temp_output_pre)

    # plt.plot(test_normalize_output, 'b*')
    # plt.plot(test_normalize_output, 'b')
    # plt.plot(test_output_normal_data_pre, 'r*')
    # plt.plot(test_output_normal_data_pre, 'r')

    # 反归一化
    test_antinormalize_output = antinormalization_max_min(test_normalize_output,max_val_test, min_val_test)
    test_antinormalize_output_pre = antinormalization_max_min(test_output_normal_data_pre, max_val_test, min_val_test)

    # RMSE: root mean squre
    error_array = test_antinormalize_output_pre - test_antinormalize_output
    # error = error_array.flatten()
    n = len(error_array)
    error_square_sum = (error_array**2).sum()
    RMSE = np.sqrt(error_square_sum/n)

    # MAPE: mean absolute percentage error
    # 去除为零的项
    row_count = len(test_antinormalize_output)
    col_count = len(test_antinormalize_output[0])
    test_antinormalize_output_process = np.sum((row_count, col_count), dtype = np.float64)
    error_percent_array_sum = 0
    for r in range(row_count):
        for c in range(col_count):
            val = test_antinormalize_output[r][c]
            if abs(val) != 0:
                error_percent = abs(error_array[r][c]/val)
                error_percent_array_sum += error_percent

    MAPE = error_percent_array_sum/n
    print ("RMSE:", '%.6f' %RMSE)
    print ("MAPE:", '%.6f' %MAPE)

    # ax = figure.add_axes([0.1, 0.1, 0.6, 0.75])
    plt.plot(test_antinormalize_output, 'b*')
    l1 = plt.plot(test_antinormalize_output, 'b')
    plt.plot(test_antinormalize_output_pre, 'r*')
    l2 = plt.plot(test_antinormalize_output_pre, 'r')
    plt.title('Travel Time Prediction Using Autoencoder')
    # figure.legend((l1, l2), ('TrueValue','PredictValue'), 'upper right')
    plt.legend((l1, l2), ('TrueValue', 'PredictValue'))
    # 保存为PDF格式，也可保存为PNG等图形格式
    # plt.savefig('D:\\test.pdf')



    # plt.plot(true_val, pre_val)

    figure.show()
    plt.draw()
    plt.waitforbuttonpress()
    print ("done!")