# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on traffic prediction.

"""

# 2016年12月10日

# 路网交通预测
# 1. 获得路网中路段Id
# 2. 每个路段分别encode, decode,训练
# 3. 保存模型, 模型测试


from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
from fileDirectory import Dir_Structure as dir
from fileOperation import ReadWriteFile as rwf
from dataProcess import Normalization as norml
from dataProcess import Features as feature
from util import SetOperation as setOper
import sys
from pylab import *
import copy


reload(sys)
sys.setdefaultencoding('utf8')
mpl.rcParams['font.sans-serif'] = ['SimHei']


# model prediction 模型预测
# Applying encode and decode over test set
# sess：
# y_pred：模型预测值
# test_input_data：输入
# test_output_data：输出

def model_predict(sess, y_pred, test_input_data, test_output_data):
    t_normalize = norml.Normalization(test_input_data)
    test_normalize_input, max_val_test, min_val_test = t_normalize.max_minNormalization(test_input_data)

    t_normalize = norml.Normalization(test_output_data)
    test_normalize_output, _, _ = t_normalize.max_minNormalization(test_output_data, max_val_test, min_val_test)

    encode_decode_test = sess.run(y_pred, feed_dict={X: test_normalize_input})
    test_input_count = len(test_normalize_input)
    test_output_normal_data_pre = np.zeros((test_input_count, 1), dtype=np.float64)
    temp_output_pre = []
    for i in range(test_input_count):
        temp_output_row = []  # 输出行
        output_val = encode_decode_test[i][decoder_col_index]
        temp_output_row.append(output_val)
        temp_output_pre.append(temp_output_row)
    test_output_normal_data_pre = np.array(temp_output_pre)

    # 反归一化
    test_antinormalize_output = norml.Normalization.antinormalization_max_min(test_normalize_output, max_val_test,
                                                                              min_val_test)
    test_antinormalize_output_pre = norml.Normalization.antinormalization_max_min(test_output_normal_data_pre,
                                                                                  max_val_test, min_val_test)
    return test_antinormalize_output, test_antinormalize_output_pre


# RMSE: root mean squre error
# test_antinormalize_output：真实值
# test_antinormalize_output_pre：预测值
def calculate_RMSE(test_antinormalize_output, test_antinormalize_output_pre):
    # RMSE: root mean squre error
    error_array = test_antinormalize_output_pre - test_antinormalize_output
    # error = error_array.flatten()
    n = len(error_array)
    error_square_sum = (error_array ** 2).sum()
    RMSE = np.sqrt(error_square_sum / n)
    return RMSE


# MAPE: mean absolute percentage error
# test_antinormalize_output：真实值
# test_antinormalize_output_pre：预测值
def calculate_MAPE(test_antinormalize_output, test_antinormalize_output_pre):
    error_array = test_antinormalize_output_pre - test_antinormalize_output
    n = len(error_array)
    # 去除为零的项
    row_count = len(test_antinormalize_output)
    col_count = len(test_antinormalize_output[0])
    error_percent_array_sum = 0
    for r in range(row_count):
        for c in range(col_count):
            val = test_antinormalize_output[r][c]
            if abs(val) != 0:
                error_percent = abs(error_array[r][c] / val)
                error_percent_array_sum += error_percent
    MAPE = error_percent_array_sum / n
    return MAPE


# 获得下一批次数据
# inputData:list输入数据，输入数据的下面n行数据作为同一个时间特征数据
# next_n：当前行的以下行数作为输入向量
# output_col：输出数据所在的列，列数从0算起
# return ndarray


def obtain_next_batch(inputData, next_n, output_col):
    output_col += 1
    row_num = len(inputData)
    if row_num < next_n:
        # 有点问题，不能为0维数组
        result_input_data = []
        result_output_data = []
    else:
        last_num = row_num - next_n
        result_input_data = np.zeros((last_num, input_column_num), dtype=np.float64) #定义数组
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
                    if t_column == output_col:
                        temp_output_row.append(t)
                temp_input_data.append(temp_input_row)
                temp_output_data.append(temp_output_row)
            result_input_data = np.array(temp_input_data)
            result_output_data = np.array(temp_output_data)

        except Exception as ex:
            traceback.print_exc()
            print (Exception, ":", ex)
    return result_input_data, result_output_data


# 获得输入列数
def obtain_input_col_num(model_type, time_space_num, period_num, cycle_num):
    column_num = 0
    # 时间
    if model_type == 1:
        column_num = 7 * time_space_num
    # 时段
    elif model_type == 2:
        column_num = 7 * period_num
    # 周期
    elif model_type == 3:
        column_num = 7 * cycle_num
    # 综合
    else:
        column_num = 7 * time_space_num + 7 * period_num + 7 * cycle_num
    return column_num


# 获得输入/输出数据
# input_data: 输入数据
# model_type：模型类别;时间横向量/纵向量/周期向量/综合
# next_n
# output_col
def obtain_input_output(input_data, next_row, output_col_index, input_row_num, input_column_num,model_type_dic, model_type,
                        period_index, period_num, cycle_index, cycle_num):
    # 时间横向量
    if model_type == model_type_dic["time_span"]:
        link_train_input_data, link_train_output_data = feature.FeaturesExtraction.obtain_time_span_vector(
            input_data, next_row, input_row_num, input_column_num, output_col_index)
        return link_train_input_data, link_train_output_data

    # 时段纵向量
    elif model_type == model_type_dic["period"]:

        try:
            t_row_num = len(input_data)
            t_col_num = len(input_data[0])
            link_train_input_data = np.zeros((input_row_num, input_column_num), dtype= np.float64)
            link_train_output_data = np.zeros((input_row_num, 1))
            max_row = input_row_num + period_num * feature.FeaturesExtraction.perid_of_day
            if max_row < t_row_num:
                temp_input = []
                temp_output = []
                r_num = 0
                for r_t in input_data:
                    if r_num == input_row_num:
                        break
                    else:
                        # 时段向量
                        temp_output_row = []  # 输出行
                        out_val = np.float64(r_t[output_col_index])
                        temp_output_row.append(out_val)
                        period_val = int(r_t[period_index])
                        period_vector = feature.FeaturesExtraction.obtain_period_vert_vector(
                            input_data, period_index, period_val, period_num)
                        # 合并时段向量
                        t1_num = 0
                        t_r_p = []
                        for r_p in period_vector:
                            r_p = np.float64(r_p)
                            if t1_num == 0:
                                t_r_p = r_p
                                t1_num += 1
                            else:
                                t_r_p = np.append(t_r_p, r_p)  # 一维数组拼接
                        temp_output.append(temp_output_row)
                        temp_input.append(t_r_p)
                        r_num += 1
                link_train_input_data = np.array(temp_input)
                link_train_output_data = np.array(temp_output)
            else:
                raise Exception("矩阵行数不足以产生符合要求的时段向量！")
        except Exception as ex:
            traceback.print_exc()
            print(Exception, ":", ex)
        return link_train_input_data, link_train_output_data

    # 周期向量
    elif model_type == model_type_dic["cycle"]:
        try:
            t_row_num = len(input_data)
            t_col_num = len(input_data[0])
            link_train_input_data = np.zeros((input_row_num, input_column_num), dtype= np.float64)
            link_train_output_data = np.zeros((input_row_num, 1))
            max_row = input_row_num + cycle_num * feature.FeaturesExtraction.cycle_of_week * feature.FeaturesExtraction.perid_of_day
            if max_row < t_row_num:
                temp_input = []
                temp_output = []
                r_num = 0
                for r_t in input_data:
                    if r_num == input_row_num:
                        break
                    else:
                        # 周期向量
                        temp_output_row = []  # 输出行
                        out_val = np.float64(r_t[output_col_index])
                        temp_output_row.append(out_val)
                        period_val = int(r_t[period_index])
                        cycle_val = int(r_t[cycle_index])
                        cycle_vector = feature.FeaturesExtraction.obtain_cycle_vector(
                            input_data, period_val, cycle_index, cycle_num)
                        # 合并时段向量
                        t1_num = 0
                        t_r_p = []
                        for r_p in cycle_vector:
                            r_p = np.float64(r_p)
                            if t1_num == 0:
                                t_r_p = r_p
                                t1_num += 1
                            else:
                                t_r_p = np.append(t_r_p, r_p)  # 一维数组拼接
                        temp_output.append(temp_output_row)
                        temp_input.append(t_r_p)
                        r_num += 1
                link_train_input_data = np.array(temp_input)
                link_train_output_data = np.array(temp_output)
            else:
                raise Exception("矩阵行数不足以产生符合要求的时段向量！")
        except Exception as ex:
            traceback.print_exc()
            print(Exception, ":", ex)
        return link_train_input_data, link_train_output_data
        # return link_train_input_data, link_train_output_data

    # 时间横/纵/周期
    else:
        time_period_cycle_input, time_period_cycle_output = feature.FeaturesExtraction.time_period_cycle_vector(
            link_train_data, next_row, input_row_num, input_column_num, output_col_index,
            period_index,period_num,cycle_index, cycle_num)
        return time_period_cycle_input, time_period_cycle_output

# 数据分为两部分，一部分训练，一部分验证
# train_percentage:
# validate_percent:
# a = [randint(范围) for _ in range(次数)]：这样就会return一list的随机数字


def data_split(data, train_percent, validate_percent):
    #  随机取值 random
    r_num = len(data)
    col_num = len(data[0])
    train_num = int(r_num * train_percent)
    validate_num = r_num - train_num
    train_data = np.zeros((train_num, col_num), dtype= np.float64)  # 训练数据
    validate_data = np.zeros((validate_num, col_num), dtype=np.float64)  # 验证数据
    t_train_data = []
    t_validate_data = []
    try:
        all_ls = range(0, r_num)
        rand_train_ls = [randint(0, r_num - 1) for _ in range(train_num)]  # 训练随机数
        rand_validate_ls = [randint(0, r_num - 1) for _ in range(validate_num)]  # 验证随机数
        # for r in all_ls:
        #     if setOper.is_list_contains_id(rand_train_ls, r):
        #         continue
        #     else:
        #         rand_validate_ls.append(r)
        #
        #     # train data
        #     temp_row = copy.deepcopy(data[r])
        #     t_train_data.append(temp_row)
        # train_data = np.array(t_train_data)

        # train_data
        for r in rand_train_ls:
            temp_row = copy.deepcopy(data[r])
            t_train_data.append(temp_row)
        train_data = np.array(t_train_data)

        # validate data
        for r in rand_validate_ls:
            temp_row = copy.deepcopy(data[r])
            t_validate_data.append(temp_row)
        validate_data = np.array(t_validate_data)

    except Exception as ex:
        traceback.print_exc()
        print(Exception, ":", ex)
    return train_data, validate_data

# # 两个子图作法
# def subplot_figure_two(fig, x, y, x_label, y_label):
#     p1.plot(test_antinormalize_output, 'b*')  # 点
#     p1.plot(test_antinormalize_output, 'b',
#             label=str(link_id) + "_true link travel time")  # 线
#     p1.plot(test_antinormalize_output_pre, 'r*')
#
#     p1.plot(test_antinormalize_output_pre, 'r',
#             label=str(link_id) + "_predicted link travel time")
#     p1.set_title("Travel Time Prediction Using Autoencoder", fontsize=18)
#     p1.grid(True)
#     p1.legend()
#     # 设置label
#     p1.set_xlabel('Time Instant Number')
#     p1.set_ylabel('Travel Time(s)')
#
#     return fig


# Parameters
# 文件路径
file_type = ".txt"

# file path
root_dir = dir.RootDir.getRootDir()

# 时间横向量
file_name_train = "603-1_15minuteAllTimeSpan_train_674_last"
# file_name_train = "603-1_15minuteAllTimeSpan_oneweek_0_673"
file_name_test = "603-1_15minuteAllTimeSpan_oneweek_0_673"
file_path_train = root_dir + "/data/" + file_name_train + file_type
file_path_test = root_dir + "/data/" + file_name_test + file_type

# # 时段纵向量
# file_name_train = "603_-1_0800periodVertVector_train"
# file_name_test = "603_-1_0800periodVertVector_test"
# file_path_train = root_dir + "/data/" + file_name_train + file_type
# file_path_test = root_dir + "/data/" + file_name_test + file_type

# # 周期向量
# file_name_train = "603_-1_0730_0800cycleVector_train"
# file_name_test = "603_-1_0730_0800cycleVector_test"
# file_path_train = root_dir + "/data/" + file_name_train + file_type
# file_path_test = root_dir + "/data/" + file_name_test + file_type

# model 获得数据的模式
# 1：时间横向量
# 2：时段纵向量
# 3: 周期向量
# 4：三者的综合
model_type_dic = {"time_span": 1, "period": 2, "cycle": 3, "time_period_cycle": 4}  # 四种训练模式
model_type = 4  # 定义时间1/时段2/周期3/综合4
time_space_num = 4  # 时间横向量个数
period_index = 1  # 时段纵向量所在列索引，从0开始
period_num = 7  # 时段纵向量数目（默认一个周）
cycle_index = 0  # 周期向量所在列索引，从0开始
cycle_num= 4  # 周期向量数目（默认4个周）
# time_perid = 29 # 8:00-8:15 所在时段，从1开始
next_row = time_space_num - 1 # 所以总输入行为 time_space_num
output_col_index = 2  # 训练数据中目标输出列所在的列,从0开始
decoder_col_index = output_col_index  # 解码列,训练数据中目标输出列所在的列,从0开始

# 训练/测试 数据量
train_data_input_num = 7 * feature.FeaturesExtraction.perid_of_day
test_data_input_num = 1 * feature.FeaturesExtraction.perid_of_day
# input_row_num = 2 * feature.FeaturesExtraction.perid_of_day  # 模型输入

# Network Parameters
MAPE_expected = 0.2
learning_rate = 0.01
training_epochs = 4000 # 训练次数
display_step = 1
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_hidden_3 = 128 # 3nd layer num features
n_hidden_4 = 128 # 4nd layer num features
input_column_num = obtain_input_col_num(model_type, time_space_num, period_num, cycle_num) # 输入数据列数
n_input = input_column_num  # 输入特征个数

# 文件保存路径
# 实验图片保存路径
file_type_fig = ".png"
save_path_fig = root_dir + "/experimentResult/" + file_name_test + "_" + str(training_epochs) + file_type_fig

# 训练信息保存txt文件
save_path_train = root_dir + "/experimentResult/train_results_" + file_name_test + "_" + str(training_epochs) + file_type

# 训练数据/测试数据读取
train_data_origin = rwf.TxtFile.read_txt_file_data(file_path_train)
test_data_origin = rwf.TxtFile.read_txt_file_data(file_path_test)

# 路段id集合
link_id_list = rwf.TxtFile.obtain_link_id(train_data_origin)

link_train_data_dic = setOper.obtain_data_dic(train_data_origin, link_id_list)
link_test_data_dic = setOper.obtain_data_dic(test_data_origin, link_id_list)

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


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    row_count = len(train_data_origin) # row number

    # 训练信息
    train_result_list = []
    temp_str = "Total_epoch:" + str(training_epochs) + "\n"
    train_result_list.append(temp_str)

    # 均方根误差与平均绝对百分比误差
    RMSE_dic = {}
    MAPE_dic = {}
    # 真实值与预测值
    travel_time_true_dic = {}
    travel_time_pre_dic = {}
    # 归一化时的最大/最小值
    max_min_dic = {}

    # 每一个路段获得相应数据
    # 归一化并训练
    for link_id in link_id_list:
        link_data = link_train_data_dic.get(link_id)
        link_train_data = rwf.TxtFile.obtain_train_data(link_data)

        # 获得训练数据
        # link_train_input_data, link_train_output_data = obtain_next_batch(link_train_data, next_row, output_col_index)

        link_train_input_data, link_train_output_data = obtain_input_output(
            link_train_data, next_row,output_col_index, train_data_input_num, input_column_num, model_type_dic,model_type, period_index,period_num,cycle_index, cycle_num)

        # 将训练数据分为两部分，一部分训练/一部分验证
        link_train_input_split_data, link_validate_input_split_data = data_split(link_train_input_data, 0.8, 0.2)
        t_link_validate_output_split_data = link_validate_input_split_data[:,output_col_index]
        t_row = len(t_link_validate_output_split_data)
        link_validate_output_split_data = t_link_validate_output_split_data.reshape(t_row, 1)

        t_normalize = norml.Normalization(link_train_input_split_data)
        link_train_normalize_input, max_val_train, min_val_train = t_normalize.max_minNormalization(link_train_input_split_data)



        # 最大/最小值
        max_min_list = []
        max_min_list.append(max_val_train)
        max_min_list.append(min_val_train)
        max_min_dic[link_id] = max_min_list

        # 输入
        batch_xs = link_train_normalize_input
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})

            # 模型测试test：原始数据与预测数据，如何通过误差控制模型训练次数
            test_antinormalize_output, test_antinormalize_output_pre = model_predict(
                sess, y_pred, link_validate_input_split_data, link_validate_output_split_data)
            MAPE = calculate_MAPE(test_antinormalize_output, test_antinormalize_output_pre)
            if MAPE < MAPE_expected:
                temp_str = "link:" + str(link_id) + " Epoch:" + '%04d' % (epoch + 1) + "cost =" + "{:.9f}".format(c)
                train_result_list.append(temp_str)
                print ("达到期望精度！！" + str(MAPE))
                break
            else:
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("link:",link_id, "Epoch:", '%04d' % (epoch+1),
                         "cost =", "{:.9f}".format(c), "MAPE", "{:.9f}".format(MAPE))

                temp_str = "link:" + str(link_id) + " Epoch:" + '%04d' % (epoch + 1) + "cost =" + "{:.9f}".format(c)
                train_result_list.append(temp_str)

            # temp_str = "link:" + str(link_id) + "," + "Epoch:" + '%04d' % (epoch + 1) + "," + "cost =" + "{:.9f}".format(c) + '\n'
            # train_result_list.append(temp_str)
            # if epoch % display_step == 0:
            #     print("link:", link_id, "Epoch:", '%04d' % (epoch + 1),
            #           "cost =", "{:.9f}".format(c))

        print(str(link_id) + ":Optimization Finished!")

        # Applying encode and decode over test set
        # 测试数据预测
        link_data = link_test_data_dic.get(link_id)
        link_test_data = rwf.TxtFile.obtain_train_data(link_data)
        # 获得输入/输出数据
        # test_input_data, test_output_data = obtain_next_batch(link_test_data, next_row, output_col_index)

        test_input_data, test_output_data = obtain_input_output(
            link_test_data, next_row, output_col_index, test_data_input_num, input_column_num, model_type_dic, model_type, period_index, period_num,
            cycle_index, cycle_num)

        # 模型测试test：原始数据与预测数据
        test_antinormalize_output, test_antinormalize_output_pre = model_predict(sess, y_pred, test_input_data, test_output_data)

        RMSE = calculate_RMSE(test_antinormalize_output, test_antinormalize_output_pre)
        MAPE = calculate_MAPE(test_antinormalize_output, test_antinormalize_output_pre)

        RMSE_dic[link_id] = RMSE
        MAPE_dic[link_id] = MAPE
        travel_time_true_dic[link_id] = test_antinormalize_output
        travel_time_pre_dic[link_id] = test_antinormalize_output_pre
        train_result_list.append("均方根误差：" + str('%.6f' %RMSE) + "\n")
        train_result_list.append("平均绝对百分比误差：" + str('%.2f' %(MAPE*100)) + "%" + "\n")
        print ("RMSE:", '%.2f' %RMSE)
        print ("MAPE:", '%.2f' %(MAPE*100) + "%")
        # 训练信息写入txt文件
        rwf.TxtFile.write_to_file(train_result_list, save_path_train, 0)
    sess.close()


    # # # 画图:多个子图作法;多个子图作法
    # figure, axs = plt.subplots(1, 2, figsize=(15, 15))
    # p1 = plt.subplot(1,2,1) # 第一个图
    # p2 = plt.subplot(1,2,2) # 第二个图
    # # 图中标注画法
    # # p1.plot(t, f1(t), "g-", label="$f(t)=e^{-t} \cdot \cos (2 \pi t)$")
    # # p2.plot(t, f2(t), "r-.", label="$g(t)=\sin (2 \pi t) \cos (3 \pi t)$", linewidth=2)
    #
    # i = 0
    # for link_id in link_id_list:
    #     i+=1
    #     # 误差输出
    #     RMSE = RMSE_dic.get(link_id)
    #     MAPE = MAPE_dic.get(link_id)
    #     print(str(link_id) + "_RMSE", '%.6f' % RMSE)
    #     print(str(link_id) + "_MAPE", '%.6f' % MAPE)
    #
    #     label_str = str(link_id) + "_predicted link travel time, RMSE:" + str('%.4f' %RMSE) + ";MAPE:" + str('%.4f' %MAPE)
    #     test_antinormalize_output = travel_time_true_dic.get(link_id)
    #     test_antinormalize_output_pre = travel_time_pre_dic.get(link_id)
    #
    #     if i == 1:
    #         # 真实值与预测值
    #         # 图中多个子图的做法
    #         p1.plot(test_antinormalize_output, 'b*') #点
    #         p1.plot(test_antinormalize_output, 'b',
    #                 label = str(link_id) + "_true link travel time")  # 线
    #         p1.plot(test_antinormalize_output_pre, 'r*')
    #
    #         p1.plot(test_antinormalize_output_pre, 'r',
    #                 label = str(link_id) + "_predicted link travel time")
    #         p1.set_title("Travel Time Prediction Using Autoencoder", fontsize=18)
    #         p1.grid(True)
    #         p1.legend()
    #         # 设置label
    #         p1.set_xlabel('Time Instant Number')
    #         p1.set_ylabel('Travel Time(s)')
    #     elif i == 2:
    #         p2.plot(test_antinormalize_output, 'b*')  # 点
    #         p2.plot(test_antinormalize_output, 'b',
    #                  label=str(link_id) + "_true link travel time")  # 线
    #         p2.plot(test_antinormalize_output_pre, 'r*')
    #         p2.plot(test_antinormalize_output_pre, 'r',
    #                  label=str(link_id) + "_predicted link travel time")
    #         p2.set_title("Travel Time Prediction Using Autoencoder", fontsize=18)
    #         p2.grid(True)
    #         p2.legend()
    #         # 设置label
    #         p2.set_xlabel('Time Instant Number')
    #         p2.set_ylabel('Travel Time(s)')
    # plt.show()
    # plt.draw()
    # plt.waitforbuttonpress()

    # 单个图作法
    for link_id in link_id_list:
        # 误差输出
        RMSE = RMSE_dic.get(link_id)
        MAPE = MAPE_dic.get(link_id)
        print(str(link_id) + "_RMSE", '%.6f' % RMSE)
        print(str(link_id) + "_MAPE", '%.6f' % MAPE)

        label_str = str(link_id) + "_predicted link travel time, RMSE:" + str('%.4f' % RMSE) + ";MAPE:" + str(
            '%.4f' % MAPE)
        test_antinormalize_output = travel_time_true_dic.get(link_id)
        test_antinormalize_output_pre = travel_time_pre_dic.get(link_id)

        fig = plt.figure()
        plt.plot(test_antinormalize_output, 'b*')  # 点
        plt.plot(test_antinormalize_output, 'b',
                 label=str(link_id) + "_true link travel time")  # 线
        plt.plot(test_antinormalize_output_pre, 'r*')

        plt.plot(test_antinormalize_output_pre, 'r',
                 label=str(link_id) + "_predicted link travel time")
        y_min_val = min(min(test_antinormalize_output), min(test_antinormalize_output_pre))
        y_max_val = max(max(test_antinormalize_output), max(test_antinormalize_output_pre))
        plt.ylim(0, y_max_val * 1.5)

        plt.title("Travel Time Prediction Using Autoencoder", fontsize=16)
        plt.grid(True)
        plt.legend()
        # 设置label
        plt.xlabel('Time Instant Number')
        plt.ylabel('Travel Time(s)')
        plt.show()
        plt.draw()
        plt.waitforbuttonpress()
        # 保存为PNG等图形格式
        fig.savefig(save_path_fig)
        plt.clf()  # 清除图形
    print ("done!")
