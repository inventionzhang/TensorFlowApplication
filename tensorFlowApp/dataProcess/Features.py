#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16-12-22 上午11:30
# @Author  : Aries
# @Site    : 
# @File    : Features.py
# @Software: PyCharm Community Edition

import traceback
import numpy as np
import copy


# 特征提取
class FeaturesExtraction:

    perid_of_day = 96 # 一天中的时段
    cycle_of_week = 7 # 一周中的周期

    def __init__(self):

        print ("initial")

    # 时间横向量
    # 当前时刻的前n个时刻的数据作为时间横向两的输入
    # input_data_ls:list输入数据，输入数据的下面n行数据作为同一个时间特征数据
    # next_n：当前行的以下行数作为输入向量
    # input_row_num:输入数据行数
    # input_col_num：输入数据列数
    # output_col_index：输出数据所在的列，列数从0算起
    # return ndarray
    # python中默认传递地址
    # 传值的参数类型：数字，字符串，元组
    # 传址的参数类型：列表，字典
    # 若只取值要用copy或者deepcopy
    @staticmethod
    def obtain_time_span_vector(input_data_ls, next_row = 0,
                         input_row_num = 0, input_col_num = 0, output_col_index = 0):
        output_col_index += 1
        row_num = len(input_data_ls)
        if row_num < next_row and (input_row_num + next_row) > row_num:
            # 有点问题，不能为0维数组
            result_input_data = []
            result_output_data = []
        else:
            result_input_data = np.zeros((input_row_num, input_col_num), dtype=np.float64)  # 定义数组
            result_output_data = np.zeros((input_row_num, 1), dtype=np.float64)

            pro_data = []
            try:
                for i in range(0, input_row_num):
                    t_data = copy.deepcopy(input_data_ls[i])

                    for j in range(1, next_row + 1):
                        tt_data = copy.deepcopy(input_data_ls[i + j])
                        t_data.extend(tt_data)

                    pro_data.append(t_data)
                input_data = pro_data
                # output_data = pro_data[:,2]
                temp_input_data = []
                temp_output_data = []
                for r in pro_data:
                    temp_input_row = []  # 输入行
                    temp_output_row = []  # 输出行
                    t_column = 0
                    for c in r:
                        t_column += 1
                        t = np.float64('%.6f' % np.float64(c))  # 保留小数点后6位
                        temp_input_row.append(t)
                        if t_column == output_col_index:
                            temp_output_row.append(t)
                    temp_input_data.append(temp_input_row)
                    temp_output_data.append(temp_output_row)
                result_input_data = np.array(temp_input_data)
                result_output_data = np.array(temp_output_data)

            except Exception, ex:
                traceback.print_exc()
                print (Exception, ":", ex)
        return result_input_data, result_output_data

    # 指定时段的 时段纵向量
    # 获得时段纵向量：obtain_period_vert_vector
    # 获得某个时间所在指定列的时段纵向量
    # 输入：数组
    # 输出：时段纵向量数组
    # data:数据
    # col_index:相对于时间横向量 时段所在指定列,列从0开始
    # time_period:时段1-96
    # time_period_num: 时段数，默认为7,一个周的时段
    @ staticmethod
    def obtain_period_vert_vector(data, period_col_index, time_period_val, time_period_num=7):
        # result_data = np.zeros((last_num, column_num), dtype=np.float64)  # 定义数组
        try:
            # col_index = col_number - 1
            row_len = len(data)
            if row_len > time_period_num * FeaturesExtraction.perid_of_day:
                col_len = len(data[0])
                temp_input_data = []
                t_num = 0
                for r in range(row_len):
                    temp_input_row = []
                    val = int(data[r][period_col_index])
                    if val == int(time_period_val):
                        t_num += 1
                        for c in range(col_len):
                            temp_input_row.append(data[r][c])
                        temp_input_data.append(temp_input_row)
                        if t_num == time_period_num:
                            break
                        else:
                            continue

                    else:
                        continue
                result_data = np.array(temp_input_data)
            else:
                print ("错误：数据行数小于时段数！")
                result_data = []
                raise Exception("错误：数据行数小于时段数！无法提取时段纵向量！")

        except Exception, ex:
            traceback.print_exc()
            print(Exception, ":", ex)
        return result_data

    # 指定周期的 周期向量
    # 获得时段纵向量：obtain_cycle_vector
    # 获得某个时间所在指定列的周期向量
    # 输入：数组
    # 输出：周期向量数组
    # data:数据
    # cycle_col_index:相对于时间横向量 周期所在指定列,列从0开始
    # cycle_val:周期值 1-7
    # period_val:时段值
    # cycle_num: 周期数，默认为4,一个月的周期
    @staticmethod
    def obtain_cycle_vector(data, period_val, cycle_col_index, cycle_val,  cycle_num=4 ):
        try:
            row_len = len(data)
            if row_len > cycle_num * FeaturesExtraction.cycle_of_week * FeaturesExtraction.perid_of_day:
                col_len = len(data[0])
                temp_input_data = []
                t_num = 0
                for r in range(row_len):
                    temp_input_row = []
                    t_cy_val = int(data[r][cycle_col_index]) #周期值
                    t_p_val = int(data[r][cycle_col_index + 1])  #时段值
                    if (t_cy_val == cycle_val) and (t_p_val == period_val):
                        t_num += 1
                        for c in range(col_len):
                            temp_input_row.append(data[r][c])
                        temp_input_data.append(temp_input_row)
                        if t_num == cycle_num:
                            break
                        else:
                            continue

                    else:
                        continue
                result_data = np.array(temp_input_data)
            else:
                result_data = []
                raise Exception("错误：数据行数小于周期数！无法获得周期向量！")

        except Exception, ex:
            traceback.print_exc()
            print(Exception, ":", ex)
        return result_data

    # 综合向量： 时间/时段/周期向量
    # 获得某个时间所在指定列的周期向量
    # 输入：数组
    # 输出：周期向量数组
    # data:数据
    # next_row
    # input_row_num
    # input_column_num,
    # output_col_index
    # period_index
    # period_num,
    # cycyle_index:相对于时间横向量 周期所在指定列,列从0开始
    # cycle_val:周期值 1-7
    # cycle_num: 周期数，默认为4,一个月的周期
    @staticmethod
    def time_period_cycle_vector(data, next_row, input_row_num, input_column_num,
                                 output_col_index, period_index, period_num, cycle_index, cycle_num):

        t_row_num = len(data)
        t_col_num = len(data[0])
        time_period_cycle_input = np.zeros((input_row_num, input_column_num), dtype=np.float64)  # 定义时间/时段/周期数组
        output_data = np.zeros((input_row_num, 1), dtype=np.float64)
        temp_time_period_cycle = []
        try:
            max_row_period = input_row_num + period_num * FeaturesExtraction.perid_of_day
            max_row_cycle = input_row_num + cycle_num * FeaturesExtraction.cycle_of_week * FeaturesExtraction.perid_of_day
            if max_row_period < t_row_num and max_row_cycle < t_row_num:

                link_train_input_data_time_span, link_train_output_data_time_span = FeaturesExtraction.obtain_time_span_vector(
                    data, next_row, input_row_num, input_column_num, output_col_index)
                output_data = copy.deepcopy(link_train_output_data_time_span)
                # 合并时间/时段/周期向量
                for r_t in link_train_input_data_time_span:
                    # 时段向量
                    period_val = r_t[period_index]
                    period_vector = FeaturesExtraction.obtain_period_vert_vector(
                        data, period_index, period_val, period_num)

                    # 周期向量
                    cycle_val = r_t[cycle_index]
                    cycle_vector = FeaturesExtraction.obtain_cycle_vector(data, period_val,cycle_index, cycle_val, cycle_num)

                    # 合并时段向量
                    t1_num = 0
                    t_r_p = []
                    for r_p in period_vector:
                        r_p = np.float64(r_p)
                        if t1_num == 0:
                            t_r_p = r_p
                            t1_num +=1
                        else:
                            t_r_p = np.append(t_r_p, r_p) # 一维数组拼接

                    # 合并周期向量
                    t2_num = 0
                    t_r_cy = []
                    for r_cy in cycle_vector:
                        r_cy = np.float64(r_cy)
                        if t2_num == 0:
                            t_r_cy = r_cy
                            t2_num += 1
                        else:
                            t_r_cy = np.append(t_r_cy, r_cy)

                    concate_array = np.concatenate((r_t, t_r_p, t_r_cy),axis=0)
                    temp_time_period_cycle.append(concate_array)
            else:
                raise Exception("矩阵行数不足以产生符合要求的输入！")

            time_period_cycle_input = np.array(temp_time_period_cycle)

        except Exception, ex:
            traceback.print_exc()
            print(Exception, ":", ex)
        return time_period_cycle_input, output_data





