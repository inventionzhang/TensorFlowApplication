#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16-12-25 下午2:59
# @Author  : Aries
# @Site    : 
# @File    : TrafficFigure.py
# @Software: PyCharm Community Edition


import numpy as np
import matplotlib.pyplot as plt
from dataProcess import Features as feature
from fileOperation import ReadWriteFile as rwf
from util import SetOperation as setOper
from fileDirectory import Dir_Structure as dir
import traceback
import timeit
import time
import copy


def f1(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

def f2(t):
    return np.sin(2*np.pi*t)*np.cos(3*np.pi*t)

t = np.arange(0.0,5.0,0.02)


# 返回时间横向量列表数据
# data:
# start_time:取值开始时间
# time_index:时间所在索引，从0算起
# val_index: 数值所在列索引
# count：取值数目
# direction：取值的方向
def obtain_plot_data_time_span(data, start_time, time_index, val_index, count, direction):
    # result_array = np.zeros((count, 2), dtype=np.float64)
    result_data = []
    try:
        row_index = -1
        temp_count = 0
        for r in data:
            row_index += 1
            date_time_str = r[time_index]
            # time_array = time.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            if start_time == date_time_str:
                temp_count += 1
                # temp_ls = []
                # # temp_ls.append(r[4])  # 时间
                # temp_ls.append(r[val_index])  # 平均速度
                # result_data.append(copy.deepcopy(temp_ls))
                result_data.append(r[val_index])
                for i in range(row_index + 1, row_index + 1 + count):
                    temp_count += 1
                    # temp_ls = []
                    # # temp_ls.append(data[i][4])  # 时间
                    # temp_ls.append(data[i][val_index])  # 平均速度
                    # result_data.append(copy.deepcopy
                    result_data.append(data[i][val_index])
                    if temp_count == count:
                        break
            else:
                continue
        # result_array = np.array(result_data)
        return result_data

    except Exception as ex:
        traceback.print_exc()
        print(Exception, ":", ex)
        return result_data


# 返回时段向量列表数据
# type:向量类型,时间time/时段period/周期：cycle
# data:数据
# start_time:取值开始时间
# time_index:时间日期所在索引，从0算起
# travel_time_index：行程时间所在索引
# cycle_index:周期索引
# period：时段
# period_index：时段所在索引
# val_index: 数值所在列索引
# plot_count：作图所需取值数目
# direction：取值的方向
def obtain_plot_data_time_period_cycle_vector(type_vector, data, start_time, time_index, travel_time_index,
                                              cycle_index,cycle_val, period, period_index, plot_count, direction):
    # result_array = np.zeros((count, 2), dtype=np.float64)
    result_data = []
    try:
        # 寻找开始行
        type_ls = ["time","period","cycle"]
        row_index = -1
        temp_count = 0  # 数量
        start_index = 0
        for r in data:
            row_index += 1
            date_time_str = r[time_index]
            # time_array = time.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
            if start_time == date_time_str:
                temp_count += 1
                start_index = row_index
                result_data.append(r[travel_time_index])
                break
            else:
                continue

        # 时间
        if type_vector == type_ls[0]:
            pass

        # 时段
        elif type_vector == type_ls[1]:
            for i in range(start_index + 1, len(data)):
                period_val = np.int(data[i][period_index])
                if period_val == period:
                    temp_count += 1
                    result_data.append(data[i][travel_time_index])
                    if temp_count == plot_count:
                        break
                    else:
                        continue

                else:
                    continue
        # 周期
        else:
            for i in range(start_index + 1, len(data)):
                t_cycle_val = np.int(data[i][cycle_index])
                period_val = np.int(data[i][period_index])
                if t_cycle_val == cycle_val and period_val == period:
                    temp_count += 1
                    result_data.append(data[i][travel_time_index])
                    if temp_count == plot_count:
                        break
                    else:
                        continue

                else:
                    continue

    except Exception as ex:
        traceback.print_exc()
        print(Exception, ":", ex)

    return result_data


# 做交通行程时间曲线图
# fig
# x_data_dic：x轴数据
# y_data_dic：y轴数据
# period_ls：时段数
# x_label：x轴标签
# y_label：y轴标签
# title：标题
def plot_traffic_figure(fig, x_data_dic, y_data_dic,period_ls, x_label, y_label, title):
    try:
        # 做图
        # ax = fig.add_subplot(111)
        ax = fig.gca()
        # 确定y坐标轴最大最小
        y_min_val = 0
        y_max_val = 0
        for period in period_ls:
            t_x_val = x_data_dic[period]
            t_y_val = y_data_dic[period]
            t_y_min = np.float64(min(np.array(t_y_val)))
            t_y_max = np.float64(max(np.array(t_y_val)))
            if y_min_val > t_y_min:
                y_min_val = t_y_min
            if y_max_val < t_y_max:
                y_max_val = t_y_max

        x_max_range = len(x_data_dic[period_ls[0]]) + 1
        x_val_num = range(1, x_max_range)

        ax.set_ylim(0, y_max_val * 1.5)
        line_dic = {}

        for period in period_ls:
            x_val = x_data_dic[period]
            y_val1 = y_data_dic[period]
            color_style1 = color_dic[period]
            # 画图
            line1 = plt.plot(x_val_num, y_val1, color_style1)
            # line1 = plt.plot(x_val_num, y_val1, color_style1, label = "13:00")
            line_dic[period] = line1
            x_tick_label = []
            x_tick_mum = []
            for i in x_val_num:
                if i % 16 == 0:
                    x_tick_mum.append(i)
                    dt = x_val[i]
                    # index = dt.index(" ")
                    t_array = dt.split(" ")
                    hms = t_array[1]
                    x_tick_label.append(hms)
                if i == len(x_val) - 1:
                    x_tick_label.append(x_val[i].split(" ")[1])

                    # ax.set_xticks(x_tick_mum,x_tick_label)
                    # plt.xticks(x_tick_mum, x_tick_label)

        # ax.set_xlabel(u'时刻')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel(x_label)
        ax.yaxis.set_ticks_position('left')
        ax.set_ylabel(y_label)
        ax.set_title(title, fontsize=16)
        # 必须放入对象line，不知道为什么
        # 但是传参时，会出现图例解释文字只显示第一个字符，需要在传参时在参数后加一个逗号
        # 加一个逗号
        plt.legend((legend_ls[0], legend_ls[1], legend_ls[2], legend_ls[3],), loc='upper right', fontsize=10)
        plt.grid(True)
        plt.show()
        plt.draw()
        # plt.waitforbuttonpress()
        # fig.savefig(save_path_fig)

    except Exception as e:
        traceback.print_exc()
        print (Exception, ":", e)


if __name__ == "__main__":
    # parameters
    period_of_day = 96
    cycle_of_week = 7
    start_time = "2014-07-30 23:45:00"
    time_index = 4  # 日期时间所在索引
    travel_time_index = 7  # 行程时间所在索引
    count = 1 * period_of_day
    period_count = 8 * cycle_of_week
    period_index = 6
    type_ls = ["time","period","cycle"]
    cycle_count = 22  # 周期向量数
    cycle_index = 5
    cycle_val = 0

    # 时段向量作图，参数修改区域
    # 时段向量作图，参数修改区域
    # 时段向量作图，参数修改区域
    # 时段向量作图，参数修改区域
    # type_vector = type_ls[1]
    # title = "Period line chart of link travel time for link 603"
    # plot_count = period_count

    # # 7:00 -8:00 做图
    # period_ls = [29, 30, 31, 32]
    # legend_ls = ["7:00-7:15", "7:15-7:30", "7:30-7:45", "7:45-8:00"]
    # color_dic = {29: 'rs--', 30: 'bo:', 31: 'kp-.', 32: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_7:00-8:00"

    # # 8:00 -9:00 做图
    # period_ls = [33, 34, 35, 36]
    # legend_ls = ["8:00-8:15", "8:15-8:30", "8:30-8:45", "8:45-9:00"]
    # color_dic = {33: 'rs--', 34: 'bo:', 35: 'kp-.', 36: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_8:00-9:00"

    # # 11:00 -12:00 做图
    # period_ls = [45, 46, 47, 48]
    # legend_ls = ["11:00-11:15", "11:15-11:30", "11:30-11:45", "11:45-12:00"]
    # color_dic = {45: 'rs--', 46: 'bo:', 47: 'kp-.', 48: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_11:00-12:00"

    # # 12:00 -13:00 做图
    # period_ls = [49, 50, 51, 52]
    # legend_ls = ["12:00-12:15", "12:15-12:30", "12:30-12:45", "12:45-13:00"]
    # color_dic = {49: 'rs--', 50: 'bo:', 51: 'kp-.', 52: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_12:00-13:00"

    # # # 17:00 -18:00 做图
    # period_ls = [69, 70, 71, 72]
    # legend_ls = ["17:00-17:15", "17:15-17:30", "17:30-17:45", "17:45-18:00"]
    # color_dic = {69: 'rs--', 70: 'bo:', 71: 'kp-.', 72: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_17:00-18:00"

    # # # 18:00 -19:00 做图
    # period_ls = [73, 74, 75, 76]
    # legend_ls = ["18:00-18:15", "18:15-18:30", "18:30-18:45", "18:45-19:00"]
    # color_dic = {73: 'rs--', 74: 'bo:', 75: 'kp-.', 76: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "period_vector_603_18:00-19:00"

    # 周期向量作图，参数修改区域
    # 周期向量作图，参数修改区域
    # 周期向量作图，参数修改区域
    # 周期向量作图，参数修改区域
    cycle_val = 3
    type_vector = type_ls[2]
    title = "Cycle line chart of link travel time for link 603"
    plot_count = cycle_count

    # 7:00 -8:00 做图
    # period_ls = [29, 30, 31, 32]
    # legend_ls = ["7:00-7:15", "7:15-7:30", "7:30-7:45", "7:45-8:00"]
    # color_dic = {29: 'rs--', 30: 'bo:', 31: 'kp-.', 32: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "cycle_vector_603_7:00-8:00"

    # 8:00 -9:00 做图
    # period_ls = [33, 34, 35, 36]
    # legend_ls = ["8:00-8:15", "8:15-8:30", "8:30-8:45", "8:45-9:00"]
    # color_dic = {33: 'rs--', 34: 'bo:', 35: 'kp-.', 36: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "cycle_vector_603_8:00-9:00"

    # # 11:00 -12:00 做图
    # period_ls = [45, 46, 47, 48]
    # legend_ls = ["11:00-11:15", "11:15-11:30", "11:30-11:45", "11:45-12:00"]
    # color_dic = {45: 'rs--', 46: 'bo:', 47: 'kp-.', 48: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "cycle_vector_603_11:00-12:00"

    # # 12:00 -13:00 做图
    # period_ls = [49, 50, 51, 52]
    # legend_ls = ["12:00-12:15", "12:15-12:30", "12:30-12:45", "12:45-13:00"]
    # color_dic = {49: 'rs--', 50: 'bo:', 51: 'kp-.', 52: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "cycle_vector_603_12:00-13:00"

    # # # 17:00 -18:00 做图
    # period_ls = [69, 70, 71, 72]
    # legend_ls = ["17:00-17:15", "17:15-17:30", "17:30-17:45", "17:45-18:00"]
    # color_dic = {69: 'rs--', 70: 'bo:', 71: 'kp-.', 72: 'gx-'}  # 颜色/线性/标记风格
    # save_name = "cycle_vector_603_17:00-18:00"

    # # 18:00 -19:00 做图
    period_ls = [73, 74, 75, 76]
    legend_ls = ["18:00-18:15", "18:15-18:30", "18:30-18:45", "18:45-19:00"]
    color_dic = {73: 'rs--', 74: 'bo:', 75: 'kp-.', 76: 'gx-'}  # 颜色/线性/标记风格
    save_name = "cycle_vector_603_18:00-19:00"

    file_type = ".txt"
    # file path
    root_dir = dir.RootDir.getRootDir()
    # file_name_train = "603-1_15minuteAllTimeSpan_oneweek_0_673"
    file_name_train = "603-1_15minuteAllTimeSpan"
    file_path_train = root_dir + "/data/" + file_name_train + file_type

    file_type_fig = ".png"
    save_path_fig = root_dir + "/experimentResult/" + save_name + file_type_fig

    train_data_origin = rwf.TxtFile.read_txt_file_data(file_path_train)

    link_id_list = rwf.TxtFile.obtain_link_id(train_data_origin)
    link_train_data_dic = setOper.obtain_data_dic(train_data_origin, link_id_list)

    for link_id in link_id_list:
        link_data = link_train_data_dic.get(link_id)

        # 存放做图数据
        x_data_dic = {}
        y_data_dic = {}
        for period in period_ls:
            # y_data_plot = obtain_plot_data_time_span(link_data, start_time, time_index, travel_time_index, count, "past")
            # x_data_plot = obtain_plot_data_time_span(link_data, start_time, time_index, time_index, count, "past")

            y_data_plot = obtain_plot_data_time_period_cycle_vector(type_vector, link_data, start_time, time_index, travel_time_index, cycle_index, cycle_val, period, period_index, plot_count, "past")

            x_data_plot = obtain_plot_data_time_period_cycle_vector(type_vector, link_data, start_time, time_index,time_index, cycle_index, cycle_val, period, period_index, plot_count, "past")
            count = len(y_data_plot)
            x_val = []
            y_val = []
            for i in range(count-1, -1, -1):
                x_val.append(x_data_plot[i])
                y_val.append(np.float64(y_data_plot[i]))
            x_data_dic[period] = x_val
            y_data_dic[period] = y_val

        # 做图
        fig = plt.figure()
        x_label = 'Time Instant'
        y_label = 'Travel Time(s)'
        plot_traffic_figure(fig, x_data_dic, y_data_dic,period_ls, x_label, y_label, title)
        fig.savefig(save_path_fig)
        # plt.clf()  # 清除图形
        print ("done")