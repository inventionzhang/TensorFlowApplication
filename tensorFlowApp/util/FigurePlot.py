#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16-12-20 上午10:44
# @Author  : Aries
# @Site    : 
# @File    : FigurePlot.py
# @Software: PyCharm Community Edition

import matplotlib.pyplot as plt
from dataProcess import Features as feature



# 折线图
def plot_line(x_data, y_data, x_label, y_label, color, save_path_fig):
    fig = plt.figure()
    plt.plot(test_antinormalize_output, 'b*')  # 点
    plt.plot(x_data, y_data, 'b',
             label=str(link_id) + "_true link travel time")  # 线
    plt.plot(test_antinormalize_output_pre, 'r*')

    plt.plot(test_antinormalize_output_pre, 'r',
             label=str(link_id) + "_predicted link travel time")
    y_min_val = min(min(test_antinormalize_output), min(test_antinormalize_output_pre))
    y_max_val = max(max(test_antinormalize_output), max(test_antinormalize_output_pre))
    plt.ylim(0, y_max_val * 1.5)

    plt.title("Travel Time Prediction Using Autoencoder", fontsize=18)
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
    pass




