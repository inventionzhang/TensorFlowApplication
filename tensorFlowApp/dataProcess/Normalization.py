# -*- coding: utf-8 -*-
# 数据归一化
import numpy as np
import traceback


# 整体归一化
# 有必要做每列的归一化？
class Normalization:

    def __init__(self, data, max_val = 0, min_val = 0):
        if (max_val == 0) and (min_val == 0):
            self.max_val = np.max(data)
            self.min_val = np.min(data)
        else:
            self.max_val = max_val
            self.min_val = min_val

    # 最大/最小数据归一化
    def max_minNormalization(self, data, max_val = 0, min_val = 0):
        try:
            if (max_val == 0) and (min_val == 0):
                self.max_val = np.max(data)
                self.min_val = np.min(data)
            else:
                self.max_val = max_val
                self.min_val = min_val
            row_len = len(data)
            col_len = len(data[0])
            result_data = np.zeros((row_len, col_len), dtype=np.float64)  # 定义数组
            temp_input_data = []
            for r in range(row_len):
                temp_input_row = []
                for c in range(col_len):
                    x_val = data[r][c]
                    temp = (x_val - self.min_val) / (self.max_val - self.min_val)
                    # temp = np.float64('%.6f' % np.float64(temp))  # 保留小数点后6位
                    temp_input_row.append(temp)
                temp_input_data.append(temp_input_row)
            result_data = np.array(temp_input_data)

        except Exception, ex:
            traceback.print_exc()
            print (Exception, ":", ex)
        return result_data, self.max_val, self.min_val

    @staticmethod
    # 最大最小反归一化
    def antinormalization_max_min(data, max_val, min_val):
        try:
            row_len = len(data)
            col_len = len(data[0])
            result_data = np.zeros((row_len, col_len), dtype=np.float64)  # 定义数组
            temp_input_data = []  # 列表
            for r in range(row_len):
                temp_input_row = []
                for c in range(col_len):
                    x_val = data[r][c]
                    temp = x_val * (max_val - min_val) + min_val
                    # temp = np.float64('%.6f' % np.float64(temp))  # 保留小数点后6位
                    temp_input_row.append(temp)
                temp_input_data.append(temp_input_row)
            result_data = np.array(temp_input_data)

        except Exception, ex:
            traceback.print_exc()
            print (Exception, ":", ex)
        return result_data
