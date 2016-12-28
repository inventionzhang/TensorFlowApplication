# -*- coding: utf-8 -*-
from util import SetOperation as setOper
import os
import traceback


class TxtFile:

    train_col_start_index = 5

    # train_col_start_index:训练数据开始列，从0开始
    def __init__(self):
        print ("")

    # 读取原始txt数据
    @staticmethod
    def read_txt_file_data(file_path):
        data = []
        file = open(file_path, "r")
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
                # subStr = lineArray[ReadWritTxt.colum_num_train:]
                data.append(lineArray)
        file.closed
        return data

    # 获得输入路段的link id,
    # 返回list
    @staticmethod
    def obtain_link_id(data):
        list_id = []
        for r in data:
            temp_id = int(r[0])
            # 如果包含则continue，否则，加入list_id
            if setOper.is_list_contains_id(list_id, temp_id):
                continue
            else:
                list_id.append(temp_id)
        return list_id

    # 读取训练数据：
    # colum_num:参数，从某列开始截取数据
    @staticmethod
    def obtain_train_data(data):
        train_data = []
        for line in data:
            subStr = line[TxtFile.train_col_start_index:]
            train_data.append(subStr)
        file.closed
        return train_data

    # obtain test data
    @staticmethod
    def obtain_test_data(data):
        test_data = []
        # file = open(file_path_test, "r")
        # test_lines = file.readlines()
        # i = 0
        for line in data:
            # line = line.strip('\r\n')
            # i += 1
            # if i == 1:
            #     continue
            # else:
            # lineArray = line.split(",")
            subStr = line[TxtFile.train_col_start_index:]
            test_data.append(subStr)
        file.closed
        return test_data

    # 将信息写入文件
    # ls:数组数据
    # file_path_name:写入文件路径
    # write_style:写入文件方式，覆盖：0;追加：1  默认为覆盖方式写入
    @staticmethod
    def write_to_file(ls, file_path_name, write_style = 0):
        # wirte（）方法把字符串写入文件，writelines（）方法可以把列表中存储的内容写入文件。
        # 覆盖
        index = file_path_name.rindex("/")
        file_path = file_path_name[0:index]
        try:
            if(os.path.exists(file_path)):
                # print (file_path_name)
                if(write_style == 0):
                    f = file(file_path_name, "w+")
                    f.writelines(ls)
                # 追加
                elif(write_style == 1):
                    f = file(file_path_name, "a+")
                    f.writelines(ls)
                f.close()
            else:
                print ("文件路径不存在！" + "\n")
        except Exception, ex:
            traceback.print_exc()
            print (Exception, ":", ex)


class CsvFile:

    @staticmethod
    def read_csv(self):

        print ("done!")

    @staticmethod
    def write_csv(self):

        print ("done!")
