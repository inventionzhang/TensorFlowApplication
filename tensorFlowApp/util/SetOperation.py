# -*- coding: utf-8 -*-
# 集合操作函数


# list中是否包含某一id
# list元素为int， id也是int类型
# 若包含，则返回true 1
# 否则，返回false 0

def is_list_contains_id(list_data, id):
    is_ok = 0
    # 如果不为空
    if list_data:
        for t in list_data:
            if t == id:
                is_ok = 1
                break
            else:
                continue
    else:
        is_ok = 0
    # if is_ok == 1:
    #     return
    # else:
    #     return false
    return is_ok


# 根据路段id获得对应的数据，并放入字典
# 输入：数据，以及路段id
# 输出：字典
# 相同路段的数据都存放在同一个区域
#

def obtain_data_dic(data, link_id_list):
    link_dic = {}
    row_count = len(data)
    for link_id in link_id_list:
        link_data = [] # 对应路段数据
        is_start_line = 0  # 是否为路段开始的数据
        is_end_line = 0  # 是否为路段结束数据
        i_num = 0 # 计数
        for r in data:
            i_num += 1
            temp_id = int(r[0])
            if link_id == temp_id:
                is_start_line = 1
                link_data.append(r)
                if i_num == row_count:  #最后一行
                    is_end_line = 1
                    link_dic[link_id] = link_data
            else:
                is_end_line = 1
                # 结束遍历路段数据
                if is_start_line & is_end_line:
                    link_dic[link_id] = link_data
                    break
    return link_dic



