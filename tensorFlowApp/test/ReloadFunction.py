# -*- coding: utf-8 -*-
# 乘法重定义有2个
# 如果乘法操作符“*”的左右操作数都是用户定义的数据类型，那么
# 调用 __mul__ 。若左边的操作数是原始数据类型，而右边是用户定义数据类
# 型，则调用 __rmul__ 。
class Line:
    def __init__(self, length):
        self.length = length
    def __str__(self):
        return self.length
        # 乘法重定义1，两个参数都是用户自定义数据类型，返回一个矩形
    def __mul__(self, other):
        return Rect(self.length, other.length)
        # 乘法重定义2，左边是原始数据类型，右边是用户自定义数据类型
    def __rmul__(self, other):
        return Line(self.length * other)
    def printline(self):
        print self.length

class Rect:
    def __init__(self, width, heigh):
        self.width = width
        self.heigh = heigh
    def printrect(self):
        print self.width, self.heigh


line1 = Line(20)
# 调用乘法重载2，左边是原始类型，右边是用户自定义类型
line2 = 3 * line1
# 交换调用顺序会出错
# lint3 = lin1*4 #it is error
line2.printline()
# 调用line的乘法重载1
rec1 = line1 * line2
rec1.printrect()  # 20,60