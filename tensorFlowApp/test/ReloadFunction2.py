# -*- coding: utf-8 -*-
# 定义在类的内部


class time:
    # 构造函数
    def __init__(self, hour=0, minutes=0, seconds=0):
        self.hour = hour
        self.minutes = minutes
        self.seconds = seconds
        # 函数的第一个参数是self，如果没有第二个参数，具体就用self.hour

    def printTime(self, t):
        print str(t.hour) + ":" + \
              str(t.minutes) + ":" + \
              str(t.seconds)
        # 定义增加秒数的函数,并设置缺省参数为30

    def increseconds(self, sec=30):
        self.seconds += sec
        if (self.seconds > 60):
            self.seconds = self.seconds - 60
            self.minutes += 1
        if (self.minutes > 60):
            self.minutes = self.minutes - 60
            self.hour += 1


t1 = time()
t1.hour = 10
t1.minutes = 8
t1.seconds = 50

# 第一个参数self就是类的对象t1
t1.printTime(t1)  # 10:8:50
t1.increseconds(20);
t1.printTime(t1)  # 10:9:10

# 有的参数之所以可以省略，是因为函数中已经给出了缺省的参数
t1.increseconds();
t1.increseconds();
t1.printTime(t1)  # 10:10:10

# 构造函数是任何类都有的特殊方法。当要创建一个类时，就调用构造函数。它的名字是： __init__ 。
t2 = time(10, 44, 45)
t2.printTime(t2)


print ("done!")