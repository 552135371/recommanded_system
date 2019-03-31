import sys
import random
import math
import os
import codecs
import re
from collections import defaultdict
import time

# 先抽取每个用户的用户-物品-评分-时间序列，然后计算每个用户评分时间的最大值和最小值存于dict之中
# 根据dict中的最大最小值，确定每个用户的70%的时间段为训练集 30%为测试集
file_path='./ml-100k/u.data'
all_data=defaultdict(dict)
user_time_list=defaultdict(dict) #0---max,1---min
def loadfile_UI( filename):
    fp = open(filename, 'r')
    for i, line in enumerate(fp):  # 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for 循环当中。
        yield line.strip('\r\n')  # 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
    fp.close()
    print('load %s succeed' % filename, file=sys.stderr)

def generate_dataset(filename_IU):
    trainset_len = 0
    testset_len = 0
    pivot = 0.8
    aa = 0
    for line in loadfile_UI(filename_IU):
        user, movie, rating, time = line.split()
        # split the data by pivot
        # if random.random() < pivot:
        #     rating_dic_tr = defaultdict(list)
        #     rating_dic_tr[rating].append(int(time) / 86400)
        #     self.trainset.setdefault(user, {})
        #     self.trainset[user][movie] = rating_dic_tr
        #     trainset_len += 1
        # else:
        #     rating_dic_te = defaultdict(list)
        #     rating_dic_te[rating].append(int(time) / 86400)
        #     self.testset.setdefault(user, {})
        #     self.testset[user][movie] = rating_dic_te
        #     testset_len += 1
        list_time_rating=[int(rating),int(time)]
        all_data[user][movie]=list_time_rating


def seperate_data(all_data):
    for user, rest in all_data.items():
        for item, rating_time_list in rest.items():
            if str(user) in user_time_list.keys():
                if rating_time_list[1]>user_time_list[user][0]:
                    user_time_list[user][0]=rating_time_list[1]
                else:
                    if rating_time_list[1]<user_time_list[user][1]:
                        user_time_list[user][1] = rating_time_list[1]
            else:
                # print(rating_time_list)
                list=[rating_time_list[1],rating_time_list[1]]
                user_time_list[user]=list

        max_time=user_time_list[user][0]
        print('use %s max: %d'%(str(user), max_time))
        min_time=user_time_list[user][1]
        print('use %s min: %d' % (str(user), min_time))
        """********************************************************"""
        time_div_first=(float(min_time)+(float(max_time)-float(min_time))*0.8)#区分训练集的时间
        time_div_last=(float(min_time)+(float(max_time)-float(min_time))*0.8)#区分测试集的时间
        fp_time_div.writelines(str(user)+'\t'+str(time_div_last/86400)+'\n')
        for item,rating_time_list in rest.items():
            if rating_time_list[1]>time_div_last:
                fp_test.writelines(str(user)+'\t'+str(item)+'\t'+str(rating_time_list[0])+'\t'+str(rating_time_list[1])+'\n')
            else:
                if rating_time_list[1]<time_div_first:
                    fp_train.writelines(str(user)+'\t'+str(item)+'\t'+str(rating_time_list[0])+'\t'+str(rating_time_list[1])+'\n')

fp_test=open('./ml-100k/test_data.txt','w')
fp_train=open('./ml-100k/train_data.txt','w')
fp_time_div=open('./ml-100k/time_div.txt','w')
generate_dataset(file_path)
seperate_data(all_data)
fp_test.close()
fp_train.close()
fp_time_div.close()