# -*- coding: utf-8 -*-
# spark-submit movie_rec.py
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql import Row
import  os
import sys
import random
import math
import codecs
import re
from collections import defaultdict
import time,datetime
import codecs

# 获取所有movie名称和id对应集合
def movie_dict(filename):
    dict = {}
    fp = codecs.open(filename,'r', encoding="ISO-8859-1")
    for line in fp:
        arr = re.split(r'[|]',line)
        movie_id = int(arr[0])
        movie_name = str(arr[1])
        dict[movie_id] = movie_name
    fp.close()
    return dict


# 转换用户评分数据格式
def get_rating(str):
    arr=str.split()[0:3]
    user_id = int(arr[0])
    movie_id = int(arr[1])
    user_rating = float(arr[2])
    return Rating(user_id, movie_id, user_rating)

os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3.6"
conf = SparkConf().setMaster('local').setAppName('MovieRec').set("spark.executor.memory", "512m")
sc = SparkContext(conf=conf)

# 加载数据
movies = movie_dict('/Users/apple/PycharmProjects/recommanded_system/ml-100k/u.item')
sc.broadcast(movies)
data = sc.textFile('/Users/apple/PycharmProjects/recommanded_system/ml-100k/usmall.data')

print('dict')
print([i for i in movies.items()][:5])
print('data')
data.first()
# 转换 (user, product, rating) tuple
print('rowrating')
ratings=data.map(get_rating)

# 建立模型
rank = 10
iterations = 5
model = ALS.train(ratings, rank, iterations)

# 对指定用户ID推荐
userid = 10
user_ratings = ratings.filter(lambda x: x[0] == userid)

# 按得分高低推荐前10电影
rec_movies = model.recommendProducts(userid, 10)
print('\n################################\n')
print('recommend movies for userid %d:' % userid)
for item in rec_movies:
    print('name:' + movies[item[1]] + '==> score: %.2f' % item[2])
print('\n################################\n')
sc.stop()