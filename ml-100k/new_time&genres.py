
import sys
import random
import math
import os
import codecs
import re
from collections import defaultdict
import time
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

T0=60 #信息半衰期
T1=3  #信息保持期
T_=30 #影响力加强周期
alpha=1
beta=0
epsilon=0.3
theta=3
new_ratings_txt_path='./new_ratings_newmodel'
random.seed(0)
time_div_path='./time_div.txt'
test_data_path='./test_data.txt'

list_for_test = [(0,1),(0.1, 0.9), (0.2, 0.8), (0.3, 0.7),(0.4, 0.6), (0.5, 0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1),(1,0)]
psi=0
omega=0

class New_algorithm(object):
    def __init__(self):
        self.model=New_Time_Genres()
    def culculate(self):
        filename_UI = os.path.join('train_data.txt')
        filename_item_genres = os.path.join('u.item')
        self.model.generate_dataset(filename_UI, filename_item_genres)
        fp_result=open('./testresult_new.txt','a')
        fp_result.write('\nalpha:1,beta:0 \n')
        for i in range(len(list_for_test)):
            global psi
            psi= list_for_test[i][0]
            global omega
            omega= list_for_test[i][1]
            fp_result.writelines('psi=%d'%psi+'\t'+'omega=%d'%omega+'\n')
            for j in range(5):
                test_RMSE=self.model.get_matrix()
                fp_result.writelines('j=%d'%j+'\t'+'test_RMSE=%f'%test_RMSE+'\n')
        fp_result.close()

class New_Time_Genres(object):

    def __init__(self):
        self.trainset=defaultdict(dict)
        self.testset=defaultdict(dict)
        self.user_genres_freq = defaultdict(dict)  # keys= users value={genre:weight}
        self.ui_time_weight=defaultdict(dict)
        self.ui_weight=defaultdict(dict)
        self.ui_rating=defaultdict(dict)
        self.time_div=defaultdict(dict)

    # load the UI-rating and build the matrix(dictionary exactly) into trainset and testset
    def loadfile_UI(self,filename):
        fp=open(filename ,'r')
        for i, line in enumerate(fp): # 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for 循环当中。
            yield line.strip('\r\n')# 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        fp.close()
        print('load %s succeed' % filename, file=sys.stderr)

    # add the corresponding genres into the matrix
    def load_itme_genres(self,filename):
        fp1=codecs.open(filename,'r',encoding = "ISO-8859-1")
        item_mat=defaultdict(list)
        for i, line in enumerate(fp1):
            line_list = []
            split_list=re.split(r'[| \n]',line)
            item_number=split_list[0]
            length=len(split_list)
            for j in range(length-20,length-1):
                line_list.append(split_list[j])
            item_mat[item_number]=line_list
        fp1.close()
        return item_mat

    def generate_dataset(self,filename_IU,filename_item_genres):
        trainset_len = 0
        testset_len = 0
        pivot=0.8
        item_gneres=self.load_itme_genres(filename_item_genres)
        aa=0
        for line in self.loadfile_UI(filename_IU):
            user, movie, rating, time = line.split()
            # split the data by pivot
            rating_dic_tr=defaultdict(list)
            rating_dic_tr[rating].append(int(time)/86400)
            rating_dic_tr[rating].extend(item_gneres[movie])
            self.trainset.setdefault(user, {})
            self.trainset[user][movie] = rating_dic_tr
            trainset_len += 1
        print('split training set and test set succ', file=sys.stderr)
        print('train set account = %s' % trainset_len, file=sys.stderr)
        print('test set account = %s' % testset_len, file=sys.stderr)
        self.load_time_div(time_div_path)

    def load_time_div(self,time_div_path):
        fp=open(time_div_path,'r')
        for line in fp.readlines():
            user,time_div=line.split()
            self.time_div[user]=float(time_div)
        fp.close()

    def ui_weight_all(self):
        for user, rest in self.trainset.items():
            for item, r_rest in rest.items():
                for rating, rr_rest in r_rest.items():
                    list_genres = rr_rest[1:20]
                    for i in range(19):
                        if (list_genres[i] == '1'):
                            if (str(i) in self.user_genres_freq[user].keys()):
                                if item in self.ui_rating[user].keys():
                                    try:
                                        a=self.ui_time_weight[user][item]
                                        b=self.user_genres_freq[user][str(i)]
                                        try:
                                            self.ui_weight[user][item] = math.pow(0.5, (alpha * a + beta * b))
                                        except OverflowError:
                                            self.ui_weight[user][item] = float(0)
                                        c=self.ui_weight[user][item]
                                    except KeyError:
                                        print('error in calculating ui_weight')
                                else:
                                    try:
                                        a = self.ui_time_weight[user][item]
                                        b = self.user_genres_freq[user][str(i)]
                                        try:
                                            self.ui_weight[user][item] = math.pow(0.5, (alpha * a + beta * b))
                                        except OverflowError:
                                            self.ui_weight[user][item] = float(0)

                                        c = self.ui_weight[user][item]
                                    except KeyError:
                                        print('error in calculating ui_weight')
                            else:
                                print('erro in initializing genres')
                    if item not in self.ui_rating[user].items():
                        self.ui_rating[user][item]= float(rating)*(1+epsilon*self.ui_weight[user][item])
                        d=self.ui_rating[user][item]
                    else:
                        print('error in ui_rating')
    def normolize_ui_rating(self):
        max=5
        min=5
        for user, rest in self.ui_rating.items():
            for item,rating in rest.items():
                if rating>max:
                    max=rating
                else:
                    if rating<min:
                        min=rating
        # print('max:'+str(max))
        # print('min:'+str(min))
        for user, rest in self.ui_rating.items():
            for item, rating in rest.items():
                self.ui_rating[user][item] = (4*(rating-min)/(max-min))+1
                e=self.ui_rating[user][item]

    def get_matrix(self):
        self.feature_weight()
        self.time_weight()
        self.ui_weight_all()
        self.normolize_ui_rating()
        self.write_to_txt(new_ratings_txt_path)
        test_RMSE=self.als_algorithm()
        return  test_RMSE

    def feature_weight(self):
        fp=open('freq_test','w')
        for user, rest in self.trainset.items(): # every user i
            user_genres=defaultdict(dict) #for every user, key=each genre, values= time_list
            user_genre_num=defaultdict(int)#for every user and every genre, the num of the correponding movies of the genre
            user_period_sum=defaultdict(int)
            for item, r_rest in rest.items():
                for rating, rr_rest in r_rest.items():
                    time=rr_rest[0]
                    list_genres=rr_rest[1:20]
                    for i in range(19):
                        if (str(i) in user_genres.keys()):
                            if(list_genres[i]=='1'):
                                user_genre_num[i]+=1
                                if(user_genres[str(i)][0]>=time):
                                    if(user_genres[str(i)][1]>=time):
                                        user_genres[str(i)][1]=time
                                else:
                                    user_genres[str(i)][0]=time

                        else: #for every genre, the lastest time_+ first movie time
                            if(list_genres[i]=='1'):
                                user_genre_num[i] += 1
                                user_genres[str(i)][0]=time
                                user_genres[str(i)][1]=time

            for i,temp in user_genres.items():
                for item, r_rest in rest.items():
                    for rating, rr_rest in r_rest.items():
                        time=int(rr_rest[0])
                        # print('time: '+str(time))
                        if time<=int(user_genres[str(i)][0]) and  time>=int(user_genres[str(i)][1]):
                            user_period_sum[str(i)]+=1
            """test to examine the result to freq_test"""
            fp.writelines('user: ' + user + '\n')
            for i,count_genre in user_genre_num.items():
                fp.writelines('i:'+str(i)+'\t'+str(count_genre)+'\t'+str(user_period_sum[str(i)])+'\n')
            # for i in user_period_sum.keys():
            #     print(i)
            self.user_genres_freq[user]=self.calculate_feature_weight(user,user_genres,user_genre_num,user_period_sum)
        fp.close()

    def calculate_feature_weight(self, user,user_genres,user_genre_num,user_period_sum):
        genre_freq = defaultdict(int)
        for genre, time_list in user_genres.items():
            Tnow=float(self.time_div[user])
            b=time_list[0]
            freq=user_genre_num[genre]/user_period_sum[genre]
            try:
                genre_freq[genre] = psi*((time_list[0] - time_list[1]) / (Tnow - time_list[0]) * T_)+omega*freq
            except:
                print('error user: '+str(user)+' and genre: '+str(genre))

        # print('user1 genre 16:')
        # print(genre_freq['16'])
        return genre_freq

    def time_weight(self):
        for user,rest in self.trainset.items():
            for item,r_rest in rest.items():
                for rating,rr_rest in r_rest.items():
                    time=rr_rest[0]
                    Tnow=self.time_div[user]
                    wtime=T1/T0*self.floor((Tnow-time)/T1)
                    self.ui_time_weight[user][item]=wtime

    def floor(self,time):
        return int(time)

    def recommender_time(self):
        tss1 = '1998-04-22 23:59:00'
        timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
        timeStamp = int(time.mktime(timeArray))
        return timeStamp #893260740

    def als_algorithm(self):
        os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
        conf = SparkConf().setMaster('local').setAppName('MovieRec').set("spark.executor.memory", "512m")
        sc = SparkContext(conf=conf)

        train_data = sc.textFile(new_ratings_txt_path)
        test_data = sc.textFile(test_data_path)
        train_ratings = train_data.map(self.get_rating)
        test_raings = test_data.map(self.get_rating)

        rank = 10
        iterations = 10
        train_ratings.cache()
        test_raings.cache()
        model = ALS.train(train_ratings, rank, iterations)

        self.get_test_data(test_data_path)
        test_input = test_raings.map(lambda x:(x[0],x[1]))
        pred_test = model.predictAll(test_input)
        test_reorg = test_raings.map(lambda x:((x[0],x[1]), x[2]))
        pred_reorg = pred_test.map(lambda x:((x[0],x[1]), x[2]))
        test_pred = test_reorg.join(pred_reorg)
        test_MSE = test_pred.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
        test_RMSE = math.sqrt(test_MSE)
        print('psi:'+str(psi))
        print('test_RMSE')
        print(test_RMSE)
        sc.stop()
        return test_RMSE

    def get_test_data(self,filepath):
        fp=open(filepath,'r')
        for line in fp.readlines():
            user,item,rating,time=line.split()
            self.testset[user][item]=float(rating)
        fp.close()

    def get_rating(sefl,str):
        arr = str.split()[0:3]
        user_id = int(arr[0])
        movie_id = int(arr[1])
        user_rating = float(arr[2])
        return Rating(user_id, movie_id, user_rating)

    def write_to_txt(self,filename):
        fp=open(filename,'w')
        for user,rest in self.ui_rating.items():
            for item,rating in rest.items():
                rating = self.ui_rating[user][item]
                fp.writelines(str(user)+'\t'+str(item)+'\t'+str(rating)+'\n')
        fp.close()

oa=New_algorithm()
oa.culculate()

print(len(oa.model.ui_rating))
print('ui_rating= sum all gerens 0.2*time_weight[user][item]+0.3user_fenre_freq[user][genre]')
print('ui_rating')
print([str(key)+' '+str(item) for key,item in oa.model.ui_rating.items()][0:5] )
print('user genres')
print([str(key)+' '+str(item)+'\n' for key,item in oa.model.user_genres_freq.items()][0:5] )
print('user item time_weicht')
print([str(key)+' '+str(item)+'\n' for key,item in oa.model.ui_time_weight.items()][0:5] )



