import pandas as pd
import numpy as np

import sys
import random
import math
import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def update_pos(old_pos, v, rand):
    for i in range(len(old_pos)):
        v_cal = 2 * math.atan(math.pi * v[i] / 2) / math.pi
        if rand < abs(v_cal):
            if old_pos[i] == 0:
                old_pos[i] == 1
            else:
                old_pos[i] == 0
    
    return old_pos

# 将向量2中的随机某些位变得和向量1一样
def make_2_like_1(list1, list2):
    for i in range(len(list1)):
        r = random.random()
        if r > 0.5:
            list2[i] = list1[i]
    
    return list2

# 两个列表各元素相减或相加
def two_list_cal(operation, lenght, list1, list2):
    if operation == '+':
        res = list()
        for i in range(0, lenght):
            res.append(list1[i]+list2[i])
        return res
    elif operation == '-':
        res = list()
        for i in range(0, lenght):
            res.append(list1[i]-list2[i])
        return res

    return list()

# 列表所有元素乘以一个数
def list_times_num(list1, num):
    for i in range(len(list1)):
        list1[i] = list1[i] * num
    
    return list1

# 产生随机整数数组
def random_int_list(length):
    res = list()
    for i in range(0,length):
        res.append(random.randint(0,1))
    return res

        
def inconsist_pair_num(df, pos):
    pair_num = 0
    schema = df.columns
    picked_att = list()
    for i in range(len(pos)):
        if pos[i] == 1:
            picked_att.append(schema[i])
    picked_att.append(schema[len(schema)-1])

    sub_set = df[picked_att]
    hash_cnt = {}
    for i in range(len(sub_set)):
        c_line = sub_set.iloc[i]
        c_list = list(c_line)
        condition_att = str(c_list[0:len(c_line)-1])
        decide_att = str(c_list[len(c_line)-1])
        if condition_att in hash_cnt:
            for d in hash_cnt[condition_att]:
                if decide_att != d:
                    pair_num += 1
            c_value = hash_cnt[condition_att]
            c_value.append(decide_att)
            hash_cnt[condition_att] = c_value
        else:
            c_value = [decide_att]
            hash_cnt[condition_att] = c_value

    # occured_pair_num[str(pos)] = pair_num
    # add_pair_num(pos, pair_num)
    return pair_num

# 计算适应度
def cal_fitness(df, pos, origin_pair_num):
    res = 0
    pair_num = inconsist_pair_num(df, pos)
    if pair_num == origin_pair_num:
        fenzi = 0
        for obj in pos:
            if obj == 1:
                fenzi+=1
        res = fenzi / (df.shape[1]-1)
    else:
        res = (pair_num + 1) / (origin_pair_num + 1)

    return res

# 计算KNN准确率
def knn_acc(df, pos):
    schema = df.columns
    picked_att = list()
    for i in range(len(pos)):
        if pos[i] == 1:
            picked_att.append(schema[i])
    picked_att.append(schema[len(schema)-1])

    data = df[picked_att]
    noOfFeatures = data.shape[1]
    x = data.values[:,0:noOfFeatures-1]
    y = data.values[:,noOfFeatures-1]
    def getAccuracy(features,label):
        # Split the data into testing and training
        X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size = 0.4)

        neigh = KNeighborsClassifier(n_neighbors = 5)
        neigh.fit(X_train,Y_train)
        prediction = neigh.predict(X_test)
        return accuracy_score(Y_test,prediction) * 100
    accuracy = getAccuracy(x,y)
    print('KNN accuracy: ',accuracy)

# 蝙蝠结构体
class bat:
    def __init__(self, pos, velocity, frequency, pulse_rate, a_loudness, fitness):
        self.pos = pos
        self.velocity = velocity
        self.frequency = frequency
        self.pulse_rate = pulse_rate
        self.init_pulse_rate = pulse_rate
        self.a_loudness = a_loudness
        self.fitness = fitness
    
    def show_info(self):
        print('I am a bat')
        print('pos: ' + str(self.pos))
        print('velocity: ' + str(self.velocity))
        print('frequency: ' + str(self.frequency))
        print('pulse_rate: ' + str(self.pulse_rate))
        print('init_pulse_rate: ' + str(self.init_pulse_rate))
        print('a_loudness: ' + str(self.a_loudness))
        print('fitness: ' + str(self.fitness))
        print('##################################################')

# 蝙蝠算法
# test_file = '/Users/qinliyang/desktop/python_project/论文对比算法/test_incosist.csv'
file_path = '/Users/qinliyang/desktop/python_project/datasets/csv/heart.csv'
df = pd.read_csv(file_path)
# df = df.drop_duplicates(keep='first')
start = datetime.datetime.now()
# 条件属性个数
att_num = df.shape[1]-1
origin_pair_num = inconsist_pair_num(df, [1]*att_num)

iteration_num = 40     # int(input('迭代次数：'))
population_size = 20    # int(input('种群规模：'))
a = 0.9
r = 0.9
f_max = 2
f_min = 0

best_fitness = 10000
best_pos = [0] * att_num

all_bats = list()
# 初始化种群
for i in range(0, population_size):
    init_pos = random_int_list(att_num)
    init_velocity = [0] * att_num
    init_frequency = f_min + (f_max - f_min) * random.random()
    init_pulse_rate = random.random() # random_int_list(att_num)
    init_a_loudness = random.random() # random_int_list(att_num)
    init_fitness = 10000
    
    temp_bat = bat(init_pos, init_velocity, init_frequency, init_pulse_rate, init_a_loudness, init_fitness)
    all_bats.append(temp_bat)

# 开始迭代
iter = 0
while iter < iteration_num:
    for i in range(0, population_size):
        current_bat = all_bats[i]
        # current_bat.show_info()
        rand = random.random()
        # 更新频率，速度和位置
        current_bat.frequency = f_min + (f_max - f_min) * random.random()
        current_bat.velocity = two_list_cal('+', att_num, current_bat.velocity, list_times_num(two_list_cal('-', att_num, current_bat.pos, best_pos), current_bat.frequency))
        new_pos = update_pos(current_bat.pos, current_bat.velocity, rand)

        # 使蝙蝠的位置更像当前最优解
        if rand > current_bat.pulse_rate:
            current_bat.pos = make_2_like_1(best_pos, current_bat.pos)
            
        fitness = cal_fitness(df, new_pos, origin_pair_num)
        # 如果移动后适应度下降，更新位置
        if fitness <= current_bat.fitness and rand < current_bat.a_loudness:
            current_bat.pos = new_pos
            current_bat.fitness = fitness
            current_bat.a_loudness = a * current_bat.a_loudness
            current_bat.pulse_rate = current_bat.init_pulse_rate * (1 - math.exp(-r * iter))
        
        if fitness <= best_fitness:
            best_fitness = fitness
            best_pos = new_pos
    
    
    print('best_pos: '+str(best_pos))
    print('best_fitness: '+str(best_fitness))
    print('iteration: '+str(iter))
    print('---------------------------------------')
    iter += 1
print(best_pos)    
# knn准确率
end = datetime.datetime.now()
print(end-start)
cnt = 0
for obj in best_pos:
    if obj == 1:
        cnt += 1
print(cnt)
knn_acc(df, best_pos)