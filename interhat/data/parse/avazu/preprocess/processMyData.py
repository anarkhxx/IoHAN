import pickle
import numpy as np
import pandas as pd
import csv

#（1）默认的一些初始化

#默认train存储
f = open('train_ind.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
ftarget=open('train_label.csv', 'w', encoding='utf-8', newline='')
csv_writer_target=csv.writer(ftarget)

#测试数据存储
testf = open('test_ind.csv', 'w', encoding='utf-8', newline='')
testcsv_writer = csv.writer(testf)
testftarget=open('test_label.csv', 'w', encoding='utf-8', newline='')
testcsv_writer_target=csv.writer(testftarget)


#（2）开始某个用户的visit和code数据处理,首先定好visit个数，和每次visit的codes数

#先处理train的数据
data_train_visit=np.array(pickle.load(open("data_train_3digit.pkl", 'rb')))
test_data_train_visit=np.array(pickle.load(open("data_test_3digit.pkl", 'rb')))
TF_IDF_weight=pickle.load(open("TF_IDF.pkl", 'rb'))

max_visit_count=0
max_codes_count=0
min_codes_count=44
maxNumber=0
compute_aver_codes=0
compute_aver_codes_count=0



for visit in data_train_visit:
    #print(visit)
    #用户序号
    #print(visit[0])
    #csv_writer.writerow(visit[1])
    #这是numeric相关
    #print(visit[2])
    #time相关
    #print(visit[3])
    #用户住院年龄和性别
    #print(visit[4])
    #code visits
    if len(visit[1])>max_visit_count:
        max_visit_count=len(visit[1])
    for codes in visit[1]:
        if(len(codes)>max_codes_count):
            max_codes_count=len(codes)
        #计算每个visit的average
        compute_aver_codes+=len(codes)
        compute_aver_codes_count+=1
        for code in codes:
            if code>maxNumber:
                maxNumber=code

for visit in test_data_train_visit:
    if len(visit[1])>max_visit_count:
        max_visit_count=len(visit[1])
    for codes in visit[1]:
        if(len(codes)>max_codes_count):
            max_codes_count=len(codes)
        # 计算每个visit的average
        compute_aver_codes += len(codes)
        compute_aver_codes_count += 1
        for code in codes:
            if code>maxNumber:
                maxNumber=code
    for codes in visit[1]:
        if(len(codes)<min_codes_count):
            min_codes_count=len(codes)



compute_aver_codes=int(compute_aver_codes/compute_aver_codes_count)+1

print(max_visit_count)
print(max_codes_count)
print(min_codes_count)
print(maxNumber)
print(int(compute_aver_codes))
# 42
# 39
# 1070
# 11

max_codes_count=11

# （3）达成规范后，处理数据

for visit in data_train_visit:
    # 每个病人向量
    visits_source=visit[1]
    #对于每个vistis_source=34*36打成一维度
    #修改成3个visit，每个visit36个codes。
    max_visit_count=3
    one_res=[]
    for codes in visits_source:
        # 使用TF-IDF算法得到的权重进行排序
        # TF_IDF_value = {}
        # sorted_result = []
        # for index in range(len(codes)):
        #     TF_IDF_value[codes[index]] = TF_IDF_weight[codes[index]]
        # sorted_result = sorted(TF_IDF_value.items(), key=lambda item: item[1], reverse=True)
        # for index in range(len(sorted_result)):
        #     sorted_result[index] = sorted_result[index][0]

        for index in range(min(len(codes),max_codes_count)):
            code=codes[index]
            one_res.append(code)
            #不能超过max_codes_count
        for temp in range(max_codes_count-len(codes)):
            one_res.append(maxNumber+1)
        if(len(one_res)==max_visit_count*max_codes_count):
            break
    for temp1 in range(max_visit_count-len(visits_source)):
        for temp2 in range(max_codes_count):
            one_res.append(maxNumber+1)
    # print(one_res)
    # print(len(one_res))

    csv_writer.writerow(one_res)


for visit in test_data_train_visit:
    visits_source=visit[1]
    #对于每个vistis_source=34*36打成一维度
    # 修改成3个visit，每个visit36个codes。
    max_visit_count = 3
    one_res=[]
    for codes in visits_source:
        # 使用TF-IDF算法得到的权重进行排序
        # TF_IDF_value = {}
        # sorted_result = []
        # for index in range(len(codes)):
        #     TF_IDF_value[codes[index]] = TF_IDF_weight[codes[index]]
        # sorted_result = sorted(TF_IDF_value.items(), key=lambda item: item[1], reverse=True)
        # for index in range(len(sorted_result)):
        #     sorted_result[index] = sorted_result[index][0]

        for index in range(min(len(codes),max_codes_count)):
            code=codes[index]
            one_res.append(code)
        for temp in range(max_codes_count-len(codes)):
            one_res.append(maxNumber+1)
        if (len(one_res) == max_visit_count * max_codes_count):
            break

    for temp1 in range(max_visit_count-len(visits_source)):
        for temp2 in range(max_codes_count):
            one_res.append(maxNumber+1)
    #print(one_res)
    # print(len(one_res))
    testcsv_writer.writerow(one_res)

#（4）用户label数据处理
data_train_target=np.array(pickle.load(open("target_train.pkl", 'rb')))
for label in data_train_target:
    #print(label[1])
    csv_writer_target.writerow(str(label[1]))

test_data_train_target=np.array(pickle.load(open("target_test.pkl", 'rb')))
for label in test_data_train_target:
    #print(label[1])
    testcsv_writer_target.writerow(str(label[1]))

f.close()
ftarget.close()
testf.close()
testftarget.close()