# change none but max
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from layer_assist import Unit
#from Unit import call
import tensorflow as tf


def load_assist_data(dataset,model_name):
    if model_name == 'tgcn':
        print("11")
        sz_adj = pd.read_excel('./TGCNdata/%s_adj.xlsx'%dataset, header=None)
        adj = np.mat(sz_adj)
        data = pd.read_excel('./TGCNdata/%s_feature_matrix_X.xlsx'%dataset)
    else:
        print("10")
        sz_adj = pd.read_excel('./Scene-GCNdata/%s_adj.xlsx'%dataset, header=None)
        adj = np.mat(sz_adj)
        data = pd.read_excel('./Scene-GCNdata/%s_feature_matrix_X.xlsx'%dataset)
    return data, adj

#data, adj = load_assist_data('3611817550')
#time_len = data.shape[0]
#num_nodes = data.shape[1]

def preprocess_data(data1,  time_len, train_rate, seq_len, pre_len, model_name, scheme):
    train_size = int(time_len * train_rate)#时间序列长度的0.8
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]            
            
    # if model_name == 'tgcn'or'gru':################TGCN###########################
    if model_name == 'tgcn':
        print("11")
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]#同一批的训练长度，向后循环
            trainX.append(a1[0 : seq_len])#自变量训练
            trainY.append(a1[seq_len : seq_len + pre_len])#相对于自变量的因变量训练长度
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0 : seq_len])
            testY.append(b1[seq_len : seq_len + pre_len])
            
    elif model_name == 'gru':
        print("12")
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]#同一批的训练长度，向后循环
            trainX.append(a1[0 : seq_len])#自变量训练
            trainY.append(a1[seq_len : seq_len + pre_len])#相对于自变量的因变量训练长度
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0 : seq_len])
            testY.append(b1[seq_len : seq_len + pre_len])
            
    elif model_name == 'Scene-GCN':################Scene-GCN###########################
        print("10")
        direction = pd.read_excel('./Scene-GCNdata/3611817550_direction.xlsx',header = None)
        #direction = np.transpose(direction)# 矩阵转置
        direction_max = np.max(np.max(direction))
        #direction_nor = direction/direction_max# 归一化
        direction_nor = direction

        beach_width = pd.read_excel('./Scene-GCNdata/3611817550_beach_width.xlsx',header = None)
        beach_width = np.mat(beach_width)
        beach_width_max = np.max(np.max(beach_width))
        #beach_width_nor = beach_width/beach_width_max# 归一化
        beach_width_nor = beach_width
        beach_width_nor_train = beach_width_nor[0:train_size]
        beach_width_nor_test = beach_width_nor[train_size:time_len]

        Reservoir_water_level = pd.read_excel('./Scene-GCNdata/3611817550_Reservoir_water_level.xlsx',header = None)
        Reservoir_water_level = np.mat(Reservoir_water_level)
        Reservoir_water_level_max = np.max(np.max(Reservoir_water_level))
        #Reservoir_water_level_nor = Reservoir_water_level/Reservoir_water_level_max# 归一化
        Reservoir_water_level_nor = Reservoir_water_level
        Reservoir_water_level_nor_train = Reservoir_water_level_nor[0:train_size]
        Reservoir_water_level_nor_test = Reservoir_water_level_nor[train_size:time_len]

        phreatic_line = pd.read_excel('./Scene-GCNdata/3611817550_phreatic_line.xlsx',header = None)
        phreatic_line = np.mat(phreatic_line)
        phreatic_line_max = np.max(np.max(phreatic_line))
        #phreatic_line_nor = phreatic_line/phreatic_line_max# 归一化
        phreatic_line_nor = phreatic_line
        phreatic_line_nor_train = phreatic_line_nor[0:train_size]
        phreatic_line_nor_test = phreatic_line_nor[train_size:time_len]

        precipitation = pd.read_excel('./Scene-GCNdata/3611817550_precipitation.xlsx',header = None)
        precipitation = np.mat(precipitation)
        precipitation_max = np.max(np.max(precipitation))
        if precipitation_max == 0:
            print("被除数为0，无需归一化")
            precipitation_nor = precipitation
            precipitation_nor_train = precipitation_nor[0:train_size]
            precipitation_nor_test = precipitation_nor[train_size:time_len]
        else:
            #precipitation_nor = precipitation/precipitation_max# 归一化
            precipitation_nor = precipitation
            precipitation_nor_train = precipitation_nor[0:train_size]
            precipitation_nor_test = precipitation_nor[train_size:time_len]


        if scheme == 1:#add poi(dim+1)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],direction_nor[:1]))# a1的0-seq_len行和direction_nor的第一行
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],direction_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        elif scheme == 2:#add beach_width(dim+11)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = beach_width_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)# 考虑输入长度的训练数据+（动态属性的对应于训练长度数据+预测用动态属性）
                trainY.append(a1[seq_len : seq_len + pre_len])# 预测长度的训练数据
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = beach_width_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        elif scheme == 3:#Reservoir_water_level
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = Reservoir_water_level_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)# 考虑输入长度的训练数据+（动态属性的对应于训练长度数据+预测用动态属性）
                trainY.append(a1[seq_len : seq_len + pre_len])# 预测长度的训练数据
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = Reservoir_water_level_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        elif scheme == 4:#phreatic_line
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = phreatic_line_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)# 考虑输入长度的训练数据+（动态属性的对应于训练长度数据+预测用动态属性）
                trainY.append(a1[seq_len : seq_len + pre_len])# 预测长度的训练数据
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = phreatic_line_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        elif scheme == 5:#precipitation
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = precipitation_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)# 考虑输入长度的训练数据+（动态属性的对应于训练长度数据+预测用动态属性）
                trainY.append(a1[seq_len : seq_len + pre_len])# 预测长度的训练数据
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = precipitation_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        elif scheme == 6:#Dynamic attribute
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = beach_width_nor_train[i: i + seq_len + pre_len]
                a3 = Reservoir_water_level_nor_train[i: i + seq_len + pre_len]
                a4 = phreatic_line_nor_train[i: i + seq_len + pre_len]
                a5 = precipitation_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],a3[0: seq_len + pre_len],a4[0: seq_len + pre_len],a5[0: seq_len + pre_len]))
                trainX.append(a)# 考虑输入长度的训练数据+（动态属性的对应于训练长度数据+预测用动态属性）
                trainY.append(a1[seq_len : seq_len + pre_len])# 预测长度的训练数据
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = beach_width_nor_test[i: i + seq_len + pre_len]
                b3 = Reservoir_water_level_nor_test[i: i + seq_len + pre_len]
                b4 = phreatic_line_nor_test[i: i + seq_len + pre_len]
                b5 = precipitation_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],b3[0: seq_len + pre_len],b4[0: seq_len + pre_len],b5[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        else:#add kg(dim+12)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = beach_width_nor_train[i: i + seq_len + pre_len]
                a3 = Reservoir_water_level_nor_train[i: i + seq_len + pre_len]
                a4 = phreatic_line_nor_train[i: i + seq_len + pre_len]
                a5 = precipitation_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],a3[0: seq_len + pre_len],a4[0: seq_len + pre_len],a5[0: seq_len + pre_len],direction_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = beach_width_nor_test[i: i + seq_len + pre_len]
                b3 = Reservoir_water_level_nor_test[i: i + seq_len + pre_len]
                b4 = phreatic_line_nor_test[i: i + seq_len + pre_len]
                b5 = precipitation_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],b3[0: seq_len + pre_len],b4[0: seq_len + pre_len],b5[0: seq_len + pre_len],direction_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])


    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)

    testY1 = np.array(testY)
    print(trainX1.shape)
    print(trainY1.shape)
    print(testX1.shape)
    print(testY1.shape)
    
    return trainX1, trainY1, testX1, testY1
