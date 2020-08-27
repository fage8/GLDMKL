#!/usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
import k_means
import numpy.linalg as LA
from k_means import *
import numpy as np
from sklearn.cluster import KMeans
import sklearn


# def load_data(path):
#     path = path
#     f = np.load(path)
#     x_train, y_train = f['x_train'], f['y_train']
#     x_test, y_test = f['x_test'], f['y_test']
#     f.close()
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
#     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
#     y_train = y_train.reshape(y_train.shape[0], 1)
#     y_test = y_test.reshape(y_test.shape[0], 1)
#
#     return (x_train, y_train), (x_test, y_test)

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    labels = []
    dataList = listdir(dirName)
    m = len(dataList)
    dataMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = dataList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 6:
            labels.append(-1)
        else:
            labels.append(1)
        dataMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return dataMat, labels

def loadDataSet(fileName): # 读取数据
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        # .strip(m)  # 删除s开头、结尾的rm
        lineArr = line.strip().split(',')  # 这一行预处理需要根据实际数据集的情况进行改变
        # dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 数据集只有三个子元素，前两子元素作为数据特征
        # labelMat.append(float(lineArr[2]))  # 第三个子元素作为数据类别
        d = len(lineArr)  # 输出子元素个数，列表没有行列的概念，就相当于一维数组
        fltLine = list(map(float, lineArr[0:d - 1]))  # [0, d-1)
        dataMat.append(fltLine)
        # print(lineArr[-1])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat  # 返回数据特征和数据类别

# def loadDataSet2(fileName1, fileName2): # 读取数据
#     dataMat = []
#     labelMat = []
#     fr1 = open(fileName1)
#     fr2 = open(fileName2)
#     for line1 in fr1.readlines():
#         # .strip(m)  # 删除s开头、结尾的rm
#         lineArr1 = line1.strip().split(' ')  # 这一行预处理需要根据实际数据集的情况进行改变
#         # dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 数据集只有三个子元素，前两子元素作为数据特征
#         # labelMat.append(float(lineArr[2]))  # 第三个子元素作为数据类别
#         d1 = len(lineArr1)  # 输出子元素个数，列表没有行列的概念，就相当于一维数组
#         fltLine1 = list(map(float, lineArr1[0:d1 - 1]))  # [0, d-1)
#         dataMat.append(fltLine1)
#         # print(lineArr[-1])
#         lineArr1[-1] = 1
#         labelMat.append(float(lineArr1[-1]))
#     for line2 in fr2.readlines():
#         lineArr2 = line2.strip().split(' ')  # 这一行预处理需要根据实际数据集的情况进行改变
#         # dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 数据集只有三个子元素，前两子元素作为数据特征
#         # labelMat.append(float(lineArr[2]))  # 第三个子元素作为数据类别
#         d2 = len(lineArr2)  # 输出子元素个数，列表没有行列的概念，就相当于一维数组
#         fltLine2 = list(map(float, lineArr2[0:d2 - 1]))  # [0, d-1)
#         dataMat.append(fltLine2)
#         # print(lineArr[-1])
#         lineArr2[-1] = -1
#         labelMat.append(float(lineArr2[-1]))
#     return dataMat, labelMat  # 返回数据特征和数据类别

def selectJrand(i, n):  # 在0-n中随机选择一个不是i的整数，样本个数n也是所有alpha的数目，选择第二个alpha
    j = i
    while (j == i):  # j不等于i才退出循环
        j = int(random.uniform(0, n))
    return j

def clipAlpha(aj, H, L):  # 对于优化后的aj进行剪辑，保证aj在L和H范围内（L <= aj <= H）
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup):  # 核函数，X:支持向量的特征或者分组后的数据特征；A：某样本特征数据；kTup：('lin',k1)核函数的类型和参数，深度操作可以在这里进行
    n, d = shape(X)
    K = mat(zeros((n, 1)))  # 将核函数放到核矩阵的相应位置序号为i列中，返回的就是n行1列的数据
    if kTup[0] == 'lin':  # 线性函数
        K = X * A.T
    elif kTup[0] == 'poly':  # 多项式函数
        K = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
    elif kTup[0] == 'poly2':  # 两层poly函数
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        K = pow(np.array(float(kTup[1]) * K1 + float(kTup[2])), float(kTup[3]))
    elif kTup[0] == 'poly3':  # 三层poly函数
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        K2 = pow(np.array(float(kTup[1]) * K1 + float(kTup[2])), float(kTup[3]))
        K = pow(np.array(float(kTup[1]) * K2 + float(kTup[2])), float(kTup[3]))
    elif kTup[0] == 'polymkl':  # 两层poly深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K = pow(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])), float(kTup[3]))
    elif kTup[0] == 'polymkl2':  # 三层poly深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K21 = pow(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])), float(kTup[3]))
        K = sqrt(abs(2 - 2 * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)))
        K22 = exp(K / (-1 * kTup[1]))
        K23 = np.tanh(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])))
        K = 1 - (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)
        K24 = exp(K / (-1 * kTup[1] ** 2))  # 此处会发生溢出

        K = pow(np.array(float(kTup[1]) * (0.15 * K21 + 0.35 * K22 + 0.15 * K23 + 0.35 * K24) + float(kTup[2])), float(kTup[3]))
    elif kTup[0] == 'rbf':  # 径向基函数(radial bias function)
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K = exp(K / (-2 * kTup[1] ** 2))  # '/'表示对矩阵元素展开计算，返回生成的结果
    elif kTup[0] == 'rbf2':  # 两层rbf核函数
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K1 = 1 - exp(K / (-2 * kTup[1] ** 2))
        K = exp(K1 / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'rbf3':  # 三层rbf核函数
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K1 = 1 - exp(K / (-2 * kTup[1] ** 2))
        K2 = 1 - exp(K1 / (-1 * kTup[1] ** 2))
        K = exp(K2 / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'rbf4':  # 四层rbf核函数
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K1 = 1 - exp(K / (-2 * kTup[1] ** 2))
        K2 = 1 - exp(K1 / (-1 * kTup[1] ** 2))
        K3 = 1 - exp(K2 / (-1 * kTup[1] ** 2))
        K = exp(K3 / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'rbf5':  # 五层rbf核函数
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K1 = 1 - exp(K / (-2 * kTup[1] ** 2))
        K2 = 1 - exp(K1 / (-1 * kTup[1] ** 2))
        K3 = 1 - exp(K2 / (-1 * kTup[1] ** 2))
        K4 = 1 - exp(K3 / (-1 * kTup[1] ** 2))
        K = exp(K4 / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'rbfmkl':  # 两层rbf深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K = 1 - (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)
        K = exp(K / (-1 * kTup[1] ** 2))  # 此处会发生溢出
    elif kTup[0] == 'rbfmkl2':  # 三层rbf深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K21 = pow(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])), float(kTup[3]))
        K = sqrt(abs(2 - 2 * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)))
        K22 = exp(K / (-1 * kTup[1]))
        K23 = np.tanh(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])))
        K = 1 - (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)
        K24 = exp(K / (-1 * kTup[1] ** 2))

        K = 1 - (0.15 * K21 + 0.35 * K22 + 0.15 * K23 + 0.35 * K24)
        K = exp(K / (-1 * kTup[1] ** 2))
    elif kTup[0] == 'tan':
        K = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
    elif kTup[0] == 'tan2':  # 两层tan核函数
        K1 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        K = np.tanh(np.array(float(kTup[1]) * K1 + float(kTup[2])))
    elif kTup[0] == 'tan3':  # 三层tan核函数
        K1 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        K2 = np.tanh(np.array(float(kTup[1]) * K1 + float(kTup[2])))
        K = np.tanh(np.array(float(kTup[1]) * K2 + float(kTup[2])))
    elif kTup[0] == 'tanmkl':  # 两层tan深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K = np.tanh(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])))
    elif kTup[0] == 'tanmkl2':  # 三层tan深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K21 = pow(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])), float(kTup[3]))
        K = sqrt(abs(2 - 2 * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)))
        K22 = exp(K / (-1 * kTup[1]))
        K23 = np.tanh(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])))
        K = 1 - (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)
        K24 = exp(K / (-1 * kTup[1] ** 2))  # 此处会发生溢出

        K = np.tanh(np.array(float(kTup[1]) * (0.15 * K21 + 0.35 * K22 + 0.15 * K23 + 0.35 * K24) + float(kTup[2])))
    elif kTup[0] == 'lap':
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K = exp(K / (-1 * kTup[1]))
    elif kTup[0] == 'lap2':  # 两层lap核函数
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K1 = sqrt(2 - 2 * exp(K / (-1 * kTup[1])))
        K = exp(K1 / (-1 * kTup[1]))
    elif kTup[0] == 'lap3':  # 三层lap核函数
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K1 = sqrt(2 - 2 * exp(K / (-1 * kTup[1])))
        K2 = sqrt(2 - 2 * exp(K1 / (-1 * kTup[1])))
        K = exp(K2 / (-1 * kTup[1]))
    elif kTup[0] == 'lapmkl':  # 两层lap深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))  # 返回数组
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))  # 返回矩阵
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))  # 返回数组
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))  # 返回矩阵
        # 数字可以和矩阵相加减，数组和矩阵也可以运算，注意根号要注意带绝对值
        K = sqrt(abs(2 - 2 * 0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4))
        K = exp(K / (-1 * kTup[1]))
    elif kTup[0] == 'lapmkl2':  # 三层lap深度多核学习
        K1 = pow(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])), float(kTup[3]))
        for j in range(n):
            K[j] = sqrt(X[j, :] * X[j, :].T + A * A.T - 2 * X[j, :] * A.T)  # # xj-xi
        K2 = exp(K / (-1 * kTup[1]))
        K3 = np.tanh(np.array(float(kTup[1]) * (X * A.T) + float(kTup[2])))
        for j in range(n):
            deltaRow = X[j, :] - A  # xj-xi
            K[j] = deltaRow * deltaRow.T  # (xj-xi)T(xj-xi)
        K4 = exp(K / (-2 * kTup[1] ** 2))
        K21 = pow(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])), float(kTup[3]))
        K = sqrt(abs(2 - 2 * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)))
        K22 = exp(K / (-1 * kTup[1]))
        K23 = np.tanh(np.array(float(kTup[1]) * (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4) + float(kTup[2])))
        K = 1 - (0.15 * K1 + 0.35 * K2 + 0.15 * K3 + 0.35 * K4)
        K24 = exp(K / (-1 * kTup[1] ** 2))

        K = sqrt(abs(2 - 2 * (0.15 * K21 + 0.35 * K22 + 0.15 * K23 + 0.35 * K24)))
        K = exp(K / (-1 * kTup[1]))  # 此处会发生溢出
    # Arc-cos反余弦核函数
    elif kTup[0] == 'Arcos':
        if kTup[2] == 'L0':
            Theta = np.arccos((np.matmul(X, A.T)) / (LA.norm(X, 1, axis=1, keepdims=True) * LA.norm(A, 1, axis=1)))
        elif kTup[2] == 'L1':
            Theta = np.arccos((np.matmul(X, A.T)) / (LA.norm(X, 2, axis=1, keepdims=True) * LA.norm(A, 2, axis=1)))
            # Debug
            # print(LA.norm(X, 2, axis=1, keepdims=True) * LA.norm(A, 2, axis=1))
            # print(np.matmul(X,A.T).shape)
            # print((LA.norm(X, 2, axis=1, keepdims=True) * LA.norm(A, 2, axis=1)))
            # print(np.matmul(X, A.T) / (LA.norm(X, 2, axis=1, keepdims=True) * LA.norm(A, 2, axis=1)))
            # print(LA.norm(X, 2, axis=0))
            # print(A)
            # print(LA.norm(A, 2, axis=1))
            # print(LA.norm(X, 2, axis=1, keepdims=True))
            # print(Theta)
        elif kTup[2] == 'L2':
            Theta = np.arccos(
                (np.matmul(X, A.T)) / (LA.norm(X, np.inf, axis=1, keepdims=True) * LA.norm(A, np.inf, axis=1)))
        J = J_solution(Theta, kTup)
        if kTup[2] == 'L0':
            K = (1 / np.pi) * LA.norm(X, 1, axis=1, keepdims=True) * LA.norm(A, 1, axis=1) * J
        elif kTup[2] == 'L1':
            # Debug
            # print(1/np.pi)
            # print((LA.norm(X, 2,axis=1,keepdims=True)  * LA.norm(A,2,axis=1)).shape)
            K = (1 / np.pi) * LA.norm(X, 2, axis=1, keepdims=True) * LA.norm(A, 2, axis=1) * J
        elif kTup[2] == 'L2':
            K = (1 / np.pi) * LA.norm(X, np.inf, axis=1, keepdims=True) * LA.norm(A, np.inf, axis=1) * J
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K  # 返回的就是n行1列的数据



# def randGradientAscent(oS, kTup):  # 反向传播和梯度上升优化betas
#     # dataMatrix: 数据矩阵，classLabels: 标签矩阵，numIter: 外循环次数，返回权重数组与回归系数矩阵
#     n, d = np.shape(oS.X)  # n为样本数，d为维数
#     for g in range(4):
#         for j in range(oS.m):  # 遍历第l层第g组所有基础内核的梯度计算公式
#             for i in range(n):
#                 etaS = 0.001
#                 # Ei = calcEk(oS, i)
#                 # E = 1/(2*n) * Ei**2
#                 # 第l层第g分组第m个核函数的梯度计算公式
#                 error = 1/n * oS.eCache[:, 1].T * multiply(oS.alphas, oS.labelMat) * kernelTrans(oS.X, oS.X[i, :], kTup)
#                 oS.betas = oS.betas - etaS * error
#     return oS.betas


# #多核k-means聚类算法，先不知道类别
# def selectEtaD(oS, k):  # 通过居中内核对齐计算距离权重eta
#     etaD = randCent(oS.dataSet, k)
#     return etaD
#
# # 计算k-meansu多核距离
# def distEclud(oS, kTup):
#     etaD = selectEtaD()
#     for i in range(oS.n):
#         oS.distJI[:, i] = multiply(etaD, kernelTrans(oS.X, oS.X[i, :], kTup))  # 返回的是n行1列的距离向量
#     return oS.distJI  # 返回距离矩阵
#
# # 构建聚簇中心，取k个(此例中为4)随机质心，在数据集每一维的最小和最大值之间的
# def randCent(dataSet, k):
#     d = shape(dataSet)[1]
#     centroids = mat(zeros((k, d)))  # 每个质心有d个坐标值，总共要k个质心
#     for j in range(d):
#         minJ = min(dataSet[:, j])
#         maxJ = max(dataSet[:, j])
#         rangeJ = float(maxJ - minJ)
#         centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
#     return centroids
#
# # k-means 聚类算法
# def kMeans(oS, k, distMeans=distEclud, createCent=randCent):
#     n = shape(oS.dataSet)[0]
#     clusterAssment = mat(zeros((n, 2)))  # 用于存放该样本属于哪类及质心距离
#     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
#     centroids = createCent(oS.dataSet, k)
#     clusterChanged = True  # 用来判断聚类是否已经收敛
#     while clusterChanged:
#         clusterChanged = False
#         for i in range(n):  # 把每一个数据点划分到离它最近的中心点
#             minDist = inf
#             minIndex = -1
#             for j in range(k):  # 寻找最近的质心
#                 distJI = distMeans(centroids[j, :], oS.dataSet[i, :])
#                 if distJI < minDist:
#                     minDist = distJI
#                     minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
#             if clusterAssment[i, 0] != minIndex:
#                 clusterChanged = True  # 如果分配发生变化，则需要继续迭代
#             clusterAssment[i, :] = minIndex, minDist ** 2  # 并将第i个数据点的分配情况存入字典
#         print(centroids)
#         for cent in range(k):  # 重新计算中心点，更新中心点位置
#             ptsInClust = oS.dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 去第一列等于cent的所有列
#             centroids[cent, :] = mean(ptsInClust, axis=0)  # 算出这些数据的中心点
#     return centroids, clusterAssment
#
# def multipleKernel(oS):  # 多核融合成一个大核
#     for i in range(oS.n):
#         mK = oS.mK[:, i]
#         for j in range(oS.m):  # m指的是基本内核的个数
#             oS.mK[:, i] = kernelTrans(oS.X, oS.X[i, :], oS.kTup) * oS.betas[j]  # 这里的内核组合应该是一个个的样本来处理，只是写的时候写成一个数据集和单个样本处理
#             mK += oS.mK
#         return mK  # 每个组返回一个组合内核,该组合内核是针对单样本的
#
# def stoGradAscent(oS, kTup, numIter=150):  # 门模型随机梯度上升算法
#     n, d = shape(oS.X)
#     weights = ones(d)
#     myCentroids, clustAssing = kMeans(oS.X, 4)  # 多核k-means聚类分成组
#     G = clustAssing[0]
#     s = 0
#     J = 0
#     for j in range(numIter):
#         for i in range(n):
#             dataIndex = range(n)
#             a_t = 4 / (1.0 + j + i) + 0.01  # 步长每次迭代时需要调整
#             randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取更新
#             for j in range(d):
#                 for i in range(n):
#                     s[j] = 1/2 * (oS.alphas[i] * oS.labelMat[i]) * multiply(oS.alphas, oS.labelMat).T * kernelTrans(oS.X, oS.X[i, :], kTup)
#                 for g in range(G):
#                     betas = selectBeta(oS)
#                     J += betas[j] * s[j]
#             error = oS.labelMat[randIndex] - J
#             weights = weights + a_t * error * oS.X[randIndex]
#             del(dataIndex[randIndex])
#     return weights
#
#
# def selectBeta(oS, kTup):  # 门模型优化beta
#     myCentroids, clustAssing = kMeans(oS.X, 4)  # 多核k-means聚类分成组
#     n_g = myCentroids  # n_g表示第g组的样本数
#     G = clustAssing[0]
#     expSum = 0
#     a = b = v = 0
#     p = int(random.uniform(0, 1))
#     for g in range(G):
#         for j in range(oS.m):  # m为内核数
#             for i in range(oS.n):  # n为样本数
#                 kY = 1/ n_g * kernelTrans(oS.X, oS.X[i, :], kTup)
#                 kK = 1/ power(n_g,4) * kernelTrans(oS.X, oS.X[i, :], kTup) * kernelTrans(oS.X, oS.X[i, :], kTup)
#                 yY = n_g * n_g
#                 v = kY / sqrt(kK * yY)
#                 expSum += exp(p(a * v + b))
#                 betas = exp(a * v + b) / power(expSum, 1/p)
#         return betas
#
#
# def iteration(X, A, n, l, kTup):
#     n, d = X.shape
#     iter = 0
#     Kl_ii = mat(zeros((n, 1)))
#     Kl_ij = kernelTrans(X, A, kTup)
#     # print(Kl_ij.shape, "........")
#     for i in range(n):
#         if(kTup[0] == 'Arcos'):  # 反余弦核比较特殊
#             Kl_ii[i] = kernelTrans(X[i, :], X[i, :], kTup)
#     Kl_jj = kernelTrans(A, A, kTup)  # A=X[i, :]
#     # print(Kl_ii.shape, "........")
#     # print(Kl_ii.shape)
#     # print(Kl_jj.shape)
#     while(iter < round(int(l))):  # round(number)四舍五入到最接近的整数,l或许为最大准确率不变的层数
#         Kl_ij = Deep_kernel(Kl_ij, Kl_ii, Kl_jj, kTup)
#         Kl_ii = Deep_kernel(Kl_ii, Kl_ii, Kl_ii, kTup)
#         Kl_jj = Deep_kernel(Kl_jj, Kl_jj, Kl_jj, kTup)
#         # print(Kl_ij)
#         # print(Kl_ii)
#         # print(Kl_jj)
#         iter = iter + 1
#
# return Kl_ij  # k(xi,xj) = Φ(xi)*Φ(xj)

# 定义类，方便存储数据，用于清理代码的数据结构
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup, m):  # 存储各类参数
        self.X = mat(dataMatIn)  # 数据特征矩阵，n行d列，mat函数是将数组（列表）转化成矩阵，transpose对换数组的维度
        self.labelMat = mat(classLabels).transpose()  # 数据类别特征，n行1列
        self.C = C  # 惩罚参数C，参数越大，非线性拟合能力越强，这个参数感觉可以自己调整一下
        self.tol = toler  # 停止阀值
        self.kTup = kTup  # 核函数参数
        self.m = m  # m为内核数
        self.n, self.d = shape(dataMatIn)  # 数据行数，也就是样本个数; 数据列数，也就是特征个数
        self.alphas = mat(zeros((self.n, 1)))
        # 以下beta初始化部分可以适当修改
        self.betas = [0] * self.m  # 初始化β为m行1列的零数组
        for i in range(self.m):
            self.betas[i] = float(1 / self.m)
        # sum = 0
        # while(sum != 1):
        #     for i in range(self.m):  # beta个数就是内核个数
        #         self.betas[i] = random.rand(0, 1)  # 将βi赋值为1/m
        #         sum += self.betas[i]
        self.b = 0  # 初始设为0
        self.eCache = mat(zeros((self.n, 2)))  # 初始化缓存为n行2列的零矩阵
        self.K = mat(zeros((self.n, self.n)))  # 初始化n行n列的基础核矩阵，核函数的计算结果
        self.mK = mat(zeros((self.n, self.n)))  # 初始化n行n列的多核矩阵
        self.distJI = mat(zeros((self.n, self.n)))  # 初始化n行n列的k-means距离矩阵
        # for i in range(self.n):  # 注意i是在[0，n-1）范围内
        #     # if(kTup[0] == 'Arcos'):
        #     #     self.K[:, i] = iteration(self.X, self.X[i, :], kTup[3], l, kTup)  # iteration(X, A, d, l, kTup)
        #     # else:
        #         self.K[:, i] = kernelTrans(self.X, self.X[i, :], self.kTup)  # x，xi，核函数类型及参数，将核函数放到核矩阵的相应位置序号为i列中
        for i in range(self.n):
            for j in range(self.m):  # m指的是基本内核的个数, 不同类型核函数
                self.mK[:, i] += kernelTrans(self.X, self.X[i, :], self.kTup) * self.betas[j]  # 这里的内核组合应该是一个个的样本来处理，只是写的时候写成一个数据集和单个样本处理


def calcEk(oS, k):  # 计算误差数值Ek（参考《统计学习方法》p127公式7.105）
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.mK[:, k] + oS.b)  # 序号为k的样本的核函数是放在和矩阵的序号k的列
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 内循环中的启发式方法，最大化步长选择第二个alpha_j，并返回其Ej值
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 缓存矩阵第一列为eCache是否有效的标志位，第二列为Ei
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 返回矩阵中的非零Ei的行号所对应的alpha，A代表取矩阵中的对应元素
    if (len(validEcacheList)) > 1:  # 有不为0的误差Ei，得继续优化使得步长最大
        for k in validEcacheList:  # 遍历不为0的误差Ei序号
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)  # 步长
            if (deltaE > maxDeltaE):  # 返回步长最大的aj
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # 误差Ei均为0
        j = selectJrand(i, oS.n)  # 随机选择第二个alpha
        Ej = calcEk(oS, j)
        return j, Ej


def updateEk(oS, k):  # 更新os误差值存入缓存中，在对alpha进行优化之后会用到这个值
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


# 如果两个选择的两个输入向量的X相同，那么eta为0，此时要另外选择一个j
def secondChoiceJ(oS, i):
    # 首先先从非边界元素中寻找是否eta不为零的j
    nonBounds = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]
    n = len(nonBounds)
    st = int(random.uniform(0, n))  # 随机选择起始位置
    for i in range(n):
        j = nonBounds[(st+i) % n]
        if (j == i):
            continue
        kIi = oS.mK[i, i]
        kIj = oS.mK[i, j]
        kJj = oS.mK[j, j]
        eta = kIi + kJj - 2*kIj
        if (eta > 0):
            return j
    # 如果非边界找不到，那么到边界上找j
    bounds = nonzero((oS.alphas.A == 0) + (oS.alphas.A == oS.C))[0]
    n = len(bounds)
    st = int(random.uniform(0, n))
    for i in range(n):
        j = bounds[(st+i) % n]
        if (j == i):
            continue
        kIi = oS.mK[i, i]
        kIj = oS.mK[i, j]
        kJj = oS.mK[j, j]
        eta = kIi + kJj - 2*kIj
        if (eta > 0):
            return j
    return -1


# 首先检验alpha_i是否满足KKT条件，如果不满足，随机选择alpha_j进行优化，更新alpha_i, alpha_j, b值
def innerL(i, oS):  # 输入参数i和所有参数数据
    Ei = calcEk(oS, i)  # 计算误差Ei值
    # 如果数据误差很大，不符合KKT条件，需要优化一对alpha 参考《统计学习方法》p128公式7.111-113
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 选取步长最大的alpha_j，并返回其误差Ej值
        alphaIold = oS.alphas[i].copy()  # 将第一个alpha和第二个alpha备份一下
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):  # 确定剪辑上下界，同时也确保alpha在0到C之间，以下代码的公式参考《统计学习方法》p126
            L = max(0, alphaJold - alphaIold)  # 下界大于或等于0
            H = min(oS.C, oS.C + alphaJold - alphaIold)  # 上界小于或等于C
        else:  # yi=yj
            L = max(0, alphaJold + alphaIold - oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        if L == H:  # 如果上界等于下界，直接轮到下一个i
            print("L==H")
            return 0
        eta = oS.mK[i, i] + oS.mK[j, j] - 2.0 * oS.mK[i, j]   # eta是alpha_j的最优修改量，必须是大于0，参考《统计学习方法》p127公式7.107
        if eta <= 0:  # 如果eta非正，那么需要重新选取一个j
            print("eta<=0")
            # return 0
            # 下面是重新选择j，使得eta为正
            j = secondChoiceJ(oS, i)
            if j < 0:  # 如果还是找不到eta为正的j，直接轮到下一个i
                return 0
            # 此时需要重新计算j的信息
            alphaJold = oS.alphas[j].copy()
            if (oS.labelMat[i] != oS.labelMat[j]):  # 确保alpha在0到C之间，以下代码的公式参考《统计学习方法》p126
                L = max(0, alphaJold - alphaIold)  # 下界大于或等于0
                H = min(oS.C, oS.C + alphaJold - alphaIold)  # 上界小于或等于C
            else:  # yi=yj
                L = max(0, alphaJold + alphaIold - oS.C)
                H = min(oS.C, alphaJold + alphaIold)
            if L == H:  # 如果上界等于下界
                print("L==H")
                return 0
            eta = oS.mK[i, i] + oS.mK[j, j] - 2.0 * oS.mK[i, j]  # eta是alpha_j的最优修改量，必须是大于0，参考《统计学习方法》p127公式7.107

        aJ = alphaJold + oS.labelMat[j] * (Ei - Ej) / eta  # alpha_j根据eta来优化，选择步长或者说Ei-Ej最大的alpha_j，参考《统计学习方法》p127公式7.106
        oS.alphas[j] = clipAlpha(aJ, H, L)  # 确保新的alpha_j在L到H之间，这是剪辑后的解，参考《统计学习方法》p127公式7.108
        updateEk(oS, j)  # alpha_j更新后，更新误差Ej，存入缓存中
        if (abs(oS.alphas[j] - alphaJold) < oS.tol):  # 前后alpha变化大小阀值（自己设定，也可以和tol一样）
            print("j not moving enough")  # 说明alpha_j前后变化太小，直接轮到下一个i
            return 0
        # 对i进行修改，修改量与j相同，但方向相反，得到新的alpha_i，参考《统计学习方法》p127公式7.109
        oS.alphas[i] = alphaIold + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # alpha_i更新后，更新误差Ei，存入缓存中
        # 以下求解b的过程，参考《统计学习方法》p129公式7.114-7.116
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.mK[i, i] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.mK[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.mK[i, j] - oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.mK[j, j]
        if (0 < oS.alphas[i] < oS.C):  # 更新后的alpha_i在0到C之间
            oS.b = b1
        elif (0 < oS.alphas[j] < oS.C):  # 更新后的alpha_j在0到C之间
            oS.b = b2
        else:  # alpha_i和alpha_j是0或C
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


# SMO函数，用于快速求解出alpha和b
def smoP(oS, maxIter):  # 数据特征，数据类别，参数C，阀值toler，最大迭代次数，核函数（默认线性核）
    iter = 0
    entireSet = True
    alphaPairsChanged = 0  # 记录alpha对是否已经进行了优化
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 当迭代次数超过指定的最大值，或者遍历整个集合都未对任意alpha对进行修改，就退出循环
        alphaPairsChanged = 0  # 有的代码里写的是==，我感觉不对劲
        if entireSet:
            for i in range(oS.n):  # 遍历整个数据集，只遍历一次
                alphaPairsChanged += innerL(i, oS)  # 选择第二个alpha，并在可能对其进行优化处理，如果有任意一对alpha值发生改变，那么返回1
                # 显示第多少次迭代，那行特征数据使alpha发生了改变，这次改变了多少次alpha
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1  # 遍历完一次集合，iter+1，一次迭代就是一次循环
        else:  # 遍历完整个集合，跳转到这里
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]  # A代表取矩阵中的对应元素
            for i in nonBoundIs:  # 遍历非边界值，alpha在0到C之间，也是间隔边界上的支持向量点，一般会遍历多次
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1  # 遍历完一次非边界值，iter+1，下一次entireSet还是false，则继续遍历非边界值
        if entireSet:  # 遍历完整个集合后，接着遍历非边界值
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果遍历完非边界值alpha对没有改变，则继续遍历整个集合，若仍没有改变则退出大循环，返回优化后的b和alphas矩阵
            entireSet = True
        print("iteration number: %d" % iter)
    # return oS.b, oS.alphas  # 师兄的没有返回值，估计是这里实行深核学习的



def testRbf(d, lab, test_d, test_lab):  # 输入的参数是数组，这是一层的预测，求准确率
    # 第0组训练模型
    datMat = mat(d)  # 数组转化为n行d列矩阵
    labelMat = mat(lab).transpose()  # 数组转化为n行1列矩阵

    print("number of Candidate kernel :")
    value4 = int(input())  # 候选核个数,必须变成整数
    value = []  # 核函数类型
    value1 = []  # 核函数第一个参数,必须变成整数
    value2 = []  # 核函数第二个参数,必须变成整数
    value3 = []  # 核函数第三个参数,必须变成整数
    # value5 = []  # 反余弦核层数
    for i in range(value4):  # value4为候选核个数
        print("input kernel :")
        value.append(input())  # 输入类似kernel='rbf'
        print("input k : ")
        value1.append(float(input()))  # 输入类似k1=1.3
        print("input c : ")
        value2.append(float(input()))  # 输入类似c=2,1
        print("input n : ")
        value3.append(float(input()))  # 输入类似n=1
        # print("input Arcos layer :")
        # value5.append(input())  # 输入反余弦层数l
        oS = optStruct(d, lab, 200, 0.00000001, (value[i], value1[i], value2[i], value3[i]), value4)  # 数据结构存储，这里是oS的出处,这里就要输入m个基础内核
    # b, alphas = smoP(oS, 10000)  # 训练好的多核单层模型，返回最优的alpha矩阵和b
    smoP(oS, 100)  # 100为最大迭代次数, 模型训练

    # 构造支持向量矩阵
    svInd = nonzero(oS.alphas.A > 0)[0]  # 选取不为0数据的行数（也就是支持向量），alpha>0时，样本数据是为支持向量，A代表取矩阵中的对应元素
    sVs = datMat[svInd]  # 支持向量的特征数据
    vector = np.array(sVs)
    labelSV = labelMat[svInd]  # 支持向量的类别（1或-1）
    print("there are %d Support Vectors" % shape(sVs)[0])  # 打印出共有多少的支持向量，其实就是svInd值

    # 第0组训练集和测试集，以及支持向量可视化
    plt.scatter(d[:, 0], d[:, 1], c="red", marker='o', label='d')
    plt.scatter(test_d[:, 0], test_d[:, 1], c="yellow", marker='+', label='test_d')
    plt.scatter(vector[:, 0], vector[:, 1], c="green", marker='*', label='vector')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()

    n, d = shape(datMat)  # 训练数据特征的行列数
    print(n, d)
    errorCount = 0
    # pred_train_T = []
    # pred_test_T = []

    for i in range(n):  # 预测是一个一个的样本输入进来的
        # kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', 1.3))  # 将支持向量转化为核函数
        multipleKernel = mat(zeros((shape(sVs)[0], 1)))
        for j in range(oS.m):  # m指的是基本内核的个数为5
            mK = oS.betas[j] * kernelTrans(sVs, datMat[i, :], (value[j], value1[j], value2[j], value3[j]))   # 这里的内核组合应该是一个个的样本来处理，只是写的时候写成一个数据集和单个样本处理
            # print(mK)
            multipleKernel += mK
            # print(multipleKernel)
        kernelEval = multipleKernel
        # print("第%d个核函数\n" % i, kernelEval)
        predict = kernelEval.T * multiply(labelSV, oS.alphas[svInd]) + oS.b  # 这一行的预测结果（代码来源于《统计学习方法》p133里面最后用于预测的公式）注意最后确定的分离平面只有那些支持向量决定。
        # pred_train_T.append(predict)
        if sign(predict) != sign(lab[i]):  # sign函数 -1 if x < 0, 0 if x==0, 1 if x > 0
            errorCount += 1
    print("the training accurate rate is: %f" % (1 - (float(errorCount) / n)))  # 打印出准确率


    # 第0组测试模型
    datMat_test = mat(test_d)  # n行d列矩阵
    labelMat_test = mat(test_lab).transpose()  # n行1列矩阵
    n, d = shape(datMat_test)
    print(n, d)
    errorCount_test = 0
    for i in range(n):  # 使用已经训练好的模型的支持向量，alpha和标签，进行预测数据集的准确率
        # kernelEval = kernelTrans(sVs, datMat_test[i, :], ('rbf', 1.3))
        multipleKernel = mat(zeros((shape(sVs)[0], 1)))
        for j in range(oS.m):  # m指的是基本内核的个数为5
            mK = oS.betas[j] * kernelTrans(sVs, datMat_test[i, :], (value[j], value1[j], value2[j], value3[j]))   # 这里的内核组合应该是一个个的样本来处理，只是写的时候写成一个数据集和单个样本处理
            # print(mK)
            multipleKernel += mK
            # print(multipleKernel)
        kernelEval = multipleKernel
        predict = kernelEval.T * multiply(labelSV, oS.alphas[svInd]) + oS.b
        # pred_test_T.append(predict)
        if sign(predict) != sign(test_lab[i]):
            errorCount_test += 1
    print("the test accurate rate is: %f" % (1 - (float(errorCount_test) / n)))
    # predict1_D = mat(pred_train_T).T  # 训练集n行1列，两个n不一样
    # predict2_D = mat(pred_test_T).T  # 测试集n行1列
    # return predict1_D, predict2_D

# 聚类分组程序
def clusting(g):
    liver = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\liver.txt'
    monk2 = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\monk2.txt'
    australian = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\australian.txt'
    sonar = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\sonar.txt'
    breast = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\breast.txt'
    german = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\german.txt'
    data1 = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\data053_1.txt'
    data3 = 'E:\\刘发的程序\\python程序\\svm\\dataSet\\data175_3.txt'

    dataArr, labelArr = loadDataSet(breast)
    # x_train, y_train = loadImages('E:/刘发的程序/python程序/svm/dataSet/trainings')
    # x_test, y_test = loadImages('E:/刘发的程序/python程序/svm/dataSet/tests')
    # (x_train, y_train), (x_test, y_test) = load_data('E:\刘发的程序\python程序\svm\dataSet\mnist.npz')
    # print(y_train)
    # pca = PCA(n_components=2)
    # dataArr = pca.fit_transform(dataArr)
    for i in range(shape(dataArr)[0]):  # dataArr.shape[0]返回的是dataArr这个array的行数，也就是维度
        if labelArr[i] != 1:  # 标签为0，则将其改为-1
            labelArr[i] = -1
    # for i in range(shape(x_train)[0]):
    #     if y_train[i] != 1:  # 标签为其他，则将其改为-1
    #         y_train[i] = -1
    # for i in range(shape(x_test)[0]):
    #     if y_test[i] != 1:  # 标签为其他，则将其改为-1
    #         y_test[i] = -1
    # 将数据集50%为训练集，50%为测试集
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataArr, labelArr, random_state=1,
                                                                                train_size=0.5, test_size=0.5)
    # 训练数据集
    data = np.array(x_train, dtype=np.float64)  # 如果数据集过大请改为flost32读取
    label = np.array(y_train, dtype=np.float64)
    # print("训练集样本特征为", mat(data), shape(mat(data)))
    # print("训练集样本标签为", mat(label), shape(mat(label)))
    # 测试数据集
    testData = np.array(x_test, dtype=np.float64)
    testLabel = np.array(y_test, dtype=np.float64)
    # print("测试集样本特征为", mat(testData), shape(mat(testData)))
    # print("测试集样本标签为", mat(testLabel), shape(mat(testLabel)))

    # 库函数k-means算法，分类效果好些
    estimator = KMeans(n_clusters=g)  # 构造聚类器
    estimator.fit(data)  # 训练集聚类
    label_pred = estimator.labels_  # 获取训练集聚类组号，并不是样本分类标签

    estimator.fit(testData)  # 测试集聚类
    test_label_pred = estimator.labels_  # 获取测试集聚类组号，并不是样本分类标签
    # 自己写k-means存在随机聚类，分类效果差些
    # center, clust = k_means.biKmeans(mat(dataArr), 2)  # 训练集聚类分组
    # d0 = data[nonzero(clust[:, 0].A == 0)[0]]  # 0类标签的样本特征
    # lab0 = label[nonzero(clust[:, 0].A == 0)[0]]
    # test_center, test_clust = k_means.biKmeans(mat(testDataArr), 2)  # 测试集聚类分组
    # test_d0 = testData[nonzero(test_clust[:, 0].A == 0)[0]]  # 0类标签的样本特征
    # test_lab0 = testLabel[nonzero(test_clust[:, 0].A == 0)[0]]
    # 绘制k-means结果
    # 训练数据集分组
    d0 = data[label_pred == 0]
    d1 = data[label_pred == 1]
    d2 = data[label_pred == 2]
    d3 = data[label_pred == 3]
    d4 = data[label_pred == 4]
    d5 = data[label_pred == 5]
    d6 = data[label_pred == 6]
    d7 = data[label_pred == 7]
    d8 = data[label_pred == 8]
    d9 = data[label_pred == 9]
    lab0 = label[label_pred == 0]
    lab1 = label[label_pred == 1]
    lab2 = label[label_pred == 2]
    lab3 = label[label_pred == 3]
    lab4 = label[label_pred == 4]
    lab5 = label[label_pred == 5]
    lab6 = label[label_pred == 6]
    lab7 = label[label_pred == 7]
    lab8 = label[label_pred == 8]
    lab9 = label[label_pred == 9]
    # print("训练集第0组数据特征\n", d0, shape(d0), type(d0))
    # print("训练集第0组数据标签\n", lab0, shape(lab0), type(lab0))
    # print("训练集第1组数据特征\n", d1, shape(d1), type(d1))
    # print("训练集第1组数据标签\n", lab1, shape(lab1), type(lab1))
    # 测试数据集分组
    test_d0 = testData[test_label_pred == 0]
    test_d1 = testData[test_label_pred == 1]
    test_d2 = testData[test_label_pred == 2]
    test_d3 = testData[test_label_pred == 3]
    test_d4 = testData[test_label_pred == 4]
    test_d5 = testData[test_label_pred == 5]
    test_d6 = testData[test_label_pred == 6]
    test_d7 = testData[test_label_pred == 7]
    test_d8 = testData[test_label_pred == 8]
    test_d9 = testData[test_label_pred == 9]
    test_lab0 = testLabel[test_label_pred == 0]
    test_lab1 = testLabel[test_label_pred == 1]
    test_lab2 = testLabel[test_label_pred == 2]
    test_lab3 = testLabel[test_label_pred == 3]
    test_lab4 = testLabel[test_label_pred == 4]
    test_lab5 = testLabel[test_label_pred == 5]
    test_lab6 = testLabel[test_label_pred == 6]
    test_lab7 = testLabel[test_label_pred == 7]
    test_lab8 = testLabel[test_label_pred == 8]
    test_lab9 = testLabel[test_label_pred == 9]
    # print("测试集第0组数据特征\n", test_d0, shape(test_d0), type(test_d0))
    # print("测试集第0组数据标签\n", test_lab0, shape(test_lab0), type(test_lab0))
    # print("测试集第1组数据特征\n", test_d1, shape(test_d1), type(test_d1))
    # print("测试集第1组数据标签\n", test_lab1, shape(test_lab1), type(test_lab1))

    # return data, label, testData, testLabel
    return d0, lab0, test_d0, test_lab0, d1, lab1, test_d1, test_lab1, d2, lab2, test_d2, test_lab2, d3, lab3, test_d3, test_lab3, d4, lab4, test_d4, test_lab4, d5, lab5, test_d5, test_lab5, d6, lab6, test_d6, test_lab6, d7, lab7, test_d7, test_lab7, d8, lab8, test_d8, test_lab8, d9, lab9, test_d9, test_lab9



if __name__=='__main__':
    print("number of groups:")
    g = int(input())
    d0, lab0, test_d0, test_lab0, d1, lab1, test_d1, test_lab1, d2, lab2, test_d2, test_lab2, d3, lab3, test_d3, test_lab3, d4, lab4, test_d4, test_lab4, d5, lab5, test_d5, test_lab5, d6, lab6, test_d6, test_lab6, d7, lab7, test_d7, test_lab7, d8, lab8, test_d8, test_lab8, d9, lab9, test_d9, test_lab9 = clusting(g)  # 学习之前需要进行聚类,主要是2聚类
    print("number of model layers:")
    value7 = input()  # 训练次数
    l = 0
    # 这两层循环就是深核学习了
    while(l < int(value7)):  # 训练次数
        # testRbf(d3, lab3, test_d3, test_lab3)  # 一层的第0组局部多核学习，在这里面进行输入多个内核
        # testRbf(d1, lab1, test_d1, test_lab1)  # 一层的第 组局部多核学习
        testRbf(d0, lab0, test_d0, test_lab0)
        l += 1