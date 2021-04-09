import json
import os
import sklearn.ensemble

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#打开文件
def openFile(filename):
    data = pd.read_csv(filename)
    return data

#往txt文件中写入字典
def writeDict(filename, dictName, title):
    with open(filename,"w", encoding="utf-8") as f:
        f.write(title)
        f.write(str(dictName))

#缺失值
def checkMissingValues(data):
    na = data.isna().sum()
    return na.to_dict()

#一些基本信息
def showInformation(data):
    na = checkMissingValues(data)
    print("\n How many rows we have? \n {}".format(data.shape[0]))
    print("\n How many columns we have? \n {}".format(data.shape[1]))
    print(f"\n Is there any missing values? \n {na}")

#频数统计
def checkFrequency150k(data, attr):
    return data[attr].value_counts().to_dict()

#五数概括
def checkFiveNumbers(data, attr):
    fiveNumbers = {}
    fiveNumbers["Min"] = data[attr].min()
    fiveNumbers["Q1"] = data[attr].quantile(q=0.25)
    fiveNumbers["Median"] = data[attr].median()
    fiveNumbers["Q3"] = data[attr].quantile(q=0.75)
    fiveNumbers["Max"] = data[attr].max()
    return fiveNumbers

#直方图
def drawHistogram(data, attr, path):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(data[attr].tolist(), log=True, bins=50, color='blue', alpha=0.7)
    plt.savefig(path)

#盒图
def drawBox(data, attr, path):
    fig = plt.figure()
    newData = deleteNa(data)
    plt.boxplot(newData[attr].tolist(), sym='o', whis=1.5)

    plt.savefig(path)

#清除无效数据
def deleteNa(data):
    newData = data.dropna(inplace=False)
    return newData

#用众数填补数据
def fillMod(data, attr):
    newData = data[attr].fillna(data[attr].mode())
    return newData

def set_missing_prices(data):

    newData = data.copy()
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    price_df = newData[['price', 'points']]
    # 乘客分成已知年龄和未知年龄两部分
    known_price = price_df[price_df.price.notnull()].values
    unknown_price = price_df[price_df.price.isnull()].values
    # y即目标年龄
    y = known_price[:, 0]
    # X即特征属性值
    X = known_price[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = sklearn.ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedprice = rfr.predict(unknown_price[:, 1:])
    # 用得到的预测结果填补原缺失数据
    newData.loc[ (newData.price.isnull()), 'price' ] = predictedprice

    return newData

def knn_missing_filled(data, k = 3, dispersed = True):

    newData = data.copy()
    x_train = newData[newData.price.notnull()]['points'].values.reshape(-1,1)
    y_train = newData[newData.price.notnull()]['price'].values.reshape(-1,1)
    print(len(x_train))
    print(len(y_train))
    test = newData[newData.price.isnull()]['points'].values.reshape(-1,1)

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    newData.loc[ (newData.price.isnull()), 'price' ] = clf.predict(test)

    return newData

if __name__=="__main__":
    #filename = 'data/winemag-data-130k-v2.csv'
    filename = 'data/winemag-data_first150k.csv'
    data = openFile(filename)

    attrList1 = ['country', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery']
    for item in attrList1:
        frequencyDict = checkFrequency150k(data, item)
        writeDict("frequency_"+item+".txt", frequencyDict, item+':\n')
    
    attrList2 = ['points', 'price']
    for item in attrList2:
        numbersDict = checkFiveNumbers(data, item)
        writeDict("fiveNumbers_"+item+".txt", numbersDict, item+':\n')

    naDict = checkMissingValues(data)
    writeDict("missingValues.txt", naDict, "Missing Values:")
    
    for item in attrList2:
        drawHistogram(data, item, "histogram_"+item+".jpg")
        drawBox(data, item, "box_"+item+".jpg")

    #数据缺失的处理
    #1) 将缺失部分剔除
    # deleteData = deleteNa(data)
    # for item in attrList2:
    #     drawHistogram(data, item, "delete/histogram_original_"+item+".jpg")
    #     drawHistogram(deleteData, item, "delete/histogram_new_"+item+".jpg")
    #     drawBox(data, item, "delete/box_original_"+item+".jpg")
    #     drawBox(deleteData, item, "delete/box_new_"+item+".jpg")

    #2) 用最高频率值来填补缺失值
    # for item in attrList2:
    #     modData = fillMod(data, item)
    #     drawHistogram(data, item, "mod/histogram_original_"+item+".jpg")
    #     drawHistogram(deleteData, item, "mod/histogram_new_"+item+".jpg")
    #     drawBox(data, item, "mod/box_original_"+item+".jpg")
    #     drawBox(deleteData, item, "mod/box_new_"+item+".jpg")

    #3) 通过属性的相关关系来填补缺失值
    # relatedData = set_missing_prices(data)
    # for item in attrList2:
    #     drawHistogram(data, item, "related/histogram_original_"+item+".jpg")
    #     drawHistogram(relatedData, item, "related/histogram_new_"+item+".jpg")
    #     drawBox(data, item, "related/box_original_"+item+".jpg")
    #     drawBox(relatedData, item, "related/box_new_"+item+".jpg")

    #4) 通过数据对象之间的相似性来填补缺失值
    kNNData = knn_missing_filled(data)
    for item in attrList2:
        drawHistogram(data, item, "similarity/histogram_original_"+item+".jpg")
        drawHistogram(kNNData, item, "similarity/histogram_new_"+item+".jpg")
        drawBox(data, item, "similarity/box_original_"+item+".jpg")
        drawBox(kNNData, item, "similarity/box_new_"+item+".jpg")