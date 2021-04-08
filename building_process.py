import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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
    data_2 = data.copy()
    for col_name in attr:
        dataCol = data_2[col_name]
        data_2[col_name] = dataCol.fillna(dataCol.mode())
    return data_2

def set_missing_Forest(data, attr):

    newData = data.copy()

    for item in attr:
        price_df = newData[['INSPECTION NUMBER', 'PROPERTY GROUP', item]]

        known_price = price_df[price_df[item].notnull()].values
        unknown_price = price_df[price_df[item].isnull()].values
        y = known_price[:, 0]
        X = known_price[:, 1:]
        rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
        rfr.fit(X, y)
        predictedprice = rfr.predict(unknown_price[:, 1:])
        newData.loc[ (newData[item].isnull()), item ] = predictedprice

    return newData


def knn_missing_filled(target, complete_list, data, k = 3, dispersed = True):
    newData = data.copy()
    
    complete_list_temp = complete_list.copy()
    complete_list_temp.insert(0, target)

    data_temp = newData[complete_list_temp]

    x_train = data_temp[data_temp_tdata_tempemp[target].notnull()].values[:,1:]

    y_train = data_temp[data_temp[target].notnull()].values[:,0].reshape(-1,1)
     
    test = data_temp[data_temp[target].isnull()].values[:,1:]

    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(x_train, y_train)

    newData.loc[ (newData[target].isnull()), target ] = clf.predict(test)

    return newData

def process_missing_value(data, complete_list, uncomplete_list):

    data_lost4 = data.copy()

    for i in uncomplete_list:
        data_lost4 = knn_missing_filled(i, complete_list, data_lost4)

    return data_lost4

if __name__=="__main__":
    filename = 'data/winemag-data-130k-v2.csv'
    filename = 'data/building-violations.csv'
    data = openFile(filename)

    numericalAttr = ['INSPECTION NUMBER', 'PROPERTY GROUP', 'Community Areas', 'Census Tracts','Wards','Historical Wards 2003-2015']
    completeList = ['INSPECTION NUMBER', 'PROPERTY GROUP']
    uncompleteList = ['Community Areas', 'Census Tracts','Wards','Historical Wards 2003-2015']

    nominalAttr=['VIOLATION LAST MODIFIED DATE', 'VIOLATION DATE',
       'VIOLATION CODE', 'VIOLATION STATUS', 'VIOLATION STATUS DATE',
       'VIOLATION DESCRIPTION', 'VIOLATION LOCATION',
       'VIOLATION INSPECTOR COMMENTS', 'VIOLATION ORDINANCE', 'INSPECTOR ID', 'INSPECTION STATUS', 'INSPECTION WAIVED',
       'INSPECTION CATEGORY', 'DEPARTMENT BUREAU', 'ADDRESS', 'STREET NUMBER',
       'STREET DIRECTION', 'STREET NAME', 'STREET TYPE', 
       'SSA', 'LATITUDE', 'LONGITUDE', 'LOCATION',
       'Zip Codes', 'Boundaries - ZIP Codes']

    for item in nominalAttr:
        frequencyDict = checkFrequency150k(data, item)
        writeDict("nominal/frequency_"+item+".txt", frequencyDict, item+':\n')
    
    for item in numericalAttr:
        numbersDict = checkFiveNumbers(data, item)
        writeDict("numberical/fiveNumbers_"+item+".txt", numbersDict, item+':\n')

    naDict = checkMissingValues(data)
    writeDict("missingValues.txt", naDict, "Missing Values:")
    
    for item in numericalAttr:
        drawHistogram(data, item, "numberical/histogram_"+item+".jpg")
        drawBox(data, item, "numberical/box_"+item+".jpg")

    #数据缺失的处理
    #1) 将缺失部分剔除
    deleteData = deleteNa(data)
    for item in uncompleteList:
        drawHistogram(deleteData, item, "delete/histogram_new_"+item+".jpg")
        drawBox(deleteData, item, "delete/box_new_"+item+".jpg")

    #2) 用最高频率值来填补缺失值
    modData = fillMod(data, uncompleteList)
    for item in uncompleteList:
        drawHistogram(modData, item, "mod/histogram_new_"+item+".jpg")
        drawBox(modData, item, "mod/box_new_"+item+".jpg")

    #3) 通过属性的相关关系来填补缺失值
    relatedData = set_missing_Forest(data, uncompleteList)
    for item in uncompleteList:
        drawHistogram(relatedData, item, "related/histogram_new_"+item+".jpg")
        drawBox(relatedData, item, "related/box_new_"+item+".jpg")

    #4) 通过数据对象之间的相似性来填补缺失值
    similarityData = process_missing_value(data, completeList, uncompleteList)
    for item in uncompleteList:
        drawHistogram(similarityData, item,'similarity/histogram_new_'+item+'.jpg')
        drawBox(similarityData, item,'similarity/box_new_'+item+'.jpg')