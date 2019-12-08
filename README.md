[https://tianchi.aliyun.com/competition/entrance/231702/rankingList](https://tianchi.aliyun.com/competition/entrance/231702/rankingList)

###一、数据和模型初探
#####1.1 数据预处理
```
# coding=utf-8
import pandas as pd
import numpy as np
from sklearn import preprocessing

df=pd.read_csv(r'/Users/ranmo/Desktop/天池/幸福感/happiness_train_complete.csv',encoding='GB2312',index_col='id')

df = df[df["happiness"]>0]   #原表中幸福度非正的都是错误数据,可以剔除12条错误数据

df.dtypes[df.dtypes==object]  #查得有四列不是数据类型，需要对其进行转化
for i in range(df.dtypes[df.dtypes==object].shape[0]):
    print(df.dtypes[df.dtypes==object].index[i])
    
    
#转化四列数据，转换后df全为数值格式
df["survey_month"] = df["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[1]).astype("int64")   #返回调查月：用空格来切分日期和时间，日期中第1项为月
df["survey_day"] = df["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[2]).astype("int64")   #返回调查日
df["survey_hour"] = df["survey_time"].transform(lambda line:line.split(" ")[1].split(":")[0]).astype("int64")   #返回调查小时
df=df.drop(columns='survey_time')

enc1=preprocessing.OrdinalEncoder()
enc2=preprocessing.OrdinalEncoder()
enc3=preprocessing.OrdinalEncoder()
df['edu_other']=enc1.fit_transform(df['edu_other'].fillna(0).transform(lambda x:str(x)).values.reshape(-1,1))
print(enc.categories_)  #查看编码类型

df['property_other']=enc2.fit_transform(df['property_other'].fillna(0).transform(lambda x:str(x)).values.reshape(-1,1))
print(enc.categories_)  #查看编码类型

df['invest_other']=enc3.fit_transform(df['invest_other'].fillna(0).transform(lambda x:str(x)).values.reshape(-1,1))
print(enc.categories_)  #查看编码类型


#确定X和Y
X=df.drop(columns='happiness').fillna(0)
Y=df.happiness
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-738976cdc4bb2039.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#####1.2 基本模型跑一遍看效果
- 线性回归
```
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection
#=============
#1、线性回归
#=============

#=============
#1.1、普通线性回归
#=============
reg11=linear_model.LinearRegression()
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
#mes1是未取整，mes2是四舍五入取整,mes3是向下取整，mes4是向上取整
mes1=[]
mes2=[]
mes3=[]
mes4=[]
kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=reg1.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    e2=metrics.mean_squared_error(np.round(y_pred),y_test)
    e3=metrics.mean_squared_error(np.trunc(y_pred),y_test)
    e4=metrics.mean_squared_error(np.ceil(y_pred),y_test)
    mes1.append(e1)
    mes2.append(e2)
    mes3.append(e3)
    mes4.append(e4)
print('normal_liner:')
print(mes1)
print(np.mean(mes1))
print('-------------')
print(mes2)
print(np.mean(mes2))
print('-------------')
print(mes3)
print(np.mean(mes3))
print('-------------')
print(mes4)
print(np.mean(mes4))
print()
print()
#表明几种取整的方案都不是很好，不如回归的效果，但是回归的非整数也不满足目标值需求，因此要考虑分类


#=============
#1.2、L1的lasso回归
#=============
reg12=linear_model.Lasso()
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
#mes1是未取整，mes2是四舍五入取整,mes3是向下取整，mes4是向上取整
mes1=[]
mes2=[]
mes3=[]
mes4=[]
kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=reg2.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    e2=metrics.mean_squared_error(np.round(y_pred),y_test)
    e3=metrics.mean_squared_error(np.trunc(y_pred),y_test)
    e4=metrics.mean_squared_error(np.ceil(y_pred),y_test)
    mes1.append(e1)
    mes2.append(e2)
    mes3.append(e3)
    mes4.append(e4)
print('Lasso:')
print(mes1)
print(np.mean(mes1))
print('-------------')
print(mes2)
print(np.mean(mes2))
print('-------------')
print(mes3)
print(np.mean(mes3))
print('-------------')
print(mes4)
print(np.mean(mes4))
print()
print()

#=============
#1.3、L2的岭回归
#=============
reg13=linear_model.Ridge()
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
#mes1是未取整，mes2是四舍五入取整,mes3是向下取整，mes4是向上取整
mes1=[]
mes2=[]
mes3=[]
mes4=[]
kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=reg3.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    e2=metrics.mean_squared_error(np.round(y_pred),y_test)
    e3=metrics.mean_squared_error(np.trunc(y_pred),y_test)
    e4=metrics.mean_squared_error(np.ceil(y_pred),y_test)
    mes1.append(e1)
    mes2.append(e2)
    mes3.append(e3)
    mes4.append(e4)
print('Ridge:')
print(mes1)
print(np.mean(mes1))
print('-------------')
print(mes2)
print(np.mean(mes2))
print('-------------')
print(mes3)
print(np.mean(mes3))
print('-------------')
print(mes4)
print(np.mean(mes4))
print()
print()

#=============
#1.4、逻辑回归
#=============
clf14=linear_model.LogisticRegression(penalty='none',solver='saga') #正则会导致准确率下降，所以不正则
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes1=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=reg3.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes1.append(e1)
print('LR:')
print(mes1)
print(np.mean(mes1))
print()
print()

#结论：普通二乘回归和逻辑回归效果最好
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-3284901c5319adb7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/18032205-29f8aac6423ab6df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- SVM
```
from sklearn import metrics
from sklearn import svm
from sklearn import model_selection
#=============
#2、SVM
#=============
clf2=svm.SVC()   #gamma和C都是默认值，没有调参
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf2.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('SVM:')
print(mes)
print(np.mean(mes))
print()
print()

#结论：效果很一般
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-65d7c814e698058d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- KNN
```
from sklearn import metrics
from sklearn import neighbors
from sklearn import model_selection
#=============
#3、KNN
#=============

for n in range(10,101,10):    #K值肯定会造成影响
    clf3=neighbors.KNeighborsClassifier(n_neighbors=n)  
    #交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
    mes=[]

    kf=model_selection.KFold(10,shuffle=True)
    for train,test in kf.split(X):
        X_train = X.iloc[train]
        y_train = Y.iloc[train]
        X_test = X.iloc[test]
        y_test = Y.iloc[test]

        y_pred=clf3.fit(X_train,y_train).predict(X_test)
        e1=metrics.mean_squared_error(y_pred,y_test)
        mes.append(e1)
    print('KNN(n=%d):'%n)
    print(mes)
    print(np.mean(mes))
    print()
    print()
#结论：效果很一般
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-7af6f59bd8b046ae.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- naive_bayes
```
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import model_selection

X_new=X   # 本来想标准化，但发现标准化后的效果更差，所以就没有标准化
#=============
#4、朴素贝叶斯
#=============
clf4=naive_bayes.GaussianNB()   #多想分布朴素贝叶斯跑不通，必须是正定矩阵什么的，所以这里用的高斯
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X_new.iloc[train]
    y_train = Y.iloc[train]
    X_test = X_new.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf4.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('bayes:')
print(mes)
print(np.mean(mes))
print()
print()

#结论：效果很差，说明确实不适合用高斯贝叶斯，如果用多项式贝叶斯想过可能会更好
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-4373339077be5b0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- 决策树
```
from sklearn import metrics
from sklearn import tree
from sklearn import model_selection
#=============
#5、决策树
#=============

clf5=tree.DecisionTreeClassifier()   
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf5.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()
#结论：效果很差
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-dde6dd1d15a5d983.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- MLP
```
from sklearn import metrics
from sklearn import neural_network
from sklearn import model_selection
#=============
#6、MLP
#=============

clf6=neural_network.MLPClassifier(hidden_layer_sizes=(10,8,5,3,2),activation='logistic')   #随意设置下隐藏层构成 
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf6.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()
#结论：效果竟然还可以，之后可以考虑利用神经网络调参

```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-d268e194dcfa61b9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 随机森林
```
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
#=============
#7、随机森林
#=============

clf7=ensemble.RandomForestRegressor(n_estimators=20,n_jobs=-1)   
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf7.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()


#结论：效果一般，之后考虑调参


#=============
#看一下特征重要程度排序
import matplotlib.pyplot as plt
%matplotlib inline

a=ensemble.RandomForestRegressor(n_estimators=20).fit(X,Y).feature_importances_
temp=np.argsort(a)  #返回index

a=list(a)
a.sort()

b=[]
for i in temp:
    b.append(X.columns[i])

plt.figure(figsize=(10,40))
plt.grid()
plt.barh(b,a,)

#参数结论：
# 1、edu_other、property_other、invest_other这三项转换数据都不太重要，而且property、invest的各项数据似乎都不重要
# 2、前十项中equity、depresion反映社会态度和心态；
#     class、family_income、floor_area反映财富;
#     birth、marital_1st、weight_jin、country反映客观状态
#     survey_day为什么也会有影响，这是一个最有疑问的指标
    
    
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-cd6c85661abdc8c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/18032205-3044b64e85b6e522.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- gdbt
```
from sklearn import metrics
from sklearn import ensemble
from sklearn import model_selection
#=============
#8、gdbt
#=============
clf8=ensemble.GradientBoostingRegressor(max_features=20)   #必须要设置参数，不然跑太慢了
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf8.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()


#结论：效果挺好



#=============
#看一下特征重要程度排序
import matplotlib.pyplot as plt
%matplotlib inline

a=ensemble.GradientBoostingClassifier().fit(X,Y).feature_importances_
temp=np.argsort(a)  #返回index

a=list(a)
a.sort()

b=[]
for i in temp:
    b.append(X.columns[i])

plt.figure(figsize=(10,40))
plt.grid()
plt.barh(b,a,)


```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-f5db5bd470512396.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- xgboost
```
from sklearn import metrics
import xgboost
from sklearn import model_selection
#=============
#9、xgboost
#=============

clf9=xgboost.XGBRegressor()   
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf9.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()


#结论：简直无语，一来就取得这么好的效果。。。。。


#=============
#看一下特征重要程度排序
import matplotlib.pyplot as plt
%matplotlib inline

a=xgboost.XGBRegressor().fit(X,Y).feature_importances_
temp=np.argsort(a)  #返回index

a=list(a)
a.sort()

b=[]
for i in temp:
    b.append(X.columns[i])

plt.figure(figsize=(10,40))
plt.grid()
plt.barh(b,a,)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-98a23394c67b87df.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- lightgbm
```
from sklearn import metrics
import lightgbm
from sklearn import model_selection
#lighgbm防报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#=============
#10、LightGBM
#=============

clf10=lightgbm.LGBMRegressor()   
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    y_pred=clf10.fit(X_train,y_train).predict(X_test)
    e1=metrics.mean_squared_error(y_pred,y_test)
    mes.append(e1)
print('Tree:')
print(mes)
print(np.mean(mes))
print()
print()


#结论：效果也很好，之后再调参。。。。。


#=============
#看一下特征重要程度排序
import matplotlib.pyplot as plt
%matplotlib inline

a=lightgbm.LGBMRegressor().fit(X,Y).feature_importances_
temp=np.argsort(a)  #返回index

a=list(a)
a.sort()

b=[]
for i in temp:
    b.append(X.columns[i])

plt.figure(figsize=(10,40))
plt.grid()
plt.barh(b,a,)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-fe9e0c89ebfa8785.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####1.3 分析结果
从模型结果来看，gdbt、xgboost和lightgbm的效果都很好，随机森林效果很一般，二乘回归和lR的效果也不错，之后考虑利用xgboost、lightgbm、gdbt和随机森林调参增强模型，然后还可以用LR进一步融合模型。

**采用基本xgboost模型提交结果，原始数据结果得分为0.48043，四舍五入得分为0.55394。。。。**
```
df1=pd.read_csv(r'/Users/ranmo/Desktop/天池/幸福感/happiness_test_complete.csv',encoding='GB2312',index_col='id')

    
    
#转化四列数据，转换后df全为数值格式
df1["survey_month"] = df1["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[1]).astype("int64")   #返回调查月：用空格来切分日期和时间，日期中第1项为月
df1["survey_day"] = df1["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[2]).astype("int64")   #返回调查日
df1["survey_hour"] = df1["survey_time"].transform(lambda line:line.split(" ")[1].split(":")[0]).astype("int64")   #返回调查小时
df1=df1.drop(columns='survey_time')



def temp1(a):
    if a not in enc1.categories_[0]:
        return 0
    else:
        return a
df1['edu_other']=enc1.transform(df1['edu_other'].transform(temp1).transform(lambda x:str(x)).values.reshape(-1,1))

def temp2(a):
    if a not in enc2.categories_[0]:
        return 0
    else:
        return a
df1['property_other']=enc2.transform(df1['property_other'].transform(temp2).transform(lambda x:str(x)).values.reshape(-1,1))

def temp3(a):
    if a not in enc3.categories_[0]:
        return 0
    else:
        return a
df1['invest_other']=enc3.transform(df1['invest_other'].transform(temp2).transform(lambda x:str(x)).values.reshape(-1,1))



#确定X_test
X_test=df1.fillna(0)

# 结果1
y_test=xgboost.XGBRegressor().fit(X,Y).predict(X_test)
df1_final=pd.DataFrame({'id':X_test.index,'happiness':y_test}).set_index('id')
df1_final.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/df1_final.csv')
# 结果1四舍五入
df1_final_round=pd.DataFrame({'id':X_test.index,'happiness':np.round(y_test)}).set_index('id')
df1_final_round.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/df1_final.csv')

```

###二、超参数搜索
#####2.1 xgboost
参考[https://blog.csdn.net/han_xiaoyang/article/details/52665396](https://blog.csdn.net/han_xiaoyang/article/details/52665396)

xgboost的参数包括：
- max_depth，这个参数的取值最好在3-10之间。
- min_child_weight，了叶子节点中，样本的权重之和，如果在一次分裂中，叶子结点上所有樣本的權重和小于min_child_weight則停止分裂，能夠有效的防止過擬合，防止學到特殊樣本，默认设置为1。
- gamma，继续分类的损失函数最小的减少值。 起始值一般比较小，0~0.2之间就可以。
- subsample, colsample_bytree:每棵树随机采样的比例，以及每个特征随机采样的比例，典型值的范围在0.5-0.9之间，设置得小容易造成欠拟合。
- scale_pos_weight: 用来解决类别不平衡问题，加快收敛（调整不同样本的学习率），具体原理没有研究，所以也不用管。

等等。

###### 2.1.1 初始化参数

![image.png](https://upload-images.jianshu.io/upload_images/18032205-0d9921a9942a6971.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
#直接按初始参数跑基本模型
clf9=xgboost.XGBRegressor(loss_function='RMSE')
clf=model_selection.GridSearchCV(clf9,{'max_depth':np.array([3])},cv=10,n_jobs=-1,scoring='neg_mean_squared_error') #用均方差计算score
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-e16860866321be79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
ps:虽然cv的scoring如果不设置默认是采用训练模型所采用的score方式，但这里不设置的话结果不对，umm。。可能是xgb的默认score不是rmse吧。。。
#####2.1.2 max_depth 和 min_weight 参数调优
```
#粗调max_depth 和 min_weight 
param_test = {
 'max_depth':range(1,10,2),
 'min_child_weight':range(1,6,2)
}

clf=model_selection.GridSearchCV(clf9,param_test,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print()
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-99aa0d018555c950.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
最优结果基础上，拓展范围进行精调
```
#精调max_depth 和 min_weight 
param_test = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print(clf.best_score_)
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-df45fbfd3a4a950d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
较原始的0.47178有较好下降。
#####2.1.3 gamma参数调优
```
#粗调gamma
#粗调gamma
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.arange(0,0.5,0.1)
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-c104edee31a617c0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
最优结果是初始化参数，所以不用调整。(但为什么结果又下降了？？？)
#####2.1.4 调整subsample 和 colsample_bytree 参数
```
#粗调subsample 和 colsample_bytree
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.array([0]),
'subsample':np.arange(0.6,1,0.1),
'colsample_bytree':np.arange(0.6,1,0.1)  
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-c7a7d585d3c19132.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
最优参数在0.9和0.8，进行精调
```
#精调subsample 和 colsample_bytree
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.array([0]),
'subsample':np.arange(0.75,0.86,0.05),
'colsample_bytree':np.arange(0.75,0.86,0.05)  
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-90e18a1919115403.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
精调寻优的结果是0.75和0.8。（score竟然上涨了？？）
#####2.1.5 正则参数寻优
```
#粗调reg_alpha和reg_lambda
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.array([0]),
'subsample':np.array([0.8]),
'colsample_bytree':np.array([0.75]),  
'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]  
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-ca879d7c62be7cf6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在1,0.1附近搜索下是否有更好的参数。
```
#精调reg_alpha和reg_lambda
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.array([0]),
'subsample':np.array([0.8]),
'colsample_bytree':np.array([0.75]),  
'reg_alpha':[0,0.5,1,2,5],
'reg_lambda':[0,0.05,0.1,0.2,0.5]  
}

clf=model_selection.GridSearchCV(clf9,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)

```
最优参数5，0.1。
#####2.1.6 低学习速率、多树调试最终结果
```
#调试最终结果
param_test = {
'max_depth':np.array([4]),
'min_child_weight':np.array([5]),
'gamma':np.array([0]),
'subsample':np.array([0.8]),
'colsample_bytree':np.array([0.75]),  
'reg_alpha':np.array([5]),
'reg_lambda':np.array([0.1]) ,
'learning_rate':np.array([0.01]),
'n_estimators':np.array([5000]), 
    
}


clf9=xgboost.XGBRegressor(loss_function='RMSE')
clf=model_selection.GridSearchCV(clf9,{'max_depth':np.array([3])},cv=10,n_jobs=-1,scoring='neg_mean_squared_error') #用均方差计算score
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_train_score']:=%s"%clf.cv_results_['mean_train_score'])
print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])

print(clf.best_score_)
print(clf.best_params_)

```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-2341e3453a8c4f4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从数据上来看，效果确实提升不少。由初始的0.47178降为0.46099。不过降低学习率并且增加树的数量后，模型明显变慢，同时在成绩取得上还降低了（可能导致过拟合，或者说本身小幅度的提升或者降低都是很正常的），因此这里实际跑模型并没有采用低学习率和多数的结构。
```
# 结果2

from sklearn import metrics
import xgboost
from sklearn import model_selection
from sklearn.externals import joblib
#=============
#xgboost_modified
#=============

clf_xgboost_modified=xgboost.XGBRegressor(max_depth=4,min_child_weight=5,gamma=0,subsample=0.8,colsample_bytree=0.75,reg_alpha=5,reg_lambda=0.1)   
#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
mes=[]
i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test1 = X.iloc[test]
    y_test1 = Y.iloc[test]
    
    clf_xgboost_modified.fit(X_train,y_train)
    y_pred=clf_xgboost_modified.predict(X_test1)
    e1=metrics.mean_squared_error(y_pred,y_test1)
    mes.append(e1)
    joblib.dump(clf_xgboost_modified,filename='/Users/ranmo/Desktop/天池/幸福感/xgboost/xgboost_%d.pkl'%i)
    
    y_test=clf_xgboost_modified.predict(X_test)

    df2_final=pd.DataFrame({'id':X_test.index,'happiness':y_test}).set_index('id')
    df2_final.to_csv('/Users/ranmo/Desktop/天池/幸福感/xgboost/df2_xgboost_%d.csv'%i)
 
    i+=1
print('clf_xgboost_modified:')
print(mes)
print(np.mean(mes))
print()
print()
```
最佳成绩为0.47675。

#####2.2 lightgbm

#####2.2.1 初始化参数
默认参数：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-9174259431481df5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#直接按初始参数跑基本模型
clf10=lightgbm.LGBMRegressor(metric='l2')   #默认default={l2 for regression}
clf=model_selection.GridSearchCV(clf10,{'max_depth':np.array([-1])},cv=10,n_jobs=-1,scoring='neg_mean_squared_error') #用均方差计算score
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-deb13de3c3d9d1e1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####2.2.2 寻优结果
超参数搜索完全可以按照xgboost那一套。
```
#调试最终结果
clf10=lightgbm.LGBMRegressor(metric='l2')   #默认default={l2 for regression}

param_test = {
'max_depth':np.array([9]),
'min_child_weight':np.array([0.0001]),
'min_split_gain':np.array([0.4]),
'subsample':np.array([0.5]),
'colsample_bytree':np.array([1]),  
'reg_alpha':np.array([1e-05]),
'reg_lambda':np.array([0.0001]) ,
'learning_rate':np.array([0.1]),
}

clf=model_selection.GridSearchCV(clf10,param_test,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
print(clf.best_score_)
print(clf.best_params_)

# 结论：{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 0.0001, 'min_split_gain': 0.4, 'reg_alpha': 1e-05, 'reg_lambda': 0.0001, 'subsample': 0.5}
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-ffb5e9284103c9db.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
rmse由0.47728降为0.47000.
用lightgbm优化后的模型提交成绩后，最优成绩为0.48128。
#####2.3 gdbt
[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)

[https://blog.csdn.net/manjhok/article/details/82017696](https://blog.csdn.net/manjhok/article/details/82017696)
#####2.3.1 初始化参数
默认参数：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-5681bc37c7b21131.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#=============
#GDBT_modified
#=============

#直接按初始参数跑基本模型
clf8=ensemble.GradientBoostingRegressor(loss='ls')   
clf=model_selection.GridSearchCV(clf8,{'max_depth':np.array([3])},cv=10,n_jobs=-1,scoring='neg_mean_squared_error') #用均方差计算score
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-3e97f6c831e01f42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
#####2.3.2 寻优结果
超参数搜索都是差不多的，名称上有差异。
```
#调试最终结果
clf8=ensemble.GradientBoostingRegressor(loss='ls')   

param_test = {   
 'max_depth':np.array([2]),
 'min_weight_fraction_leaf':np.array([0.002]), 
 'min_impurity_split':np.array([0.0001]),
 'subsample':np.array([0.96]),
 'max_features':np.array([0.88]),
 'n_estimators':np.array([80]),   
 'learning_rate':np.array([0.2]),    

}

clf=model_selection.GridSearchCV(clf8,param_test,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
print(clf.best_score_)
print(clf.best_params_)

# 结论：{'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 0.0001, 'min_split_gain': 0.4, 'reg_alpha': 1e-05, 'reg_lambda': 0.0001, 'subsample': 0.5}
```
rmse由0.47534降为0.47148.
用gbdt优化后的模型提交成绩后，最优成绩为0.48317。

#####2.4 随机森林


[https://blog.csdn.net/u012559520/article/details/77336098](https://blog.csdn.net/u012559520/article/details/77336098)

#####2.4.1 初始化参数
![image.png](https://upload-images.jianshu.io/upload_images/18032205-a4530300ccae20d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#=============
#RandomForest_modified
#=============

#直接按初始参数跑基本模型
clf7=ensemble.RandomForestRegressor(criterion='mse',n_jobs=-1)   
clf=model_selection.GridSearchCV(clf7,{'min_samples_split':np.array([2])},cv=10,n_jobs=-1,scoring='neg_mean_squared_error') #用均方差计算score
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-463f18e4221fbd50.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#####2.4.2 寻优参数
```
#调试最终结果

param_test = {   
 'min_samples_split':np.array([4]),          
 'min_weight_fraction_leaf':np.array([0.01]),   
 'min_impurity_decrease':np.array([0]),   
 'n_estimators':[150],  
 'max_features':[0.8],          #随机森林的话这个不能太高吧
}

clf=model_selection.GridSearchCV(clf7,param_test ,cv=10,n_jobs=-1,scoring='neg_mean_squared_error')
clf.fit(X_train,y_train)

print("clf.cv_results_['mean_test_score']:=%s"%clf.cv_results_['mean_test_score'])
print(clf.best_score_)
print(clf.best_params_)

# 结论：{'max_features': 0.8, 'min_impurity_decrease': 0, 'min_samples_split': 4, 'min_weight_fraction_leaf': 0.01, 'n_estimators': 150}
```
rmse由0.53373降为0.48867
用rf优化后的模型提交成绩后，最优成绩为0.51088。
###三、模型融合
#####3.1 平均融合xgboost + lightgbm + gdbt现有模型
```
#平均融合xgboost + lightgbm + gdbt现有模型


#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型
xgboost_mes=[]
lightgbm_mes=[]
gdbt_mes=[]
mix_mes=[]

i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_test1 = X_train.iloc[test]
    y_test1 = y_train.iloc[test]
    
    xgboost=joblib.load(r'C:\Users\sunsharp\Desktop\学习\幸福感\xgboost\xgboost_%d.pkl'%i)
    lightgbm=joblib.load(r'C:\Users\sunsharp\Desktop\学习\幸福感\lightgbm\lightgbm_%d.pkl'%i)
    gdbt=joblib.load(r'C:\Users\sunsharp\Desktop\学习\幸福感\gdbt\gdbt_%d.pkl'%i)
    
    xgboost_y_pred=xgboost.fit(X_train1,y_train1).predict(X_test1)
    lightgbm_y_pred=lightgbm.fit(X_train1,y_train1).predict(X_test1)
    gdbt_y_pred=gdbt.fit(X_train1,y_train1).predict(X_test1)
    mix_y_pred=(xgboost_y_pred+lightgbm_y_pred+gdbt_y_pred)/3
    
    
    
    xgboost_mes.append(metrics.mean_squared_error(xgboost_y_pred,y_test1))
    lightgbm_mes.append(metrics.mean_squared_error(lightgbm_y_pred,y_test1))
    gdbt_mes.append(metrics.mean_squared_error(gdbt_y_pred,y_test1))
    mix_mes.append(metrics.mean_squared_error(mix_y_pred,y_test1))
    
    xgboost_y_test=xgboost.predict(X_test)
    lightgbm_y_test=lightgbm.predict(X_test)
    gdbt_y_test=gdbt.predict(X_test)
    mix_y_test=(xgboost_y_test+lightgbm_y_test+gdbt_y_test)/3

    df_mix_final=pd.DataFrame({'id':X_test.index,'happiness':mix_y_test}).set_index('id')
    df_mix_final.to_csv(r'C:\Users\sunsharp\Desktop\学习\幸福感\mixmodel\df_mix_%d.csv'%i)
 
    i+=1
print('xgboost:')
print(xgboost_mes)
print(np.mean(xgboost_mes))
print()
print('lightgbm:')
print(lightgbm_mes)
print(np.mean(lightgbm_mes))
print()
print('gdbt:')
print(gdbt_mes)
print(np.mean(gdbt_mes))
print()
print('mix:')
print(mix_mes)
print(np.mean(mix_mes))
print()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-13a957d1365590cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从训练集结果上看，结果有轻微提升。
用融合模型提交成绩后，最优成绩为0.47104。

#####3.2 线性回归融合xgboost + lightgbm + gdbt
```
#LR融合xgboost + lightgbm + gdbt现有模型


#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn import metrics
import lightgbm
#lighgbm防报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import xgboost
from sklearn import ensemble
from sklearn import linear_model

xgboost_mes=[]
lightgbm_mes=[]
gdbt_mes=[]
lrmix_mes=[]

i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_test1 = X_train.iloc[test]
    y_test1 = y_train.iloc[test]
    
    print(i)
    
    xgboost=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/xgboost/xgboost_%d.pkl'%i)
    lightgbm=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/lightgbm/lightgbm_%d.pkl'%i)
    gdbt=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/gdbt/gdbt_%d.pkl'%i)
    
    

    
    xgboost_y_pred=xgboost.fit(X_train1,y_train1).predict(X_test1)
    lightgbm_y_pred=lightgbm.fit(X_train1,y_train1).predict(X_test1)
    gdbt_y_pred=gdbt.fit(X_train1,y_train1).predict(X_test1)
    #训练融合模型
    a=xgboost.fit(X_train1,y_train1).predict(X_train1)
    b=lightgbm.fit(X_train1,y_train1).predict(X_train1)
    c=gdbt.fit(X_train1,y_train1).predict(X_train1)
    lr_mix=linear_model.LinearRegression().fit(np.array([a,b,c]).T,y_train1)
    lrmix_y_pred=lr_mix.predict(np.array([xgboost_y_pred,lightgbm_y_pred,gdbt_y_pred]).T)
    
    
    xgboost_mes.append(metrics.mean_squared_error(xgboost_y_pred,y_test1))
    lightgbm_mes.append(metrics.mean_squared_error(lightgbm_y_pred,y_test1))
    gdbt_mes.append(metrics.mean_squared_error(gdbt_y_pred,y_test1))
    lrmix_mes.append(metrics.mean_squared_error(lrmix_y_pred,y_test1))
    
    xgboost_y_test=xgboost.predict(X_test)
    lightgbm_y_test=lightgbm.predict(X_test)
    gdbt_y_test=gdbt.predict(X_test)
    lrmix_y_test=lr_mix.predict(np.array([xgboost_y_test,lightgbm_y_test,gdbt_y_test]).T)

    df_lrmix_final=pd.DataFrame({'id':X_test.index,'happiness':lrmix_y_test}).set_index('id')
    df_lrmix_final.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/lrmixmodel/df_lrmix_%d.csv'%i)
    
    i+=1
print('xgboost:')
print(xgboost_mes)
print(np.mean(xgboost_mes))
print()
print('lightgbm:')
print(lightgbm_mes)
print(np.mean(lightgbm_mes))
print()
print('gdbt:')
print(gdbt_mes)
print(np.mean(gdbt_mes))
print()
print('lrmix:')
print(mix_mes)
print(np.mean(lrmix_mes))
print()
    
```

![image.png](https://upload-images.jianshu.io/upload_images/18032205-c5f8a5f5e4a15099.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
效果不是很理想。不理想的原因是因为对训练集再做回归融合(训练集的成绩能够达到0.15)，虽然能够提升训练集模型精度，但是是过拟合，然后在测试集中就不能取得很好的效果。。。

#####3.3 加权融合xgboost + lightgbm + gdbt
因为平均融合效果好，而回归融合过拟合，但是查看了回归模型的系数，和比较接近于1，因此考虑将三者模型进行加权融合（权重和为1）。
```
a=np.arange(0,1.1,0.05)
b=np.arange(0,1.1,0.05)
c=np.arange(0,1.1,0.05)

coef_list=[]
for i in a:
    for j in b:
        for k in c:
            if i+j+k==1:
                coef_list.append([i,j,k])
                
                
                
#加权融合xgboost + lightgbm + gdbt现有模型


#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn import metrics
import lightgbm
#lighgbm防报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import xgboost
from sklearn import ensemble

xgboost_mes=[]
lightgbm_mes=[]
gdbt_mes=[]
weightmix_mes=[]

i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_test1 = X_train.iloc[test]
    y_test1 = y_train.iloc[test]
    
    print(i)
    
    xgboost=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/xgboost/xgboost_%d.pkl'%i)
    lightgbm=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/lightgbm/lightgbm_%d.pkl'%i)
    gdbt=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/gdbt/gdbt_%d.pkl'%i)
    
    

    
    xgboost_y_pred=xgboost.fit(X_train1,y_train1).predict(X_test1)
    lightgbm_y_pred=lightgbm.fit(X_train1,y_train1).predict(X_test1)
    gdbt_y_pred=gdbt.fit(X_train1,y_train1).predict(X_test1)
    #训练融合模型
    error_list=[]
    for coef_i in coef_list:
        error_list.append(metrics.mean_squared_error(np.dot(np.array([xgboost_y_pred,lightgbm_y_pred,gdbt_y_pred]).T,coef_i),y_test1))
    coef=temp[np.argmin(error_list)]
    
    
    xgboost_mes.append(metrics.mean_squared_error(xgboost_y_pred,y_test1))
    lightgbm_mes.append(metrics.mean_squared_error(lightgbm_y_pred,y_test1))
    gdbt_mes.append(metrics.mean_squared_error(gdbt_y_pred,y_test1))
    weightmix_mes.append(min(error_list))
    
    xgboost_y_test=xgboost.predict(X_test)
    lightgbm_y_test=lightgbm.predict(X_test)
    gdbt_y_test=gdbt.predict(X_test)
    weightmix_y_test=np.dot(np.array([xgboost_y_test,lightgbm_y_test,gdbt_y_test]).T,coef)

    df_weightmix_final=pd.DataFrame({'id':X_test.index,'happiness':weightmix_y_test}).set_index('id')
    df_weightmix_final.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/weightmixmodel/df_weightmix_%d.csv'%i)
    
    i+=1
print('xgboost:')
print(xgboost_mes)
print(np.mean(xgboost_mes))
print()
print('lightgbm:')
print(lightgbm_mes)
print(np.mean(lightgbm_mes))
print()
print('gdbt:')
print(gdbt_mes)
print(np.mean(gdbt_mes))
print()
print('weightmix:')
print(weightmix_mes)
print(np.mean(weightmix_mes))
print()
    
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-b8aec841550611c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
模型结果有轻微提升。实际最优成绩为0.47531。


#####3.4 神经网络融合xgboost + lightgbm + gdbt
```
#神经网络融合xgboost + lightgbm + gdbt现有模型


#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn import metrics
import lightgbm
#lighgbm防报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import xgboost
from sklearn import ensemble
from sklearn import neural_network

xgboost_mes=[]
lightgbm_mes=[]
gdbt_mes=[]
MLPmix_mes=[]

i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X):
    X_train1 = X_train.iloc[train]
    y_train1 = y_train.iloc[train]
    X_test1 = X_train.iloc[test]
    y_test1 = y_train.iloc[test]
    
    print(i)
    
    xgboost=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/xgboost/xgboost_%d.pkl'%i)
    lightgbm=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/lightgbm/lightgbm_%d.pkl'%i)
    gdbt=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/gdbt/gdbt_%d.pkl'%i)
    
    

    
    xgboost_y_pred=xgboost.fit(X_train1,y_train1).predict(X_test1)
    lightgbm_y_pred=lightgbm.fit(X_train1,y_train1).predict(X_test1)
    gdbt_y_pred=gdbt.fit(X_train1,y_train1).predict(X_test1)
    #训练融合模型
    a=xgboost.fit(X_train1,y_train1).predict(X_train1)
    b=lightgbm.fit(X_train1,y_train1).predict(X_train1)
    c=gdbt.fit(X_train1,y_train1).predict(X_train1)
    MLP_mix=neural_network.MLPClassifier(hidden_layer_sizes=(5,3,2),activation='logistic').fit(np.array([a,b,c]).T,y_train1)
    MLPmix_y_pred=MLP_mix.predict(np.array([xgboost_y_pred,lightgbm_y_pred,gdbt_y_pred]).T)
    
    
    xgboost_mes.append(metrics.mean_squared_error(xgboost_y_pred,y_test1))
    lightgbm_mes.append(metrics.mean_squared_error(lightgbm_y_pred,y_test1))
    gdbt_mes.append(metrics.mean_squared_error(gdbt_y_pred,y_test1))
    MLPmix_mes.append(metrics.mean_squared_error(MLPmix_y_pred,y_test1))
    
    xgboost_y_test=xgboost.predict(X_test)
    lightgbm_y_test=lightgbm.predict(X_test)
    gdbt_y_test=gdbt.predict(X_test)
    MLPmix_y_test=MLP_mix.predict(np.array([xgboost_y_test,lightgbm_y_test,gdbt_y_test]).T)

    df_MLPmix_final=pd.DataFrame({'id':X_test.index,'happiness':MLPmix_y_test}).set_index('id')
    df_MLPmix_final.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/MLPmixmodel/df_MLPmix_%d.csv'%i)
    
    i+=1
print('xgboost:')
print(xgboost_mes)
print(np.mean(xgboost_mes))
print()
print('lightgbm:')
print(lightgbm_mes)
print(np.mean(lightgbm_mes))
print()
print('gdbt:')
print(gdbt_mes)
print(np.mean(gdbt_mes))
print()
print('MLPmix:')
print(mix_mes)
print(np.mean(MLPmix_mes))
print()
    
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-5a87aabe023ebbcf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

效果也不理想。

###四、简单特征工程
本来这部分工作应该是在建模之前做的，但是现在的集成算法已经能够很好地寻找重要特征，并且减小非重要特征的权重，所以大大减少了寻找特征工程的工作量。但是另一方面，要寻找好的特征工程并快速提高模型精度是很费精力的部分，所以限于此，先跑的模型，并基于模型给出来的特征重要性，适当进行开展特征工程。
#####4.1 去除不重要的特征
![image.png](https://upload-images.jianshu.io/upload_images/18032205-07fd64424aede1a0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从集中学习模型给出的特征重要度来看，不重要的特征主要是：
- edu_other：考虑去除edu_other
- invest和invest_other：考虑去除inverst全部项和invest_other
- property和property_other:考虑去除property_other
- s_work_type：考虑去除s_work_type
#####4.1.1 低方差
```
np.var(X_train)[np.var(X_train)<=np.percentile(np.var(X_train),20)]
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-fd2cec865aeec441.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看到，方差小于0.1的特征项：
- edu_other
- property_0、property_3~property_7
- invest_0~invest_8
之后会移除这部分特征项
#####4.1.2 卡方校验
卡方校验的时候发现出现非正定矩阵无法校验，进一步检验发现数据项中有很多负值部分：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-abd2cb4b00f6329f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
所以其实原始数据中有错误数据，并且在建模前就应该处理。
这里将负值都处理为该特征项的众数，并进行卡方校验。
```
X_train_new=X_train

#负值处理为众数
dict_temp={}
for i in X_train_new.columns:
    dict_temp[i]=X_train_new[i].value_counts().index[0]

for i in dict_temp.keys():
    X_train_new[i][X_train_new[i]<0]=dict_temp[i]

#处理完之后竟然还有负值，那就直接处理为其绝对值
X_train_new=np.abs(X_train_new)
```
```
p_value=feature_selection.chi2(X_train_new,y_train)[1]
p_value[np.isnan(p_value)]=0  #有0值



#看一下特征重要程度排序
import matplotlib.pyplot as plt
%matplotlib inline

temp=np.argsort(-p_value)  #返回index

p_value=list(p_value)
p_value=np.sort(p_value)

b=[]
for i in temp:
    b.append(X_train_new.columns[i])

plt.figure(figsize=(10,40))
plt.grid()
plt.barh(b,p_value,)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-40640c3e465ce09b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/18032205-e2ad1870a0a554a1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看到，与目标变量密切相关的主要是：
- income收入部分；
- marital婚姻情况；
- 自己以及父母的出生年份；
- public_service对公共服务的满意度等等；

而前文提到的edu_other、property_0、property_3~property_7、invest_0~invest_8基本上属于无关变量，唯一的特例是invest_6的p值较高，但是这里仍然进行移除。
#####4.2 修正模型
```
##最后一次平均融合

#平均融合xgboost + lightgbm + gdbt现有模型


#交叉验证确定准确率，因为对回归值会采用取整操作，所以不用自带的交叉验证模型

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn import metrics
import lightgbm
#lighgbm防报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import xgboost
from sklearn import ensemble

xgboost_mes=[]
lightgbm_mes=[]
gdbt_mes=[]
mix_mes=[]

i=0

kf=model_selection.KFold(10,shuffle=True)
for train,test in kf.split(X_train_new):
    X_train1 = X_train_new.iloc[train]
    y_train1 = y_train.iloc[train]
    X_test1 = X_train_new.iloc[test]
    y_test1 = y_train.iloc[test]
    
    print(i)
    
    xgboost=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/feature/xgboost/xgboost_%d.pkl'%i)
    lightgbm=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/feature/lightgbm/lightgbm_%d.pkl'%i)
    gdbt=joblib.load(r'/Users/ranmo/Desktop/天池/幸福感/feature/gdbt/gdbt_%d.pkl'%i)
    
    xgboost_y_pred=xgboost.fit(X_train1,y_train1).predict(X_test1)
    lightgbm_y_pred=lightgbm.fit(X_train1,y_train1).predict(X_test1)
    gdbt_y_pred=gdbt.fit(X_train1,y_train1).predict(X_test1)
    mix_y_pred=(xgboost_y_pred+lightgbm_y_pred+gdbt_y_pred)/3
    
    
    
    xgboost_mes.append(metrics.mean_squared_error(xgboost_y_pred,y_test1))
    lightgbm_mes.append(metrics.mean_squared_error(lightgbm_y_pred,y_test1))
    gdbt_mes.append(metrics.mean_squared_error(gdbt_y_pred,y_test1))
    mix_mes.append(metrics.mean_squared_error(mix_y_pred,y_test1))
    
    xgboost_y_test=xgboost.predict(X_test_new)
    lightgbm_y_test=lightgbm.predict(X_test_new)
    gdbt_y_test=gdbt.predict(X_test_new)
    mix_y_test=(xgboost_y_test+lightgbm_y_test+gdbt_y_test)/3

    df_mix_final=pd.DataFrame({'id':X_test.index,'happiness':mix_y_test}).set_index('id')
    df_mix_final.to_csv(r'/Users/ranmo/Desktop/天池/幸福感/feature/mixmodel/df_mix_%d.csv'%i)
    
    i+=1
print('xgboost:')
print(xgboost_mes)
print(np.mean(xgboost_mes))
print()
print('lightgbm:')
print(lightgbm_mes)
print(np.mean(lightgbm_mes))
print()
print('gdbt:')
print(gdbt_mes)
print(np.mean(gdbt_mes))
print()
print('mix:')
print(mix_mes)
print(np.mean(mix_mes))
print()

```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-71395e6970d2d3d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从结果上看，经过简单特征工程处理的模型和原有模型能够达到的最优结果是差不多的，所以确实是因为集成算法已经能够很好地处理特征了。。
最后用随机种子尝试了最终的优化（在模型稳定的基础上并无太大意义，只是看分数能不能高一点而已），baseline为0.47098。


**over**。
