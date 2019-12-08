# coding=utf-8
import pandas as pd
import numpy as np

df=pd.read_csv(r'C:\Users\sunsharp\Desktop\幸福感\happiness_train_complete.csv',encoding='GB2312')
df = df.sample(frac=1,replace=False,random_state=11)    #100%无放回抽样，相当于只是打乱原表的行的顺序

df.reset_index(inplace=True)   #将打乱顺序的表重新编码index
df = df[df["happiness"]>0]   #原表中幸福度非正的都是错误数据,可以剔除12条错误数据
Y = df["happiness"]    #训练集模型值

df["survey_month"] = df["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[1]).astype("int64")   #返回调查月：用空格来切分日期和时间，日期中第1项为月
df["survey_day"] = df["survey_time"].transform(lambda line:line.split(" ")[0].split("/")[2]).astype("int64")   #返回调查日
df["survey_hour"] = df["survey_time"].transform(lambda line:line.split(" ")[1].split(":")[0]).astype("int64")   #返回调查小时

X = df.drop(columns=["id","index","happiness","survey_time","edu_other","property_other","invest_other"])   #除了剔除几个序列标号项外，还有三项数据量很小且无关紧要的列被剔除了

from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

kfold = KFold(n_splits=15, shuffle=True, random_state=11)
model = XGBRegressor(booster='gbtree', colsample_bylevel=0.1,
                     colsample_bytree=0.971, gamma=0.11, learning_rate=0.069, max_delta_step=0,
                     max_depth=3, min_child_weight=1, missing=None, n_estimators=499,
                     n_jobs=-1, nthread=50, objective='reg:linear', random_state=0,
                     reg_alpha=0.1, reg_lambda=1, scale_pos_weight=1, seed=None,
                     silent=True, subsample=1.0)
mse = []
i = 0
for train, test in kfold.split(X):
    X_train = X.iloc[train]
    y_train = Y.iloc[train]
    X_test = X.iloc[test]
    y_test = Y.iloc[test]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    xg_mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    mse.append(xg_mse)
    print("xgboost", xg_mse)
    i += 1
print("xgboost", np.mean(mse), mse)
