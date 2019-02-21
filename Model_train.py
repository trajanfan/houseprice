# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:55:23 2019

@author: Trajan
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import time
from Parameters import param_list
import matplotlib.pyplot as plt
from Config import tuning_list, tuning_pool_list, split, data_path

def model_set(X_train, X_valid, y_train, y_valid, param):
    if param["task"] == "gred_bst":
        ## regression & pairwise ranking with xgboost
        dtrain_base = xgb.DMatrix(X_train, label=y_train)
        dvalid_base = xgb.DMatrix(X_valid, label=y_valid)
        bst = xgb.train(param, dtrain_base)
        pred = bst.predict(dvalid_base)
            
    elif param['task'] == "reg_skl_rf":
        ## regression with sklearn random forest regressor
        rf = RandomForestRegressor(n_estimators=param['n_estimators'],
                        n_jobs=param['n_jobs'],
                        random_state=param['random_state'])
        rf.fit(X_train, y_train)
        pred = rf.predict(X_valid)
            
    elif param['task'] == "reg_skl_ridge":
        ## regression with sklearn ridge regression
        ridge = Ridge(alpha=param["alpha"], normalize=True)
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_valid)
            
    elif param['task'] == "reg_skl_lasso":
        ## regression with sklearn lasso regression
        lasso = Lasso(alpha=param["alpha"], normalize=True)
        lasso.fit(X_train, y_train)
        pred = lasso.predict(X_valid)
    
    return pred

def cross_valid(path, n, param_list, columns, bagging = False, bagging_size = 100):
    MAPE_cv = []
    for i in range(n):
        train_data = pd.DataFrame(columns = columns)
        for j in range(n):
            if i==j:
                valid_data = pd.read_csv(path + "train_" + str(j) +".csv")
                valid_data = valid_data.iloc[:,1:]
            else:
                temp = pd.read_csv(path + "train_" + str(j) +".csv").iloc[:,1:]
                train_data = pd.concat([train_data,temp])
        if bagging:
            train_data_list = []
            for i in range(bagging_size):
                train_data = train_data.sample(frac=1, axis=1).reset_index(drop=True)
                train_data_list.append(train_data.iloc[:50000])
            MAPE, pred = model_train_bagging(train_data_list,valid_data,param_list)
            MAPE_cv.append(MAPE)
        else:
            MAPE, pred = model_train(train_data,valid_data,param_list)
            MAPE_cv.append(MAPE)
    return MAPE_cv,np.mean(MAPE_cv)

def MdAPE(pred, actual):
    mape = np.median(np.abs((pred-actual)/actual))
    return mape

def model_train_bagging(train_data_list, valid_data, param):
    X_valid = valid_data.iloc[:,:-1].values
    y_valid = valid_data.iloc[:,-1].values
    pred_list = []
    for train_data in train_data_list:
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        pred = model_set(X_train, X_valid, y_train, y_valid, param)
        pred_list.append(pred)
    mape = MdAPE(np.mean(pred), y_valid)
    return mape, np.mean(pred)

def model_train(train_data, valid_data, param):
    X_train = train_data.iloc[:,:-1].values
    y_train = train_data.iloc[:,-1].values
    X_valid = valid_data.iloc[:,:-1].values
    y_valid = valid_data.iloc[:,-1].values
    pred = model_set(X_train, X_valid, y_train, y_valid, param)
    mape = MdAPE(pred, y_valid)
    return mape, pred

def model_train_ensembling(train_data, valid_data, param_list, weight):
    X_train = train_data.iloc[:,:-1].values
    y_train = train_data.iloc[:,-1].values
    X_valid = valid_data.iloc[:,:-1].values
    y_valid = valid_data.iloc[:,-1].values
    pred_list = np.zeros(len(y_valid))
    for i in range(len(param_list)):
        pred = model_set(X_train, X_valid, y_train, y_valid, param_list[i])
        pred_list = pred_list+pred*weight[i]
    mape = MdAPE(pred_list, y_valid)
    return mape, pred

def parameter_tuning(param, tuning_param, tuning_pool, path, columns):
    mape_list = []
    for param[tuning_param] in tuning_pool:
        mape_cv, mape = cross_valid(path, split, param, columns)
        mape_list.append(mape)
    plt.plot(tuning_pool, mape_list)
    plt.title(param[tuning_param])
    plt.legend(tuning_param)
    plt.show()

def feature_importance(core,data):
    feature_importance = core.feature_importances_
    feature_importance = feature_importance / feature_importance.max()
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos,feature_importance[sorted_idx],  align='center')
    plt.yticks(pos, data.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
if __name__ == '__main__':
    path = data_path
    test = pd.read_csv(path + "test.csv")
    columns = test.columns[1:]
    
    # tuning parameters
    for i in range(4):
        param = param_list[i]
        tuning_param = tuning_list[i]
        tuning_pool = tuning_pool_list[i]
        parameter_tuning(param, tuning_param, tuning_pool, path, columns)
        
    # single model
    std_list = []
    mdape_list = []
    for param in param_list:
        print("--------------------------------------")
        print(param["task"])
        start = time.time()
        mape_list, mape = cross_valid(path, split, param, columns)
        end = time.time()
        std_list.append(np.std(mape_list))
        mdape_list.append(mape)
        print("time use:", end-start)
        print(mape_list)
        print(mape)
        
    # =============================================================================
    # this part is quite time comsuming, so I did't use them in the final result   
    #     #bagging model
    #     std_list = []
    #     mdape_list = []
    #     for param in param_list:
    #         print("--------------------------------------")
    #         print(param["task"])
    #         start = time.time()
    #         mape = cross_valid("data/", 4, param, columns, bagging = True)
    #         end = time.time()
    #         std_list.append(np.std(mape_list))
    #         mdape_list.append(mape)
    #         print("time use:", end-start)
    #         print(mape)
    # =============================================================================
        
    # plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx() 
    x = list(range(4))
    p1 = ax.bar(x,std_list,0.35, tick_label =["bst","rf","ridge","lasso"],color ='r')
    x=[a+0.35 for a in x]
    p2 = ax2.bar(x,mdape_list,0.35, tick_label =["bst","rf","ridge","lasso"])
    plt.title("training results")
    plt.legend((p1[0], p2[0]), ('standard deviation', 'MdAPE'))
    plt.show()
