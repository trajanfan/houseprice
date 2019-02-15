# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:26:05 2019

@author: Trajan
"""
import pandas as pd
from Model_train import model_train_ensembling, model_train
from Parameters import param_list
from Config import model_used_in_test,weight_in_test,split,data_path

if __name__ == '__main__':
    path = data_path
    test = pd.read_csv(path + "test.csv")
    columns = test.columns[1:]    

    train_data = pd.DataFrame(columns = columns)
    for j in range(split):
        temp = pd.read_csv(path + "train_" + str(j) +".csv").iloc[:,1:]
        train_data = pd.concat([train_data,temp])
    
    # test set
    mape, pred = model_train_ensembling(train_data,test.iloc[:,1:],param_list, weight_in_test)
    print('Using ensembling method',mape)
    
    # test set without ensembling
    mape, pred = model_train(train_data,test.iloc[:,1:], param_list[model_used_in_test])
    print('Not using ensembling method',mape)
    
    # correlation among predictions
    pred_pool = []
    for i in range(4):
        mape, pred = model_train(train_data,test.iloc[:,1:], param_list[i])
        pred_pool.append(pred)
    pred_array = np.array(pred_pool)
    print(np.corrcoef(pred_array))
