# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 19:59:14 2019

@author: Trajan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Feature_generate
import Config
from Parameters import param_list
from Model_train import model_train_ensembling, model_train

def sample_data(path, all_data, time = 'yearly'):
    # data for yearly/monthly prediction, for time = 'yearly' or 'monthly'
    if time == 'yearly':
        sample_data = all_data[all_data["yearsale"] != 2010]
        sample_data = sample_data.sample(Config.year_predict_size)
        sample_data["yearsale"] = 2018
        sample_data.to_csv(path + 'year_prediction.csv')
    else:
        sample_data = all_data.sample(Config.month_predict_size)
        sample_data["yearsale"] = 1514764800000000000
        sample_data.to_csv(path + 'month_prediction.csv')
        
def create_sample_file(path, data, time = 'yearly'):
    # create the sample file for prediction
    all_data = Feature_generate.feature_generate(data, time)
    all_data = Feature_generate.feature_selection(all_data)
    all_data.to_csv(path + 'all_data_' + time + '.csv')
    sample_data(path, all_data)
    
def predict(path, train_data, former_data, param, ensembling = True, time = 'yearly'):
    # prediction yearly/monthly using ensembling or not
    if time == 'yearly':
        sample_data = pd.read_csv(path + 'year_prediction.csv')
    else:
        sample_data = pd.read_csv(path + 'month_prediction.csv')
    if ensembling:
        mape, pred = model_train_ensembling(train_data,sample_data.iloc[:,1:], param, Config.weight_in_test)
    else:
        mape, pred = model_train(train_data,sample_data.iloc[:,1:], param[Config.model_used_in_test])
    length = len(sample_data)
    predict_distribution(pred, length)
    predict_in_trend(former_data, np.mean(pred), time)
    return mape, pred

def predict_distribution(pred, length):
    # draw the distribution plot
    print("mean price of prediction", np.mean(pred))
    plt.figure(figsize=(15, 7))
    t1 = range(length)
    plt.plot(t1,np.log(pred), '.')
    plt.axhline(y=np.log(np.mean(pred)), xmin=0, xmax=length,linewidth=2, color = 'r')
    plt.show()

def predict_in_trend(former_data, price, time = 'yearly'):
    # draw the trend plot
    if time == 'yearly':
        a = former_data.groupby(former_data["yearsale"]).price.mean()
    else:
        former_data["yearsale"] = pd.to_datetime(former_data["yearsale"]).dt.strftime('%Y-%m-%d').str[:7]
        a = former_data.groupby(former_data["yearsale"]).price.mean()
    b = a.copy()
    b.loc['2018.0'] = price
    plt.figure(figsize=(8, 4))
    plt.title(time + "prediction")
    plt.plot(b, 'r')
    plt.plot(a, 'g')
    plt.show()
    
if __name__ == '__main__':
    path = Config.data_path
    mapdata = pd.read_csv(path + "brooklyn_sales_map.csv")
    test = pd.read_csv(path + "test.csv")
    columns = test.columns[1:]    

    
    # Generate yearly sample
    create_sample_file(path, mapdata, 'yearly')
    # Generate monthly sample
    create_sample_file(path, mapdata, 'monthly')
    
    train_data = pd.DataFrame(columns = columns)
    for j in range(4):
        temp = pd.read_csv(path + "train_" + str(j) +".csv").iloc[:,1:]
        train_data = pd.concat([train_data,temp])
    
    former_data = pd.read_csv(path + 'all_data_yearly.csv')
    mape, pred = predict(path, train_data, former_data.iloc[:,1:], param_list, ensembling = Config.ensemble_in_predict, time = 'yearly')
    
    former_data = pd.read_csv(path + 'all_data_monthly.csv')
    mape, pred = predict(path, train_data, former_data.iloc[:,1:], param_list, ensembling = Config.ensemble_in_predict, time = 'monthly')
    
    
    
    
    