# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:08:30 2019

@author: Trajan
"""

data_path = "data/"
year_predict_size = 16000
month_predict_size = 1400
ensemble_in_predict = False
tuning_list = ['eta', 'n_estimators', 'alpha', 'alpha']
tuning_pool_list = [
        [0.2,0.5,0.7,1,1.5],
        [50,70,100],
        [0.1,0.3,0.5,0.7,0.9],
        [60,63,66,70]
        ]
model_used_in_test = 1
weight_in_test = [0.2,0.4,0.2,0.2]
split = 4
correlated_group = ['builtfAR', 'residfAR', 'facilfAR',"resarea","assessland","assesstot","exemptland","gross"
                    ,"garagearea","retailarea","bldgdepth","lotdepth","buildingclass"]
