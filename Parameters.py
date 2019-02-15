# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:07:10 2019

@author: Trajan
"""
## grediant boosting regression
param_space_reg_xgb_tree = {
    'task': 'gred_bst',
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'eta': 0.7,
    'gamma': 5,
    'silent': 1,
    'seed': 2018,
    "max_evals": 1,
}

## random forest regressor
param_space_reg_skl_rf = {
    'task': 'reg_skl_rf',
    'n_estimators': 100,
    'n_jobs': 1,
    'random_state': 2018,
    "max_evals": 1
}

## ridge regression
param_space_reg_skl_ridge = {
    'task': 'reg_skl_ridge',
    'alpha': 0.7,
    'random_state': 2018,
    "max_evals": 1,
}

## lasso regression
param_space_reg_skl_lasso = {
    'task': 'reg_skl_lasso',
    'alpha': 66,
    'random_state': 2018,
    "max_evals": 1,
}



param_list = [param_space_reg_xgb_tree, param_space_reg_skl_rf, param_space_reg_skl_ridge, param_space_reg_skl_lasso]
