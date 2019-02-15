# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:41:51 2019

@author: Trajan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Config

def feature_generate(mapdata, time = 'monthly'): # time choose from 'monthly' and 'yearly'
    # collums without missing data
    block = mapdata["block"]
    lot = mapdata["lot"]
    residential = mapdata["residential_units"]
    commercial = mapdata["commercial_units"]
    if time == 'monthly':
        yearsale = pd.to_numeric(pd.to_datetime(mapdata["sale_date"],format="%Y-%m-%d"))
    else:
        yearsale = mapdata["year_of_sale"]
    
    # building class only take the number
    buildingclass = mapdata["building_class_category"].fillna(mapdata['building_class_at_sale']).astype('category').cat.codes
    
    # convert categories to interger
    neighborhood = mapdata["neighborhood"].astype('category').cat.codes
    irrlotCode = mapdata['IrrLotCode'].fillna(mapdata["IrrLotCode"].mode()[0]).astype('category').cat.codes
    landuse = mapdata["LandUse"].astype('category').cat.codes
    texclass = mapdata["tax_class"].fillna(mapdata["tax_class"].mode()[0]).astype('category').cat.codes
    
    # replace 0 to median of the collum
    yearbuilt = mapdata["year_built"].replace(0,mapdata["year_built"].median())
    
    # replace nan to mode of the collum
    proxcode = mapdata['ProxCode'].fillna(mapdata["ProxCode"].mode()[0])
    lottype = mapdata['LotType'].fillna(mapdata["LotType"].mode()[0])
    bsmtcode = mapdata['BsmtCode'].fillna(mapdata["BsmtCode"].mode()[0])
    zipcode = mapdata["zip_code"].replace(0,mapdata["zip_code"].mode()[0])
    
    # address only take the street part
    address = mapdata["address"].str.replace("(^[0-9-]*)","")
    address = address.str.replace("((?<=[0-9])(TH|ST|RD))","")
    address = address.str.replace(" N "," NORTH ")
    address = address.str.replace("  "," ")
    address = address.astype('category').cat.codes
    
    # estimate missing land squarefeet by median of collum, and estimate missing
    # gross squarefeet by land squarefeet multiply by number of floors.
    land = mapdata["land_sqft"]
    gross = mapdata["gross_sqft"]
    gross.loc[gross==0] = land.loc[gross==0]*mapdata['NumFloors'].loc[gross==0]
    gross = gross.fillna(0.0)
    land = land.replace(0.0,land.median())
    estimate_gross = mapdata['NumFloors'].median()*land
    gross.loc[gross==0] = estimate_gross[gross==0]
    
    # replace nan to median of the collum
    comarea = mapdata["ComArea"].fillna(mapdata["ComArea"].median())
    resarea = mapdata["ResArea"].fillna(mapdata["ResArea"].median())
    officearea = mapdata["OfficeArea"].fillna(mapdata["OfficeArea"].median())
    retailarea = mapdata["RetailArea"].fillna(mapdata["RetailArea"].median())
    garagearea = mapdata["GarageArea"].fillna(mapdata["GarageArea"].median())
    strgearea = mapdata["StrgeArea"].fillna(mapdata["StrgeArea"].median())
    factryarea = mapdata["FactryArea"].fillna(mapdata["FactryArea"].median())
    otherarea = mapdata["OtherArea"].fillna(mapdata["OtherArea"].median())
    lotfront = mapdata['LotFront'].fillna(mapdata["LotFront"].median())
    bldgfront = mapdata['BldgFront'].fillna(mapdata["BldgFront"].median())
    lotdepth = mapdata['LotDepth'].fillna(mapdata["LotDepth"].median())
    bldgdepth = mapdata['BldgDepth'].fillna(mapdata["BldgDepth"].median())
    health = mapdata["HealthArea"].fillna(mapdata["HealthArea"].median())
    numbldgs = mapdata["NumBldgs"].fillna(mapdata["NumBldgs"].median())
    assessland = mapdata['AssessLand'].fillna(mapdata["AssessLand"].median())
    assesstot = mapdata['AssessTot'].fillna(mapdata["AssessTot"].median())
    exemptland = mapdata['ExemptLand'].fillna(mapdata["ExemptLand"].median())
    exempttot = mapdata['ExemptTot'].fillna(mapdata["ExemptTot"].median())
    builtfAR = mapdata['BuiltFAR'].fillna(mapdata["BuiltFAR"].median())
    residfAR = mapdata['ResidFAR'].fillna(mapdata["ResidFAR"].median())
    commfAR = mapdata['CommFAR'].fillna(mapdata["CommFAR"].median())
    facilfAR = mapdata['FacilFAR'].fillna(mapdata["FacilFAR"].median())
    
    # keep all the price data here
    price = mapdata["sale_price"]
    
    # build dataframe
    all_data = pd.DataFrame([
        block, lot, residential, commercial, yearsale, buildingclass, neighborhood, irrlotCode, landuse, zipcode, yearbuilt,
        proxcode, lottype, bsmtcode, texclass, land, gross, address, comarea, resarea, officearea, retailarea, garagearea,
        strgearea, factryarea, otherarea, lotfront, bldgfront, lotdepth, bldgdepth, health, numbldgs, assessland, assesstot,
        exemptland, exempttot, builtfAR, residfAR, commfAR, facilfAR, price
    ]).T
    all_data.columns = ["block", "lot", "residential", "commercial", "yearsale", "buildingclass", "neighborhood", "irrlotCode", 
                        "landuse", "zipcode", "yearbuilt","proxcode", "lottype", 'bsmtcode', 'texclass', "land", "gross",
                        "address", "comarea", 'resarea', 'officearea', "retailarea", 'garagearea', "strgearea", 'factryarea',
                        "otherarea", 'lotfront', "bldgfront", "lotdepth", 'bldgdepth', 'health', "numbldgs", "assessland",
                        "assesstot",'exemptland', 'exempttot', 'builtfAR', 'residfAR', 'commfAR', 'facilfAR', 'price']
    
    # drop all the rows with price equal to 0.
    all_data = all_data[all_data.price!=0]
    
    return all_data

def data_split(all_data, path):
    # build training set and test set
    length_train = int(len(all_data)/(Config.split*10000))*10000
    
    for i in range(Config.split):
        temp = all_data.iloc[length_train*i:(i+1)*length_train]
        temp.to_csv(path + 'train_' + str(i)+ '.csv')
    test = all_data.iloc[length_train*(Config.split):]
    test.to_csv(path + 'test.csv')
    
def feature_selection(all_data):
    
    # create dummy veriables
    lottype = pd.get_dummies(all_data["lottype"], prefix = "lottype")
    irrlotCode = pd.get_dummies(all_data["irrlotCode"], prefix = "irrlotCode")
    proxcode = pd.get_dummies(all_data["proxcode"], prefix = "proxcode")
    texclass = pd.get_dummies(all_data["texclass"], prefix = "texclass")
    bsmtcode = pd.get_dummies(all_data["bsmtcode"], prefix = "bsmtcode")
    neighborhood = pd.get_dummies(all_data["neighborhood"], prefix = "neighborhood")
    landuse = pd.get_dummies(all_data["landuse"], prefix = "landuse")
    price = all_data["price"]
    temp = all_data.drop(["lottype", "irrlotCode", "proxcode", "texclass", "bsmtcode", "neighborhood","landuse","price"], axis=1)
    
    # delete correlated variables
    for group in Config.correlated_group:
        temp.drop(group[1:], axis=1)
    all_data = pd.concat([temp,irrlotCode, proxcode, lottype, texclass, bsmtcode, neighborhood, landuse, price], axis  = 1)
    
    all_data = all_data.sample(frac = 1)
    all_data.index = range(len(all_data))
    
    return all_data

def trend(feature,data):
    value = []
    year = []
    for group in data.groupby('year'):
        year.append(group[0])
        value.append(group[1][feature].mean())
    plt.plot(year,value)
    plt.show()
 
def trend_draw(data, mapdata, time = 'year'):
    # draw the yearly/monthly trend of the features, time = 'year' or 'month'
    value = []
    label = []
    month = mapdata["sale_date"].str[:7]
    year = mapdata["sale_date"].str[:4]
    data["month"] = month
    data["year"] = year
    for group in data.groupby(time):
        label.append(group[0])
        value.append(group[1].count())
    print("number of sales")
    plt.plot(label,value)
    plt.show()
    for feature in ["block", "lot", "residential", "commercial", "yearbuilt", "land", "gross","comarea", 'resarea', 
                    'officearea', "retailarea", 'garagearea', "strgearea", 'factryarea', "otherarea", 'lotfront', 
                    "bldgfront", "lotdepth", 'bldgdepth', 'health', "numbldgs", "assessland","assesstot",'exemptland', 
                    'exempttot', 'builtfAR', 'residfAR', 'commfAR', 'facilfAR', 'price']:
        print("feature:"+feature)
        trend(feature, data)
        plt.show()

if __name__ == '__main__':
    path = Config.data_path
    mapdata = pd.read_csv(path + "brooklyn_sales_map.csv")
    
    # generate features
    all_data = feature_generate(mapdata, time = 'yearly')
    
    # draw trend of all features
    data_trend = all_data.copy()
    trend_draw(data_trend, mapdata)

    # calculate the correlation of the features
    print("----------------------------------------------------")
    print("Correlation pair")
    for a in all_data.columns[:-1]:
        for b in all_data.columns[:-1]:
            corr = np.corrcoef(all_data[a],all_data[b])
            if((corr[0][1]>0.5) & (a!=b)):
                print(a+"   "+b)
                print(corr[0][1])
    
    # show the highly correlated group
    print("----------------------------------------------------")
    print("Correlation group")
    for group in Config.correlated_group:
        corr = all_data[group].corr()
        print(corr)
    
    # select features
    all_data = feature_selection(all_data)
    
    # build train and test sets
    data_split(all_data, path)
    
    
    