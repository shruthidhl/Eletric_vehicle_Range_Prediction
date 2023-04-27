import pandas as pd 
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from modelling import decisionTree,runRandonForest
from sklearn.ensemble import RandomForestRegressor
from  lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from sklearn.metrics import r2_score
from math import sqrt, pow
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import time


def convert_encodecategorical(data_sel):
 

     train_data=pd.concat([X_train_sprit,y_train_sprit], axis=1)
     test_data=pd.concat([X_test_sprit,y_test_sprit], axis=1)
     train_data=pd.concat([X_train_sprit,y_train_sprit], axis=1)
     test_data=pd.concat([X_test_sprit,y_test_sprit], axis=1)
     train_data.dropna(axis=0, inplace=True)
     test_data.dropna(axis=0, inplace=True)
     y_train_sprit=train_data['trip_distance(km)']
     X_train_sprit=train_data.drop(columns=['trip_distance(km)'], axis=1)

     y_test_sprit=test_data['trip_distance(km)']
     X_test_sprit=test_data.drop(columns=['trip_distance(km)'], axis=1)
     
     fetures_sel1=[ 'fuel_date','quantity(kWh)', 
               'tire_type', 'city', 'motor_way', 'country_roads', 'driving_style',
               'consumption(kWh/100km)',  'A/C', 'park_heating',
                  'ecr_deviation','avg_speed(km/h)']
     X_test_sel=X_test_sprit[fetures_sel1].copy()
     X_train_sel=X_train_sprit[fetures_sel1].copy()
     train_model_cat=convert_encodecategorical(X_train_sel)
     test_model_cat=convert_encodecategorical(X_test_sel)
     
     X_train_sprit=convert_date(train_model_cat.copy())
     X_test_sprit=convert_date(test_model_cat.copy())
     
     return X_train_sprit,y_train_sprit,X_test_sprit, y_test_sprit

     
     
                
     return data_sel

def convert_to_numeric(columns,df):
    for col in columns:
        if col!='fuel_date':
           df.loc[:, col]=pd.to_numeric(df.loc[:,col], errors='coerce')
    return df

def convert_date(data_sel_num):
    data_sel_num['fuel_date']=data_sel_num['fuel_date'].astype('str')
    data_sel_num['fuel_date']=data_sel_num['fuel_date'].apply(lambda x :x.replace(".",'-'))
    data_sel_num['fuel_date_new']=pd.to_datetime(data_sel_num['fuel_date'], errors='coerce', format='%d-%m-%Y')
    # extracting month , day and day of the week from fuel_date
    data_sel_num['month']=data_sel_num['fuel_date_new'].dt.month
    data_sel_num['weekday']=data_sel_num['fuel_date_new'].dt.weekday
    data_sel_num['day']=data_sel_num['fuel_date_new'].dt.day
    data_sel_num.drop(columns=['fuel_date','fuel_date_new'], axis=1, inplace=True)
    return data_sel_num


def runforsinglecar():
    #5th Experiment Single car
    data_orgin=pd.read_csv("../data/spritmonitor/volkswagen_e_golf.csv")
    df=data_orgin.copy()
    df['trip_distance(km)']=df['trip_distance(km)'].astype('str')
    df['trip_distance(km)']=df['trip_distance(km)'].apply(lambda x: x.replace(',',''))
    df['trip_distance(km)']=df['trip_distance(km)'].apply(lambda x: x.replace(',',''))
    df['trip_distance(km)']=df['trip_distance(km)'].astype('float64')
    df=df[df['trip_distance(km)']!=0]
    y=pd.DataFrame(df['trip_distance(km)'])
    X=df.drop(columns=['trip_distance(km)'], axis=1)
    
    data_sel=data_sel.loc[(data_sel['driving_style']!='driving_style') & (data_sel['tire_type']!='tire_type'), :]
    data_sel.loc[:,'tire_type']=data_sel['tire_type'].map({'All-year tires':1,'Summer tires':3,'Winter tires':2, 'tire_type':0})
    data_sel.loc[:,'driving_style']=data_sel['driving_style'].map({'Fast':3,'Moderate':2,'Normal':1})
    X_train_sprit, X_test_sprit, y_train_sprit, y_test_sprit=train_test_split(X, y, test_size=0.4, shuffle=True)
  
    X_train_sprit,y_train_sprit, X_test_sprit,y_test_sprit=convert_encodecategorical(X_train_sprit, X_test_sprit, y_train_sprit, y_test_sprit)
    
    decisionTree(X_train_sprit,y_train_sprit, X_test_sprit,y_test_sprit, 'base')
    runRandonForest(X_train_sprit,y_train_sprit, X_test_sprit,y_test_sprit, 'base')
    


