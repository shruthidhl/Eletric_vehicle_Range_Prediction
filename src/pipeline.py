    

import os
import pandas as pd 
import numpy as np
import os.path as path
from modelling import decisionTree, runRandonForest, runXGB, runLightGBM
from preparation import Preparation
from preprocessing import Preprocessing
from fedTreeRegressor import runfedtree
from personalization import Kmeandata,drivingstyle,consumption
from singlecar import runforsinglecar

def read_data():
    
    
    #Exp 1
    # train_df_sprit=pd.read_csv("../data/preprocessed/train_converted.csv", index_col=0)
    # test_df_sprit=pd.read_csv("../data/preprocessed/test_converted.csv", index_col=0)

    #Exp 2 
    # train_df_sprit=pd.read_csv("../data/preprocessed/train_converted_experiment2.csv", index_col=0)
    # test_df_sprit=pd.read_csv("../data/preprocessed/test_converted_experiment2.csv", index_col=0)

    #Experiment 3
    train_df_sprit=pd.read_csv("../data/preprocessed/df_range_consumption_train_data_030923.csv", index_col=0)
    test_df_sprit=pd.read_csv("../data/preprocessed/df_range_consumption_test_data_030923.csv", index_col=0)
    train_data=train_df_sprit.copy()
    test_data=test_df_sprit.copy()
    
    #X_train=data.drop(columns=['trip_distance(km)', 'avg_speed(km/h)'],axis=1)
    X_train_sprit=train_df_sprit.drop(columns=['trip_distance(km)'],axis=1)
    y_train_sprit=train_df_sprit['trip_distance(km)']
    #X_test=test_data.drop(columns=['trip_distance(km)', 'avg_speed(km/h)'],axis=1)
    X_test_sprit=test_df_sprit.drop(columns=['trip_distance(km)'],axis=1)
    y_test_sprit=test_df_sprit['trip_distance(km)']
        
    return X_train_sprit, y_train_sprit, X_test_sprit,y_test_sprit

if __name__=="__main__":
    
    # 1. Data Preperation
    prep=Preparation()
    df=prep.read_data()
    
    # 2. Preprocessing
    preprocess=Preprocessing(df)
    train_df, test_df=preprocess.preprocessing(df)
    
    #3. Test and Train split
    X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit=read_data()
    
    #5 Modelling
    decisionTree(X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit, 'base')
    runRandonForest(X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit, 'base')
    runXGB(X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit, 'base')
    runLightGBM(X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit, 'base')
    
    #fedTree
    runfedtree(X_train_sprit, y_train_sprit, X_test_sprit, y_test_sprit, 'base')
    train_df=pd.concat([X_train_sprit, y_train_sprit], axis=1)
    test_df=pd.concat([X_test_sprit, y_test_sprit], axis=1)
    
    #6 personalization 
    personal=Kmeandata(train_df,test_df)
    
    #7 Single car
    runforsinglecar()
        
    
    
    