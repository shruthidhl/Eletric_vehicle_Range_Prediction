import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
from  sklearn.model_selection import train_test_split


def read_data():
    train_df_sprit=pd.read_csv("../data/preprocessed/df_range_consumption_train_data.csv", index_col=0)
    test_df_sprit=pd.read_csv("../data/preprocessed/df_range_consumption_test_data.csv", index_col=0)
    train_data=train_df_sprit.copy()
    test_data=test_df_sprit.copy()
    
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    train_data.drop(columns='index', axis=1,inplace=True)
    test_data.drop(columns='index', axis=1,inplace=True)
    # #X_train=data.drop(columns=['trip_distance(km)', 'avg_speed(km/h)'],axis=1)
    # X_train_sprit=train_data.drop(columns=['trip_distance(km)'],axis=1)
    # y_train_sprit=train_data['trip_distance(km)']
    # #X_test=test_data.drop(columns=['trip_distance(km)', 'avg_speed(km/h)'],axis=1)
    # X_test_sprit=test_data.drop(columns=['trip_distance(km)'],axis=1)
    # y_test_sprit=test_data['trip_distance(km)']
    
    return train_data, test_data

def preprocessForFedTree(train_df, test_df, type):
  
       #rename target variable 
       # change target variable name to y
       train_df.rename({'trip_distance(km)':'y'}, axis=1,inplace=True)
       test_df.rename({'trip_distance(km)':'y'}, axis=1,inplace=True)
       
       # reshuffle columns to put target variable first
       columns=['y','power(kW)', 'quantity(kWh)', 'odometer', 'consumption(kWh/100km)',
       'ecr_deviation',  'month', 'weekday', 'day', 'tire_type', 'city', 
       'motor_way', 'country_roads', 'driving_style', 'A/C', 'park_heating']
       ## adding labels column for kmeans clustering
       if type=='kmeans':
         columns.append('labels')
         
       train_df=train_df.loc[:, columns]
       test_df=test_df.loc[:, columns]

       #creating id column
       train_df.reset_index(inplace=True)
       test_df.reset_index(inplace=True)

       train_df.rename({'index':'id'},axis=1, inplace=True)
       test_df.rename({'index':'id'},axis=1, inplace=True)
       
       return train_df, test_df
       

def Kmeandata(train_data, test_data):
    scaler=RobustScaler()
    categorical_columns=['tire_type', 'city',
       'motor_way', 'country_roads', 'driving_style',
       'A/C', 'park_heating', 'weekday','month','day']
    numeric_columns= ['power(kW)', 'quantity(kWh)', 'odometer','consumption(kWh/100km)','ecr_deviation','trip_distance(km)']
    # Seperating Categorical Variables
    train_cat=train_data.loc[:, categorical_columns].reset_index().drop(columns='index', axis=1)
    test_cat=test_data.loc[:, categorical_columns].reset_index().drop(columns='index', axis=1)
    
    # Seperating Numerical variables
    train_scalling=train_data.loc[:, numeric_columns].reset_index().drop(columns='index', axis=1)
    test_scalling=test_data.loc[:, numeric_columns].reset_index().drop(columns='index', axis=1)
    
    # Scaling Numerical variables
    scalled_train=pd.DataFrame(scaler.fit_transform(train_scalling), columns=train_scalling.columns)
    scalled_test=pd.DataFrame(scaler.transform(test_scalling), columns=test_scalling.columns)
    
    #Merging Scaled Numerical and Unscaled categorical
    train_data_kmeans=pd.concat( [scalled_train, train_cat], axis=1)
    test_data_kmeans=pd.concat( [scalled_test, test_cat], axis=1)
    
    kmeans = KMeans(n_clusters=2)
    kidx=kmeans.fit(train_data_kmeans)
    labels=kmeans.labels_
    
    labels_df=pd.DataFrame(labels).rename({0:'labels'},axis=1)
    kmeans_cluster_train_df=pd.concat([train_data_kmeans, labels_df], axis=1)
    test_labels=kmeans.predict(test_data_kmeans)
    test_labels_df=pd.DataFrame(test_labels).rename({0:'labels'},axis=1)
    kmeans_cluster_test_df=pd.concat([test_data_kmeans, test_labels_df], axis=1)
    
    kmeans_train,kmeans_test= preprocessForFedTree(kmeans_cluster_train_df, kmeans_cluster_test_df, 'kmeans')
    
    kmeans_cluster_train_df_0=pd.DataFrame()
    kmeans_cluster_train_df_1=pd.DataFrame()
        
    #don't divide test data for fedtree
    kmeans_cluster_train_df_0=kmeans_train[kmeans_train['labels']==0]
    kmeans_cluster_train_df_1=kmeans_train[kmeans_train['labels']==1]
    
    #drop column in both the cluster
    kmeans_cluster_train_df_0.drop(columns='labels', axis=1, inplace=True)
    kmeans_cluster_train_df_1.drop(columns='labels', axis=1, inplace=True)
    kmeans_test.drop(columns='labels', axis=1, inplace=True)
    
    path='/home/shruthi/workenv/FedTree/dataset/spritmonitor/kmeans'
    kmeans_cluster_train_df_0.to_csv(path+'/kmeans_cluster_train_df_031523_p0.csv', index=0)
    kmeans_cluster_train_df_1.to_csv(path+'/kmeans_cluster_train_df_031523_p1.csv',  index=0)
    kmeans_test.to_csv(path+'/kmeans_cluster_test_df_031523.csv', index=0)
    
    
def drivingstyle(train_data, test_data, df):
    driving_style_train,driving_style_test= preprocessForFedTree(train_data, test_data, 'driving_style')
    train_df_fast_p0=pd.DataFrame()
    train_df_moderate_p1=pd.DataFrame()
    train_df_normal_p2=pd.DataFrame()
    
    train_df_fast_p0=driving_style_train[driving_style_train['driving_style']==1].copy()
    train_df_moderate_p1=driving_style_train[driving_style_train['driving_style']==2].copy()
    train_df_normal_p2=driving_style_train[driving_style_train['driving_style']==3].copy()
    path='/home/shruthi/workenv/FedTree/dataset/spritmonitor/kmeans'
    train_df_fast_p0.to_csv(path+"/train_df_fast_p0_03072023.csv", index=0)
    train_df_moderate_p1.to_csv(path+"/train_df_moderate_p1_03072023.csv", index=0)
    train_df_normal_p2.to_csv(path+'/train_df_normal_p2_03072023.csv', index=0)
    driving_style_test.to_csv(path+'/test_converted_distributed.csv', index=0)
    
    
def consumption(train_data, test_data,df):
    train_df, test_df=preprocessForFedTree(train_data, test_data, 'consumption')
    train_df_p0=train_df[train_df['consumption(kWh/100km)']<=14.0].copy()
    train_df_p1=train_df[(train_df['consumption(kWh/100km)']>14.0) & (train_df['consumption(kWh/100km)']<=20.0)].copy()
    train_df_p2=train_df[train_df['consumption(kWh/100km)']>20.0 ].copy()
    path='/home/shruthi/workenv/FedTree/dataset/spritmonitor/consumption'
    train_df_p0.to_csv(path+"/train_df_p0.csv", index=0)
    train_df_p1.to_csv(path+"/train_df_p1.csv", index=0)
    train_df_p2.to_csv(path+'/train_df_p2.csv', index=0)
    
    test_df_train=test_df.iloc[:3000].copy()
    test_df_scope=test_df.iloc[3000:].copy()
    test_df_train.to_csv(path+"/test_df_train.csv", index=0)
    test_df_scope.to_csv(path+"/test_df_scope.csv", index=0)

    
    
def splittestdata(test_df):
    test_df_temp=test_df.copy()
    test_df_temp=test_df_temp.sample(frac = 1, random_state = 200)
    test_df_val=test_df.iloc[:test_df.shape[0]//2, :]
    test_df_pred=test_df.iloc[test_df.shape[0]//2:, :]
    return test_df_val, test_df_pred

