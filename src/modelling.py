import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from  lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA 
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.metrics import r2_score
from math import sqrt, pow
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
# from tensorflow import keras
# from scikeras.wrappers import KerasRegressor
# check xgboost version
import time
import os
from evaluation import plotresults, calulate_metrics, evaluate_metrics

   
def decisionTree(X_train, y_train, X_test, y_test, flag):
    if flag=='base':
        start_time = time.time()
        dectreeRegModel=DecisionTreeRegressor()
        dectreeRegModel.fit(pd.DataFrame(X_train), np.ravel(y_train))
        timetaken=(time.time() - start_time)
        y_pred_deci=evaluate_metrics(y_test,X_test, dectreeRegModel, 'Base Decission Tree results',timetaken)
        return dectreeRegModel, y_pred_deci

    else: 
        params_grid={
                'max_depth':[ i for i in range(1,10)],
                'min_samples_leaf': [ i for i in range(1,8)],
                'min_samples_split': [ i for i in range(2,10)],
                'criterion': ['squared_error','absolute_error'],
                'splitter':['best', 'random'],
                'max_features':['sqrt','log2']}
            
        skfold= StratifiedKFold(n_splits=5, shuffle=True)
        start_time = time.time()
        dec_cv=DecisionTreeRegressor()
        dec_random = RandomizedSearchCV(estimator = dec_cv, param_distributions = params_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
        #rf_random.fit(pca_train[:,:6],np.ravel(y_train))
        dec_random.fit(X_train,np.ravel(y_train))
        timetaken= time.time()-start_time
        y_pred=evaluate_metrics(y_test,X_test,dec_random, 'Tuned DecisionTree',timetaken)
            #export_graphviz(dectreeRegModel, out_file ='../reports/tree_structure.dot', 
        #        feature_names =X_train_sprit.columns)
        return dec_random, y_pred,timetaken
    

def runRandonForest(X_train, y_train, X_test, y_test, flag):
    if flag=='base':
        start_time = time.time()
        rf=RandomForestRegressor(n_estimators=100)
        rf_model=rf.fit(X_train,np.ravel(y_train))
        timetaken=  time.time()-start_time
        y_pred_rf_base=evaluate_metrics(y_test,X_test,rf_model, 'Base RandomForest results',timetaken)
        return rf_model,y_pred_rf_base
    else:
        params_grid={
            'n_estimators': [i for i in range(1,150)],
            'bootstrap': [True, False],
            'max_depth':[ i for i in range(1,14)],
            'min_samples_leaf': [ i for i in range(1,10)],
            'min_samples_split': [ i for i in range(2,10)],
            'criterion': ['squared_error'] }
        start_time=time.time()
        rf_cv=RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf_cv, param_distributions = params_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
        #rf_random.fit(pca_train[:,:6],np.ravel(y_train))
        rf_model=rf_random.fit(X_train,np.ravel(y_train))
        timetaken=time.time()-start_time
        y_prd=evaluate_metrics(y_test,X_test,rf_model, 'RandomForest Tree results',timetaken)
        return rf_model, y_prd, timetaken
    
def  runXGB(X_train, y_train, X_test, y_test, flag):
        if flag=="base":
            # Instantiation
            start_time = time.time()
            params = {'objective':'reg:squarederror' } 
            xgb_r = xgb.XGBRegressor(**params)
            # Fitting the model
            #xgb_r.fit(pca_train[:,:6],np.ravel(y_train))
            xgb_r.fit(X_train,np.ravel(y_train))
            timetaken= time.time()-start_time 
            y_pred=evaluate_metrics(y_test,X_test,xgb_r,'Base XGB Analysis',timetaken)
            return xgb_r,y_pred
        else:
            skf = StratifiedKFold(n_splits=5,
                        shuffle=True, 
                        random_state=0)
            space={   
                'objective':['reg:squarederror'],
                'max_depth': [1, 2,3,4,5,6,7,8,9,10],
                'n_estimators': [i for i in range(1,200)],
                    'learning_rate':[0.01,0.1, 1.0] ,
                    'subsample':[0.5,.6,.7,0.8],  
                    'eta':[0.1],
                    'lambda':[0.01],
                    'booster':['gbtree']
                }
            start_time=time.time()
            xgb_r_tun = xgb.XGBRegressor( )
            xgb_cv = RandomizedSearchCV(estimator=xgb_r_tun,param_distributions=space, cv=5,n_iter=100)
            #xgb_cv=xgb_cv.fit(pca_train[:,:6], y_train)
            xgb_cv.fit(X_train, y_train)
            timetaken=  time.time()-start_time
            y_pred_test=evaluate_metrics(y_test_sprit,X_test_sprit,xgb_cv,'XGB Tuned',timetaken )
            return xgb_cv,y_pred_test,timetaken
        
def runLightGBM(X_train, y_train, X_test, y_test, flag):
    if flag=='base':
        starttime=time.time()
        lgbm_model=LGBMRegressor( objective = "regression")
        #lgbm_model.fit(pca_train[:,:6],np.ravel(y_train))
        lgbm_model.fit(X_train,np.ravel(y_train))
        timetaken= time.time()-starttime
        y_pred_test=evaluate_metrics(y_test, X_test, lgbm_model, 'Base LightGBM Results',timetaken)
        return y_pred_test, lgbm_model,timetaken
    
    else:
        skf = StratifiedKFold(n_splits=5,
                    shuffle=True, 
                    random_state=0)
        space = {
            "objective" : ["regression"],
            "metric" :[ "mae"],
            'boosting_type': ['gbdt'],
        
            "max_depth": [5, 6,7, 8],
            "learning_rate" : [0.005, 0.01, 0.02, 0.1],
            'min_child_samples': [2,5,10,20],
            'n_estimators': [i for i in range(50,500)],
            'num_leaves': [i for i in range(20,50)],
            'min_child_weight':[ 0.001, 0.002,0.003],
            "bagging_seed" :[ 42]

        }    
        lgbm_model=LGBMRegressor( objective = "regression")
        start_time=time.time()
        lgbm_cv = RandomizedSearchCV(estimator=lgbm_model,param_distributions=space, cv=5,n_iter=100)
        lgbm_cv=lgbm_cv.fit(X_train, y_train)
        timetaken=time.time()-start_time
        y_pred=evaluate_metrics(y_test, X_test, lgbm_cv, 'LightGBM Results', timetaken)
        return lgbm_cv, y_pred,timetaken

 


