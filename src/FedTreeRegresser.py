from random import shuffle
from tokenize import Ignore
import pandas as pd
from sklearn.model_selection import train_test_split
import os.path
from fedtree import FLRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold as skfold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import plotly.express as px
from scipy import interpolate
from scipy.interpolate import make_interp_spline, BSpline
import seaborn as sns
import numpy as np
import time 
from datetime import date 
def read_data():
# 1. CovType data 
    """  colnames=['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon',
        'Hillshade_3pm','Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_1','Wilderness_Area_2',
        'Wilderness_Area_3','Wilderness_Area_4']
    for i in range(1,41):
        colnames.append("Soil_Type_"+str(i))
        colnames.append('Label')

    df = pd.read_csv('../dataset/covtype.data.gz', compression='gzip',
            error_bad_lines=False,names=colnames) """

    #2. Poker data
    # data_path="../dataset/poker"
    # poker_train=pd.read_csv(data_path+"/poker-hand-training-true.data",header=None,names=['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label'])
    # poker_test=pd.read_csv(data_path+"/poker-hand-testing.data",header=None,names=['S1', 'C1','S2', 'C2','S3', 'C3','S4', 'C4','S5', 'C5','Label'])
    # df=pd.concat([poker_train,poker_test],axis=0)
    # X=df.copy().iloc[:,:-1]
    # y=df.copy().iloc[:,-1:]  
    
    #3. Sprit Monitor
   # data_path="/home/shosamane/FedTree/FedTree/thesis/code/"
    # train_data=pd.read_csv(data_path+"/train.csv")
    # test_data=pd.read_csv(data_path+"/test.csv")
    # df=pd.concat([train_data, test_data],axis=0)
    print('current working directory', os.getcwd())
    ## data_path=os.path.join(os.getcwd(), 'preprocessed.csv')
    
    
    # train_df=pd.read_csv('./data/preprocessed/train_data.csv')
    # test_df=pd.read_csv('./data/preprocessed/test_data.csv')
    
    #Vokwagen data 
    train_df=pd.read_csv("./data/preprocessed/spritmoitor_train_data.csv", index_col=0)
    test_df=pd.read_csv("./data/preprocessed/spritmoitor_train_data.csv", index_col=0)
    train_df.drop(columns='fuel_date', axis=1, inplace=True)
    test_df.drop(columns='fuel_date', axis=1, inplace=True)
    
    train_df.rename({'trip_distance(km)':'Label'}, axis=1,inplace=True)
    test_df.rename({'trip_distance(km)':'Label'}, axis=1,inplace=True)
    
    print('train data before dropping',train_df.shape)
    print('test before dropping',test_df.shape)
    train_df.dropna(axis=0,inplace=True)
    test_df.dropna(axis=0,inplace=True)
    print('after dropping',train_df.shape)
    print('after dropping',test_df.shape)
    
    X_train=train_df.copy().drop(columns='Label', axis=1)
    y_train=train_df.copy().loc[:, 'Label']
    
    X_test=test_df.copy().drop(columns='Label', axis=1)
    y_test=test_df.copy().loc[:, 'Label']


   #PCA 
    scalled_X_train=StandardScaler().fit_transform(X_train)
    scalled_X_test=StandardScaler().fit_transform(X_test)
    pca=PCA(n_components=6)
    pca_train=pca.fit_transform(scalled_X_train)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0,shuffle=True)
    X_test_pc = pca.transform(scalled_X_test)
    return pca_train,X_test_pc,y_train,y_test 
   

def plotlinegraph(y_test, y_pred_test,X_test):
    fig1=plt.figure(figsize=(5,4))
    sns.lineplot(x=X_test['power(kW)'],y=y_test, color='red', label='y_test')
    sns.lineplot(x=X_test['power(kW)'],y=y_pred_test, color='blue', label='y_pred_test')
    plt.ylim([0,500])
    plt.xlabel('power')
    plt.ylabel('Range')
    plt.title(' Range versus power')
    plt.show()
    
def plotresults(y_test, y_pred_test):
    figure=plt.figure(figsize=(5,4))
    x_axis=range(len(y_test))
    plt.plot(x_axis, y_test, linewidth=1, label="original" )
    plt.plot(x_axis, y_pred_test, linewidth=1.1, label="predicted")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.ylabel('Range')
    plt.title("Federated Tree RF Prediction Analysis")
    plt.grid(True)
    filename="FedTree_RF_"+date.today().strftime('%d_%m_%_Y')+".png"
    plt.savefig('./reports/'+filename)
    plt.show()

    
def calulate_metrics(y_test, y_pred_test):
    mse_rf=mean_squared_error(np.array(y_test), y_pred_test, squared=False)
    rmse_rf = float(format(np.sqrt(mean_squared_error(np.array(y_test), y_pred_test, squared=False)), '.3f'))
    mae= format(mean_absolute_error(np.array(y_test), y_pred_test),'.3f')
    r2_rf=r2_score(y_test,y_pred_test)
    return mse_rf, rmse_rf, mae, r2_rf
   

def evaluate_metrics(y_test, X_test,regressor_model):
    y_pred_test=regressor_model.predict(X_test)
    mse_rf, rmse_rf,mae,r2_rf=calulate_metrics(y_test, y_pred_test)
    print('mean_squared_error',mse_rf)
    print('rmse_rf', rmse_rf)
    print('mean_absolute_error', mae)
    print('r2_rf',r2_rf)
    plotresults(y_test,y_pred_test)
    return mse_rf, rmse_rf, mae, r2_rf
    
# def plotresults(power,y_test,y_pred,X_test):
#     # y_test_new=np.linspace(y_test.min(),y_test.max(),400)
#     # y_pred_new=np.linspace(y_pred.min(),y_pred.max(),400)
    
#     # y_test_spl = make_interp_spline( power,y_test, k=3)
#     # y_pred_spl = make_interp_spline( power,y_pred, k=3)
    
#     # test_smooth=y_test_spl(y_test_new)
#     # pred_smooth=y_pred_spl(y_pred_new)
    
#     plt.plot(power, )
    
#     fig=plt.figure()
#     plt.title('Results for Fed Tree GBT')
#     plt.plot(power,test_smooth,'-',label="actual range",color='blue')
#     plt.plot(power,pred_smooth,'--',label="predictde range",color='red')
#     plt.xlabel('power')
#     plt.ylabel('range')
#     fig.savefig('RF results.png')
    
    

if __name__ == '__main__':
    X_train,X_test,y_train,y_test = read_data()
    # no_class = y_train.Label.nunique()
    parameters={'Setting Name' : ['FedTree'], 
                'Data' :["SpritMonitor"],
                'Algorithm' : ['F-RF'],
                'Model Type' : ["Regressor"], 
                'n_trees' : [100],
                'bagging':[True],
                'mean_squared_error': [0],
                'mean_absolute_error':[0],
              }
    hyperparameter={
          'n_parties':[3],
          'learning_rate':[0.01,0.1,0.2 ],
          'max_depth':[4,5,6,7,8],
          'n_trees' :[100,200],
    }
    clf = FLRegressor( mode="horizontal",objective="reg:linear",bagging = True, n_parties=3, learning_rate=0.01,n_trees=400)
    randCV=RandomizedSearchCV(estimator=clf, param_distributions=hyperparameter,n_jobs=100,cv=3,return_train_score=True)
   
    randCV.fit(X_train, y_train)
    start = time.process_time()
    #clf.fit(X_train, y_train)
    print('time.process_time() - start', time.process_time() - start)
    
        
    #df=pd.concat([pd.DataFrame(),pd.DataFrame(y_test),pd.DataFrame(y_pred)],axis=1)
    #plotresults(X_test['power(kW)'],y_test,y_pred)
   
   
    mse, rmse, mae, r2=evaluate_metrics(y_test, X_test,randCV)
   
    parameters['Mean_squared_error']=mse
    parameters['Root_Mean_squared_error']=rmse
    parameters['mean_absolute_error']=mae
    parameters['R2']=r2
    path="reg_results_range_prediction.xlsx"
    res=pd.DataFrame(parameters)
   
    # read file 
    if os.path.exists(path):
        res_df=pd.read_excel(path)
        if res_df.shape[0]>=0:
            print('file exists')
            print("shape before",res_df.shape)
            new_df=pd.concat([res_df,res])
            print("shape after",new_df.shape)
            new_df.to_excel(path,header=True,index=False)

    else:
        res.to_excel(path,header=True,index=False)
        

    
    





    
      
