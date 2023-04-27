
import numpy as np
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


# Evaluation Methods
def plotresults(y_test, y_pred_test, tittle):
    x_axis=range(len(y_test))
    plt.figure(figsize=(5,5))
    plt.plot(x_axis, y_test, linewidth=1, label="Actual Values" )
    plt.plot(x_axis, y_pred_test, linewidth=1.1, label="Predicted Values")
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.xlabel('Number of Trips', weight='bold', size=8)
    plt.ylabel('Trip Distance', weight='bold',size=8)
    plt.title(tittle,weight='bold',size=10)
    plt.xticks(weight='bold', size=7)
    plt.yticks(weight='bold', size=7)
    plt.grid(True)
    plt.show()
        

def calulate_metrics(y_test, y_pred_test):
    mse=  format(mean_squared_error(np.array(y_test), y_pred_test, squared=False),'.4f')
    rmse = format(np.sqrt(mean_squared_error(np.array(y_test), y_pred_test, squared=False)), '.4f')
    mae= format(mean_absolute_error(np.array(y_test), y_pred_test),'.4f')
    r2=format(r2_score(y_test,y_pred_test),'.4f')
    return mse, rmse, mae, r2

def evaluate_metrics(y_test, X_test,regressor_model, tittle, timetaken):
    y_pred_test=regressor_model.predict(X_test)
    mse, rmse,mae,r2=calulate_metrics(np.round(y_test,6), np.round(y_pred_test,6))
    print('Y pred Mean',round(y_pred_test.mean(),6))
    print('MSE',mse)
    print('RMSE', rmse)
    print('MAE', mae)
    print('R2',r2)
    print('timetaken', timetaken)
    results={}
    results['Model Name']=tittle
    results['Y Mean']=y_test.mean()
    results['Y Pred Mean']=y_pred_test.mean()
    results['MSE']=mse
    results['RMSE']=rmse
    results['MAE']=mae
    results['R2']=r2
    results['Excution Time']=timetaken
  
  
def DistributionPlot(y_test, y_pred, Title):
    plt.figure(figsize=(4, 4))
    ax1 = sns.distplot(y_test, label='Actual values',hist=False )
    ax2 = sns.distplot(y_pred,  label='Predicted Values',hist=False)
    plt.title(Title,weight='bold', size=10)
    plt.xlabel('Trip Distance', weight='bold', size=8)
    plt.ylabel('Density',weight='bold', size=8)
    plt.xticks( weight='bold', size=7)
    plt.yticks( weight='bold', size=7)
    plt.legend()
    plt.show()
    plt.close()
   
def featureImportance(X_train,model):
    plt.figure(figsize=(5,5))
    plt.title('Random Forest Feature Importance', weight='bold', size=10)
    plt.xlabel('Feature Importance', weight='bold', size=8)
    plt.ylabel("Feature Names", weight='bold', size=8)
    plt.xticks(weight='bold', size=7)
    plt.yticks(weight='bold', size=7)
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
    filename=model+"featureImportanceplot.jpg"
    plt.savefig('../reports/'+filename)