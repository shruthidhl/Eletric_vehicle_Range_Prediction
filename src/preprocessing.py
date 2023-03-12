import pandas as pd 
from sklearn.model_selection import train_test_split

class Preprocesssing:
    
    def __init__(self, df):
        self.df=df
        
        
    def convert_encodecategorical(data_sel):
        data_sel=data_sel.loc[(data_sel['driving_style']!='driving_style') & (data_sel['tire_type']!='tire_type'), :]
        data_sel.loc[:,'tire_type']=data_sel['tire_type'].map({'All-year tires':1,'Summer tires':3,'Winter tires':2, 'tire_type':0})
        data_sel.loc[:,'driving_style']=data_sel['driving_style'].map({'Fast':1,'Moderate':2,'Normal':3})
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
    
    def fillinMissingvalues(df, columns):
        for col in columns:
            print(df[col].dtypes)
            if df[col].dtypes=='float64':
                df[col].fillna(value=df[col].mean(), inplace=True)
            elif df[col].dtypes=='object':
                df[col].fillna(value=df[col].astype('str').mode(), inplace=True)
            elif df[col].dtypes=='int64':
                df[col].fillna(value=df[col].mode(), inplace=True)
        return df
    
    def preprocessing(df):
        y=pd.DataFrame(df['trip_distance(km)'])
        X=df.drop(columns='trip_distance(km)', axis=1)
        X_train_model, X_test_model, y_train_model, y_test_model=train_test_split(X, y, test_size=0.4, shuffle=True)
        train_data=pd.concat([X_train_model,y_train_model], axis=1)
        test_data=pd.concat([X_test_model,y_test_model], axis=1)

        fetures_sel1=[ 'power(kW)', 'fuel_date', 'quantity(kWh)', 
            'tire_type', 'city', 'motor_way', 'country_roads', 'driving_style',
                'park_heating', 'avg_speed(km/h)',
            'ecr_deviation','consumption(kWh/100km)','trip_distance(km)']
        train_model_sel=train_data[fetures_sel1]
        test_model_sel=test_data[fetures_sel1]
        
        ## Fill Missing value 
        columns=['tire_type', 'driving_style','consumption(kWh/100km)','ecr_deviation',] 
        
        prepro=Preprocesssing(df)
        train_data=prepro.fillinMissingvalues(train_model_sel, columns)               
        test_data=prepro.fillinMissingvalues(test_model_sel, columns)
        
        ## data type conversion 
        
        # To Numeric 
        colums_for_num_cov=['power(kW)','quantity(kWh)','avg_speed(km/h)','ecr_deviation']
        train_model_num=prepro.convert_to_numeric(colums_for_num_cov, train_model_sel)
        test_model_num=prepro.convert_to_numeric(colums_for_num_cov, test_model_sel)
        
        # To categorical 
        train_model_cat=prepro.convert_encodecategorical(train_model_num)
        test_model_cat=prepro.convert_encodecategorical(test_model_num)
        
        
        # data to month, day and weekday
        train_converted=prepro.convert_date(train_model_cat.copy())
        test_converted=prepro.convert_date(test_model_cat.copy())
        return train_converted, test_converted