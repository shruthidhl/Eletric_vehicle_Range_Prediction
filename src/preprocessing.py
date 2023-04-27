import pandas as pd 
from sklearn.model_selection import train_test_split
import pandas as pd 
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

class Preprocessing:
    
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
    
    def createnewcolumns(df):
        print(df)
        manufacturer = re.sub('\W+','', df['manufacturer'])
        model= re.sub('\W+','', df['model'])
        year= re.sub('\W+','', str(df['year']))
        version= re.sub('\W\d+','', df['version'])
        man_model_year_verion=manufacturer+' '+model+' '+year+' '+version
        return man_model_year_verion
    
    
    def dataclean_model_range(df):
        # Removed Open Ampera as it it's hybrid car 
        df=df[df['manufacturer']!='Opel']
        mistubishi_df=df.loc[df[
                    (df['man_model_year_verion']=='Mitsubishi i-MiEV 2013 i-MiEV') | 
                    (df['man_model_year_verion']=='Mitsubishi i-MiEV 2011 i-MiEV') & 
                    (df['trip_distance(km)']<=130)].index,:]
        vokls_2014_df=df.loc[df[
                (df['man_model_year_verion']=='Volkswagen Golf 2014 e-Golf')  & 
                (df['trip_distance(km)']<=145)].index,:]
        kia_2014_df=df.loc[df[
            (df['man_model_year_verion']=='Kia Soul 2016 Kia Soul EV')  & 
            (df['trip_distance(km)']<=165)].index,:]
        hyudnai_2018_df=df.loc[df[
        (df['man_model_year_verion']=='Hyundai IONIQ 2018 electric')  & 
        (df['trip_distance(km)']<=285)].index,:]
        vokls_2015_df=df.loc[df[
                (df['man_model_year_verion']=='Volkswagen Golf 2015 e-Golf')  & 
                (df['trip_distance(km)']<=135)].index,:]
        vokls_2018_df=df.loc[df[
                    (df['man_model_year_verion']=='Volkswagen Golf 2018 e-Golf')  & 
                    (df['trip_distance(km)']<=280)].index,:]
        
        hyudnai_IONIQ_2018_df=df.loc[df[
        (df['man_model_year_verion']=='Hyundai IONIQ 2018 Roadrunner')  & 
        (df['trip_distance(km)']<=280)].index,:]
        
        Mercedez_2015_df=df.loc[df[
        (df['man_model_year_verion']=='Mercedes-Benz B-Klasse 2015 B250e')  & 
        (df['trip_distance(km)']<=220)].index,:]

        
        tesla_2015_df=df.loc[df[
        (df['man_model_year_verion']=='Tesla Motors Model S 2015 Model S85')  & 
        (df['trip_distance(km)']<=402)].index,:]
        
        vokls_2016_df=df.loc[df[
                    (df['man_model_year_verion']=='Volkswagen Golf 2016 e-Golf')  & 
                    (df['trip_distance(km)']<=145)].index,:]
        
        nisaan_2013_df=df.loc[df[
                    (df['man_model_year_verion']=='Nissan Leaf 2013 Leaf ZE0')  & 
                    (df['trip_distance(km)']<=153)].index,:]
        
        mistubishi_2014_df=df.loc[df[
                    (df['man_model_year_verion']=='Mitsubishi i-MiEV 2014 i-MiEV') & 
                    (df['trip_distance(km)']<=100)].index,:]
        tesla_2017_df=df.loc[df[
        (df['man_model_year_verion']=='Tesla Motors Model S 2017 75D')  & 
        (df['trip_distance(km)']<=540)].index,:]
        
        hyudnai_IONIQ_2017_df=df.loc[df[
        (df['man_model_year_verion']=='Hyundai IONIQ 2017 Premium')  & 
        (df['trip_distance(km)']<=280)].index,:]
        
        mistubishi_2011_df=df.loc[df[
                    (df['man_model_year_verion']=='Mitsubishi i-MiEV 2011 i-MiEV') & 
                    (df['trip_distance(km)']<=135)].index,:]

        nissan_leaf_df=df.loc[df[
        (df['man_model_year_verion']=='Nissan Leaf 2017 Black Edition')  & 
        (df['trip_distance(km)']<=155)].index,:]
        mistubishi_df
        final_df=pd.DataFrame(pd.concat([mistubishi_df, vokls_2014_df, kia_2014_df, hyudnai_2018_df,vokls_2015_df, 
                                        nisaan_2013_df, mistubishi_2014_df, tesla_2017_df, 
                                        hyudnai_IONIQ_2017_df, mistubishi_2011_df,nissan_leaf_df], axis=0))
        return df 

       
    def cleanPowerConsumption(df):
      
        ########################################################
        voks_df_2017=df[df['man_model_year_verion']=='Volkswagen Golf 2018 e-Golf']
        filtered_voks_df_2017=voks_df_2017[(voks_df_2017['consumption(kWh/100km)']>=11.4) &
                                        ( voks_df_2017['consumption(kWh/100km)']<=23.7)]
        
        tesla_models_df_2017=df[df['man_model_year_verion']=='TeslaMotors ModelS 2017 75D']
        filtered_tesla_models_df_2017=tesla_models_df_2017[(tesla_models_df_2017['consumption(kWh/100km)']>=13.4) &
                                                    ( tesla_models_df_2017['consumption(kWh/100km)']<=26.7)]
        
        hyundai_ioniq_df_2018=tesla_models_df_2017=df[df['man_model_year_verion']=='Hyundai IONIQ 2018 Roadrunner']
        filtered_hyundai_ioniq_df_2018=hyundai_ioniq_df_2018[(hyundai_ioniq_df_2018['consumption(kWh/100km)']>=9.8) &
                                                ( hyundai_ioniq_df_2018['consumption(kWh/100km)']<=20.7)]
        
        kia_soul_df_2016=df[df['man_model_year_verion']=='Kia Soul 2016 Kia Soul EV']
        filtered_kia_soul_df_2016=kia_soul_df_2016[(kia_soul_df_2016['consumption(kWh/100km)']>=11.0) &
                                                ( kia_soul_df_2016['consumption(kWh/100km)']<=23.5)]
        ########################################################
        voks_df_2015=df[df['man_model_year_verion']=='Volkswagen Golf 2015 e-Golf']
        filtered_voks_df_2015=voks_df_2015[ ( voks_df_2015['consumption(kWh/100km)']<=12.7)]
        
        nissan_leaf_2017=df[df['man_model_year_verion']=='Nissan Leaf 2017 Black Edition']
        filtered_nissan_leaf_2017=nissan_leaf_2017[ ( nissan_leaf_2017['consumption(kWh/100km)']>=11.0) &
                                                ( nissan_leaf_2017['consumption(kWh/100km)']<=23.3)]
        
        Mitsubishi_imiev_2011=df[df['man_model_year_verion']=='Mitsubishi iMiEV 2011 i-MiEV']
        filtered_Mitsubishi_imiev_2011=Mitsubishi_imiev_2011[ ( Mitsubishi_imiev_2011['consumption(kWh/100km)']>=9.8) &
                                                    Mitsubishi_imiev_2011['consumption(kWh/100km)']<=20.7]
        ########################################################
        voks_df_2016_df=df[df['man_model_year_verion']=='Volkswagen Golf 2014 e-Golf']
        filtered_voks_df_2016=voks_df_2016_df[( voks_df_2016_df['consumption(kWh/100km)']<=14.8)]
    
        
        Mitsubishi_imiev_2014=df[df['man_model_year_verion']=='Mitsubishi iMiEV 2014 i-MiEV']
        filtered_Mitsubishi_imiev_2014=Mitsubishi_imiev_2014[ ( Mitsubishi_imiev_2014['consumption(kWh/100km)']>=10.7) &
                                                        Mitsubishi_imiev_2014['consumption(kWh/100km)']<=24.2]
        
        nissan_leaf_2013=df[df['man_model_year_verion']=='Nissan Leaf 2013 Leaf ZE0']
        filtered_nissan_leaf_2013=nissan_leaf_2013[nissan_leaf_2013['consumption(kWh/100km)']<=17.3]
        
        ########################################################
        Hyundai_IONIQ_2017_eletrctric =df[df['man_model_year_verion']=='Hyundai IONIQ 2017 Premium']
        filtered_Hyundai_IONIQ_2017_eletrctric =Hyundai_IONIQ_2017_eletrctric [ ( Hyundai_IONIQ_2017_eletrctric ['consumption(kWh/100km)']>=9.8) ]
        

    
        Mitsubishi_imiev_2013=df[df['man_model_year_verion']=='Mitsubishi iMiEV 2013 i-MiEV']
        filtered_Mitsubishi_imiev_2013=Mitsubishi_imiev_2013[ ( Mitsubishi_imiev_2013['consumption(kWh/100km)']>=9.7) &
                                                        Mitsubishi_imiev_2013['consumption(kWh/100km)']<=24.7]
        
        mercedesBenz_bKlasse_2015_b250e=df[df['man_model_year_verion']=='MercedesBenz BKlasse 2015 B250e']
        filtered_mercedesBenz_bKlasse_2015_b250e=mercedesBenz_bKlasse_2015_b250e[ ( mercedesBenz_bKlasse_2015_b250e['consumption(kWh/100km)']>=10.7) &
                                                        mercedesBenz_bKlasse_2015_b250e['consumption(kWh/100km)']<=24.7]
        ########################################################
        
        tesla_models_df_2017=df[df['man_model_year_verion']=='TeslaMotors ModelS 2015 Model S85']
        filtered_tesla_models_df_2017=tesla_models_df_2017[(tesla_models_df_2017['consumption(kWh/100km)']>=13.4) &
                                                ( tesla_models_df_2017['consumption(kWh/100km)']<=26.7)]
        
        final_df=pd.DataFrame(pd.concat([
                                        voks_df_2017,    tesla_models_df_2017,   hyundai_ioniq_df_2018,      kia_soul_df_2016, 
                                        voks_df_2015,    nissan_leaf_2017,        Mitsubishi_imiev_2011,     voks_df_2016_df, 
                                        Mitsubishi_imiev_2014,      nissan_leaf_2013,  Hyundai_IONIQ_2017_eletrctric, 
                                        Mitsubishi_imiev_2013,      mercedesBenz_bKlasse_2015_b250e,  tesla_models_df_2017 ],axis=0))
        
        return final_df



    
    def preprocessing(df):
        prepro=Preprocessing(df)
        df=df.loc[df[df['manufacturer']!='Opel'].index, :]
        df=df.loc[df[df['manufacturer']!='manufacturer'].index,:]
        
        df['year']=df['year'].astype('int')
        df['man_model_year_verion']=df.apply(lambda x: prepro.createnewcolumns(x), axis=1)
        
        # Clean data 
        cleaned_df=prepro.dataclean_model_range(df)
        
        cleaned_df_cal=cleaned_df[cleaned_df['trip_distance(km)']!=0]
        
        df=prepro.cleanPowerConsumption(cleaned_df_cal)
        
        
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