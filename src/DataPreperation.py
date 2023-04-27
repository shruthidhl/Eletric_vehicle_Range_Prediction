import os
import pandas as pd 
import numpy as np
import os.path as path
import glob 


def read_files():
    currentdir=os.getcwd()
    os.chdir("/home/shruthi/workenv/dataanalysis-shruthi")
    file_path=os.path.join(os.getcwd(), "data/raw/*.csv")
    print(file_path)
    all_files=glob.glob(file_path)
    print(len(all_files))
    final_df=pd.DataFrame()
    df=pd.DataFrame()
    filesnames_list=glob.glob(file_path)
    print(len(filesnames_list))
    #df=pd.concat((pd.read_csv(f, header=0) for f in filesnames_list), axis=0,ignore_index=True,join='inner',)
    final_df=pd.read_csv(all_files[0],header=0,parse_dates=False)
    for i in range(1,len(filesnames_list)):
        filename=all_files[i]
        df=pd.read_csv(filename,parse_dates=True)
    
        filname_split=filename.split('/')
        final_df= pd.concat([final_df,df],ignore_index=True,axis=0)
        final_df.drop(axis=0,labels=np.arange(13),inplace=True)
        final_df['user_id']=filname_split[-1].split(".")[0]
    return final_df


final_df=read_files()


# data_sample=final_df[['power(kW)', 'fuel_date',
#        'odometer', 'trip_distance(km)', 'quantity(kWh)', 'fuel_type',
#        'tire_type', 'city', 'motor_way', 'country_roads', 'driving_style',
#        'consumption(kWh/100km)', 'A/C', 'park_heating', 'avg_speed(km/h)',
#        'ecr_deviation', 'fuel_note', 'user_id']]


#Data type conversion
def convert_to_numeric(columns, df):
    for col in columns:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    return df
        
colums_for_num_cov=['power(kW)','quantity(kWh)','consumption(kWh/100km)','trip_distance(km)','ecr_deviation','odometer']
final_df=convert_to_numeric(colums_for_num_cov, final_df) 

# Shuffling data 
final_df=final_df.sample(frac=1)
final_df.to_csv(os.getcwd()+'/data/prepared/prearedData.csv')