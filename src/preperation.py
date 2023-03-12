import os
import pandas as pd 
import numpy as np
import os.path as path
import glob 

#class for reading multiple data files and converting into single file
class Preperation: 
    def read_files():
        currentdir=os.getcwd()
        os.chdir("/home/shosamane/FedTree/thesis/EV_Range_Pred/")
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

    def convert_to_numeric(columns, df):
        for col in columns:
            df[col]=pd.to_numeric(df[col], errors='coerce')
            
    colums_for_num_cov=['power(kW)','quantity(kWh)','consumption(kWh/100km)','trip_distance(km)','ecr_deviation','odometer']
    convert_to_numeric(colums_for_num_cov, final_df) 