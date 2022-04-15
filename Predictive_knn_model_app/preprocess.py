import pandas as pd
import numpy as np
from sklearn import preprocessing
from pickle import load
import warnings
from pickle import dump,load

warnings.filterwarnings('ignore')

class Preprocessor:
    def __init__(self):
        pass


    def precess(self,df):
        # sort values by time
        df = df.sort_values(by='DOWNTIME')
        print('df shape at reading time',df.shape)
        
        # remove first row of dataframe
        df = df.iloc[1: , :]
        # replace nan value with null
        df = df.replace(np.nan,None)

        # convert boolean object in to integer
        df['GREDE10890_FIX_TAKE_DATA_A'] =  df['GREDE10890_FIX_TAKE_DATA_A'].astype(int)
        df['GREDE10890_FIX_TAKE_DATA_B'] =  df['GREDE10890_FIX_TAKE_DATA_B'].astype(int)
        df['GREDE10890_FIX_TRG_M_V'] =  df['GREDE10890_FIX_TRG_M_V'].astype(int)

        # remove columns whoes type is datetime
        df = df.drop(columns=['UTC_TIMESTAMP','LOCAL_TIMESTAMP'])

        # select only integer columns from data
        df = df.select_dtypes(include='number')

        return df

    def create_status_feature(self,df):
        print('df at create_status_feature shape',df.shape)
        df['failure_SANDS_MCC3_PLC_PP_TEMP_ACTUAL'] = np.where(((df.SANDS_MCC3_PLC_PP_TEMP_ACTUAL < 1000.0)|(df.SANDS_MCC3_PLC_PP_TEMP_ACTUAL >1500.0)), 'fail','pass')
        df['failure_GREDE10890_MAIN__DB_029_MA_6'] = np.where(((df.GREDE10890_MAIN__DB_029_MA_6 < 85.0)|(df.GREDE10890_MAIN__DB_029_MA_6 >115.0) ), 'fail','pass')
        df['failure_GREDE10890_MAIN__06_9_I_CH0DATA'] = np.where(((df.GREDE10890_MAIN__06_9_I_CH0DATA < 0.0)|(df.GREDE10890_MAIN__06_9_I_CH0DATA >0.8) ), 'fail','pass')
        df['failure_GREDE10890_MAIN__DB_029_MA_31'] = np.where(((df.GREDE10890_MAIN__DB_029_MA_31 < 65.0)|(df.GREDE10890_MAIN__DB_029_MA_31 >115.0) ), 'fail','pass')
        df['failure_GREDE10890_MAIN_PPM_AGUA_EN_ACEITE'] = np.where(((df.GREDE10890_MAIN_PPM_AGUA_EN_ACEITE < 0)|(df.GREDE10890_MAIN_PPM_AGUA_EN_ACEITE >800.0) ),'fail','pass')
        df = df.replace({'pass':0 , 'fail':1})
        df['status'] = df['failure_SANDS_MCC3_PLC_PP_TEMP_ACTUAL'] + df['failure_GREDE10890_MAIN__DB_029_MA_6'] + df['failure_GREDE10890_MAIN__06_9_I_CH0DATA']+ df['failure_GREDE10890_MAIN__DB_029_MA_31']+ df['failure_GREDE10890_MAIN_PPM_AGUA_EN_ACEITE']

        df['status'] = df['status'].replace([1, 2, 3], 1)
        
        df = df.drop(columns=['failure_SANDS_MCC3_PLC_PP_TEMP_ACTUAL',
                                'failure_GREDE10890_MAIN__DB_029_MA_6',
                                'failure_GREDE10890_MAIN__06_9_I_CH0DATA',
                                'failure_GREDE10890_MAIN__DB_029_MA_31',
                                'failure_GREDE10890_MAIN_PPM_AGUA_EN_ACEITE'])

        return df

    def standerdise(self,df):
        slice_1 = df.drop(columns=['status'])
        slice_2 = df[['status']]
        cols = slice_1.columns
        # dump(cols, open('model/model_columns.pkl', 'wb'))
        print('len of scaler colums',len(cols))
        scaler  = preprocessing.MinMaxScaler()
        scaler.fit(slice_1)
        np_scaled = scaler.transform(slice_1)
        slice_1_df_normalized = pd.DataFrame(np_scaled, columns = cols)
        df = pd.concat([slice_1_df_normalized, slice_2], axis=1)
        df = df.iloc[1: , :]
        return df ,scaler

    def reverse_range(self,df):
        remain_cycle = []
        df_shape = df.shape[0]
        for index, row in df.iterrows():
            value = df_shape-1
            remain_cycle.append(value)
            df_shape -= 1
            
        return remain_cycle

    def feature_engineering(self,df):
        number = 1
        cycle = []
        for index, row in df.iterrows():
            if row['status'] == 0:
                this_cycle = row['status']+number
                cycle.append(this_cycle)
            elif row['status'] == 1:
                cycle.append(0)
                number = 0
            else:
                pass
            number += 1
        df['cycle'] = cycle

        dd = df[['status','cycle']]
        dd.reset_index()

        ends_idx = np.arange(dd.shape[0])[(dd['cycle'] == 0).values]
        ends_idx

        remain_cycle = []
        ll = dd[: ends_idx[0]]
        this_remain_cycle =  self.reverse_range(ll)
        remain_cycle.extend(this_remain_cycle)

        for i in range(len(ends_idx)):
            try:
                index_lower_limit = ends_idx[i]
                index_upper_limit = ends_idx[i+1]
                kk = dd[index_lower_limit: index_upper_limit]
                this_remain_cycle =  self.reverse_range(kk)
                remain_cycle.extend(this_remain_cycle)
            except:
                kk = dd[index_upper_limit: ]
                this_remain_cycle =  self.reverse_range(kk)
                remain_cycle.extend(this_remain_cycle)
        df['remain_cycle'] = remain_cycle

        get_this_df = df[['status','cycle','remain_cycle']]
        get_this_df['remain_cycle'] = get_this_df['remain_cycle']+1

        final_remain_cycle = []
        for index, row in get_this_df.iterrows():
            
            if row['status'] == 0:
                final_remain_cycle.append(row['remain_cycle'])
            elif row['status'] == 1:
                final_remain_cycle.append(0)
            else:
                pass
        get_this_df['final_remain_cycle'] = final_remain_cycle
        get_this_df = get_this_df[['status','cycle','final_remain_cycle']]

        max_cycle = []
        ll = get_this_df[: ends_idx[0]]
        this_remain_cycle =  ll['final_remain_cycle'].max()
        ll['remain_max_cycle'] = this_remain_cycle
        max_cycle.extend(ll['remain_max_cycle'])

        for i in range(len(ends_idx)):
            try:
                index_lower_limit = ends_idx[i]
                index_upper_limit = ends_idx[i+1]
                kk = get_this_df[index_lower_limit: index_upper_limit]
                this_remain_cycle =  kk['final_remain_cycle'].max()
                kk['remain_max_cycle'] = this_remain_cycle
                max_cycle.extend(kk['remain_max_cycle'])
            except:
                kk = get_this_df[index_upper_limit: ]
                this_remain_cycle =  kk['final_remain_cycle'].max()
                kk['remain_max_cycle'] = this_remain_cycle
                max_cycle.extend(kk['remain_max_cycle'])
        get_this_df['max_cycle'] = max_cycle

        return get_this_df

    def run(self, df):
        # preprocess data
        df = self.precess(df)
        print('precess complete')

        # create_status_feature
        df = self.create_status_feature(df)
        print('status feature creation complete')

        # starnderdise data
        df, min_max_scaler = self.standerdise(df)
        print('data  starnderdisation complete')
       
        # create some feature engineering
        get_this_df = self.feature_engineering(df)
        print('data  feature_engineering complete')

        df['final_remain_cycle'] = get_this_df['final_remain_cycle']
        df['max_cycle'] = get_this_df['max_cycle']

        df = df.reset_index(drop=True)

        return df, min_max_scaler
        

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
# from scipy.stats import mode
# from numpy import dot
# from numpy.linalg import norm
# import json

# class KNeighborsRegressor():

#     def __init__( self, K, Distance):
#         self.K = K
#         self.Distance = Distance

#     def fit( self, X_train, Y_train):
#         self.X_train = X_train
#         self.Y_train = Y_train
#         # no_of_training_examples, no_of_features
#         self.m, self.n = X_train.shape

#     # Function for prediction
    
#     def predict( self, X_test):
#         self.X_test = X_test
#         # no_of_test_examples, no_of_features
#         self.m_test, self.n = X_test.shape
#         # initialize Y_predict
#         Y_predict = np.zeros( self.m_test )
        
#         for i in range( self.m_test):
#             x = self.X_test[i]
#             # find the K nearest neighbors from current test example
#             neighbors = np.zeros( self.K )
#             neighbors = self.find_neighbors( x )
            
#             # most frequent class in K neighbors
#             Y_predict[i] = round(neighbors.mean())
# #             Y_predict[i] = round(min(neighbors))

#         return Y_predict

#     # Function to find the K nearest neighbors to current test example

#     def find_neighbors( self, x):
#         # calculate all the euclidean distances between current
#         # test example x and training set X_train

#         euclidean_distances = np.zeros( self.m )
        
#         for i in range( self.m):
#             d = self.distance( x, self.X_train[i] )
#             euclidean_distances[i] = d

#         # sort Y_train according to euclidean_distance_array and
#         # store into Y_train_sorted
#         inds = euclidean_distances.argsort()
#         Y_train_sorted = self.Y_train[inds]

#         return Y_train_sorted[:self.K]

#     # Function to calculate euclidean distance
    
#     def distance( self, x, x_train):
        
#         if self.Distance == 'euclidean':
#             return np.sqrt( np.sum( np.square( x - x_train ) ) )

#         #create function to calculate Manhattan distance

#         elif self.Distance == 'Manhattan':
#             return sum(abs(val1-val2) for val1, val2 in zip(x, x_train))
        
#         #create function to calculate Manhattan distance
#         elif self.Distance == 'cosine':
#             result = dot(x, x_train)/(norm(x)*norm(x_train))
#             return result
        
# # Splitting dataset into train and test set

# X_train, X_test, Y_train, Y_test = train_test_split(
# X, Y, test_size = 1/3, random_state = 0 )

# Distance = input('Selct The Distance Fromula from list [euclidean,Manhattan,cosine] ==  ')
# K = int(input('Please Enter The K Value : '))

# # Model training
# model = KNeighborsRegressor(K, Distance)
# model.fit( X_train, Y_train )

# #Prediction on test set
# Y_pred = model.predict( X_test )

# MAE = mean_absolute_error(Y_test, Y_pred)
# print('mean_absolute_error on test set by our model:' ,MAE)

# MSE = mean_squared_error(Y_test, Y_pred, squared=False)
# print('mean_squared_error on test set by our model:' ,MSE)

# R_square = r2_score(Y_test, Y_pred)
# print('r2_score on test set by our model:' ,R_square)