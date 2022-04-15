import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from pickle import dump,load
import preprocess
import json as JSON
from flask import Flask
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/Train',  methods=['GET'])
def Train():

    # read data
    df = pd.read_csv('data/result.csv')
    p = preprocess.Preprocessor()
    df,min_max_scaler = p.run(df)
    df = df.fillna(0)

    # save scaler to use in prediction
    dump(min_max_scaler, open('model/MinMaxScaler.pkl', 'wb'))

    # split train and test
    X = df.drop(columns=['final_remain_cycle','remain_cycle','max_cycle','status','cycle'])
    X = X.values
    Y = df['final_remain_cycle'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=33)
    print('data splitting compleate')

    # Define model parameter
    K = 7
    distance = 'euclidean'

    # Define KNN Regressor Model
    model = KNeighborsRegressor(n_neighbors=K,p=3,n_jobs=-1,metric=distance,weights='distance')
    model.fit( X_train, Y_train )
    print('model training compleate')

    # Save Fitted Model
    filename = 'model\Predictive_custom_KNN_minkowski_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # Make Prediction on test data
    Y_pred = model.predict( X_test )
    print('prediction done on test data')

    # Calculate Model Performance Parameter
    MAE = mean_absolute_error(Y_test, Y_pred)
    MSE = mean_squared_error(Y_test, Y_pred, squared=False)
    R_square = r2_score(Y_test, Y_pred)

    print('mean_absolute_error on test set by our model:' ,MAE)
    print('mean_squared_error on test set by our model:' ,MSE)
    print('r2_score on test set by our model:' ,R_square)

    # response = str(R_square).to_json()
    # MAE = JSON.loads(str(MAE))
    # MSE = JSON.loads(str(MSE))
    # R_square = JSON.loads(str(R_square))

    return {
        'status': 'model training complete !!',
        'MAE': MAE,
        'MSE': MSE,
        'R_square': R_square,
        'metric':distance,
        'K': str(K)
    }

@app.route('/Predict',  methods=['GET'])
def Predict():
    
    # load csv file
    df = pd.read_csv('data/sample_test.csv')
    df = df.head(1)

    # Sort Data by Downtime
    df = df.sort_values(by='DOWNTIME')
    
    # replace nan value with null
    df = df.where(pd.notnull(df), None)

    # convert boolean object in to integer
    df['GREDE10890_FIX_TAKE_DATA_A'] =  df['GREDE10890_FIX_TAKE_DATA_A'].astype(int)
    df['GREDE10890_FIX_TAKE_DATA_B'] =  df['GREDE10890_FIX_TAKE_DATA_B'].astype(int)
    df['GREDE10890_FIX_TRG_M_V'] =  df['GREDE10890_FIX_TRG_M_V'].astype(int)

    # remove columns whoes type is datetime
    df = df.drop(columns=['UTC_TIMESTAMP','LOCAL_TIMESTAMP'])

    # select only integer columns from data
    df = df.select_dtypes(include='number')

    # Remove unwanted columns
    if any('final_remain_cycle' in string for string in df.columns):
        df = df.drop(columns=['final_remain_cycle'])
    elif any('Unnamed: 0' in string for string in df.columns):
        df = df.drop(columns=['Unnamed: 0'])
    else:
        pass
    
    # load saved scaler
    scaler = load(open('model/MinMaxScaler.pkl', 'rb'))

    # Transform data using scaler
    prediction_data = scaler.transform(df)

    # Load Saved Model
    model = pickle.load(open('model/Predictive_custom_KNN_minkowski_model.sav', 'rb'))

    # Make a prediction
    predictions = model.predict(prediction_data)
    print('Remaining Usefull Life Is:',int(predictions))
    
    # Convert Response in to Json
    response = JSON.loads(str(int(predictions)))

    return {
        'Remaining Useful Life Is: ': response,
    }


if __name__ == "__main__":
    app.run(debug=True)



# import json
# import pyqrcode
# import png
# from pyqrcode import QRCode
# import base64

# def lambda_handler(event, context):
#     payload = {
#             "queryStringParameters":{
#                                       "text": event["queryStringParameters"]["text"]
#                                     }
#              }
#     url = pyqrcode.create(event["queryStringParameters"]["text"])
    
#     url.png('/tmp/myqr.png', scale = 6)
    
#     with open("/tmp/myqr.png", "rb") as f:
#         b = base64.b64encode(f.read()).decode("utf-8")

#     return {
#         "statusCode": 200,
#         "headers": {
#             'Content-Type': 'image/png'
#         },
#         "body": b,
#         "isBase64Encoded": True
#     }