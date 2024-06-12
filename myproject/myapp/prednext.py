import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm
from tensorflow import keras

import yfinance as yf
def GetDataResIndex(symbol,startDate,endDate):
  data = yf.Ticker(str(symbol))
  data_df = data.history(start=startDate, end=endDate)
  data_df = data_df.reset_index()
  data_df['Date'] = [x.date() for x in data_df['Date']]
  return data_df


def QuickGrabAndPrepro(symbol,startDate,endDate):
  data = GetDataResIndex(symbol,startDate, endDate)
  data = data.drop(['Dividends', 'Stock Splits'], axis=1)

  #Add MACD
  placeholder = ta.macd(data.Close)
  data['MACD'] = placeholder['MACD_12_26_9']
  #Add RSI
  data['RSI'] = ta.rsi(data.Close)

  data = data.dropna(how = 'any')
  data = data.reset_index(drop = True)
  return data


def ClassificationLabeling(df):
  df['Class'] = (df['Close'] > df['Open']).astype(int).astype(float)
  return df

def SwapRSI(data,swap_col):
    RSI = data['RSI'].copy()
    col = data[swap_col].copy()
    df_data = data.copy()
    col_list = list(df_data)    
    col_list[4], col_list[7] = col_list[7], col_list[4]    
    df_data.columns = col_list
    df_data[swap_col], df_data['RSI'] = col, RSI
    return df_data

def DataPrepPipeline(data,col):
  data = ClassificationLabeling(data)
  data = SwapRSI(data,col)
  return data

def str_to_datetime(s):
  s = str(s)
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

def df_to_windowed_df(dataframe, rows,namecol, n = 3):
    subset_df = pd.DataFrame({})
    subset_df['Target'] = dataframe[namecol]
    subset_df = subset_df.tail(rows + 2*n)
    for i in range (1,n+1):
        subset_df.insert(loc = 0,
          column = f'{namecol}-{i}',
          value = subset_df['Target'].shift(i))
    subset_df = subset_df.dropna(how = 'any', axis = 0)
    return subset_df

def df_to_windowed_df_last(dataframe, rows,namecol, n = 3):
    subset_df = pd.DataFrame({})
    subset_df['Target'] = dataframe[namecol]
    subset_df = subset_df.tail(rows + 2*n)
    for i in range (1,n+1):
        subset_df.insert(loc = 0,
        column = f'{namecol}-{i}',
        value = subset_df['Target'].shift(i))
    for i in range (0,n):
        subset_df.iloc[:,i], subset_df.iloc[:,i+1] = subset_df.iloc[:,i+1], subset_df.iloc[:,i] 
    subset_df = subset_df.dropna(how = 'any', axis = 0)
    return subset_df

def FeatureEngineeringPipeline_GrabLast(data, rows, n = 3):

  dataframe = data.copy()
  #Setting the index
  dataframe['Date'] = dataframe['Date'].apply(str_to_datetime)
  dataframe = dataframe.set_index('Date', inplace = False)

  #Grab the target column first (close price)
  df_close = df_to_windowed_df_last(dataframe,rows,'Close', n)
  cols1 = dataframe.columns.drop(['Class'])

  #Creating windowed freatures
  windowed_df_list = []
  for i in cols1:
    windowed_df_list.append(df_to_windowed_df_last(dataframe,rows,i, n))

  #We want to append the windowed features to our target column
  df2 = pd.DataFrame(df_close['Target'])

  #Grabbing the second-to-last column because the last column is the target
  idx_controller = -2
  iteration_stopper = False
  while True:
    if iteration_stopper:
      break

      #Adding features
    for j in range(0,len(cols1)):
      df2.insert(loc = 0,
            column = windowed_df_list[j].iloc[:,idx_controller].name,
            value = windowed_df_list[j].iloc[:,idx_controller])
    idx_controller -= 1

    if idx_controller == -(n+2):
      iteration_stopper = True
  return df2.reset_index()

def FeatureEngineeringPipeline(data, rows, n = 3):

  dataframe = data.copy()
  #Setting the index
  dataframe['Date'] = dataframe['Date'].apply(str_to_datetime)
  dataframe = dataframe.set_index('Date', inplace = False)

  #Grab the target column first (close price)
  df_close = df_to_windowed_df(dataframe,rows-n,'Close', n)
  cols1 = dataframe.columns.drop(['Class'])

  #Creating windowed freatures
  windowed_df_list = []
  for i in cols1:
    windowed_df_list.append(df_to_windowed_df(dataframe,rows-n,i, n))

  #We want to append the windowed features to our target column
  df2 = pd.DataFrame(df_close['Target'])

  #Grabbing the second-to-last column because the last column is the target
  idx_controller = -2
  iteration_stopper = False
  while True:
    if iteration_stopper:
      break

      #Adding features
    for j in range(0,len(cols1)):
      df2.insert(loc = 0,
            column = windowed_df_list[j].iloc[:,idx_controller].name,
            value = windowed_df_list[j].iloc[:,idx_controller])
    idx_controller -= 1

    if idx_controller == -(n+2):
      iteration_stopper = True
  return df2.reset_index()

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def ScaleData(df,qu,how = 'minmax'):
    if how == 'minmax':
        
        minmax_scaler_df = MinMaxScaler()
        
        minmax_scaler_price = MinMaxScaler()
        
        q = int(len(df) * qu)

        subset_df1 = df.drop('Date', axis = 1)
        fitted_scaler = minmax_scaler_df.fit(subset_df1[:q].to_numpy())
        
        price_scaler = minmax_scaler_price.fit(subset_df1['Target'][:q].to_numpy().reshape(-1,1))
        
        df2 = fitted_scaler.transform(subset_df1.to_numpy())
        
        df2 = pd.DataFrame(df2)
        df2.insert(loc = 0,
            column = df['Date'].name,
            value = df['Date'])
        
        return df2, fitted_scaler, price_scaler
    if how == 'standard':
        standard_scaler_df = StandardScaler()
        
        standard_scaler_price = StandardScaler()
        
        q = int(len(df) * qu)

        subset_df1 = df.drop('Date', axis = 1)
        fitted_scaler = standard_scaler_df.fit(subset_df1[:q].to_numpy())
        
        price_scaler = standard_scaler_price.fit(subset_df1['Target'][:q].to_numpy().reshape(-1,1))
        
        df2 = fitted_scaler.transform(subset_df1.to_numpy())
        
        df2 = pd.DataFrame(df2)
        df2.insert(loc = 0,
            column = df['Date'].name,
            value = df['Date'])
        
        return df2, fitted_scaler, price_scaler
    
def ConvertDF(df, n, feature_count):
  dataframe = df.copy()
  df_as_np = dataframe.to_numpy()
  dates = df_as_np[:, 0]

  #Your features
  middle_matrix = df_as_np[:, 1:-1]

  #.reshape argument(row length, timestamp, feature count)
  X = middle_matrix.reshape((len(dates), n, feature_count))

  #Your target
  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

def GrabDataForNextDayReg(scaler,scaler_price, reg_model):
    #Preprocessing, no need to specify dates because the function always grabs the stock's price today until 600 days before.
    today = datetime.date.today()
    minus_600 = today - datetime.timedelta(days=600)
    df = QuickGrabAndPrepro('BBCA.JK',minus_600, today) 
    df = DataPrepPipeline(df,'Close')
    
    #Feature Engineering, the rows grabbed here does not matter 
    #We are taking the very last row, and we want today's close price as tomorrow's close-1
    df_windowed = FeatureEngineeringPipeline_GrabLast(df,120,10)
    
    #Scaling using the fitted model before
    df_windowed = df_windowed.drop('Date', axis = 1)
    
    df_scaled = scaler.transform(df_windowed.to_numpy())
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.insert(loc = 0,
            column = df['Date'].name,
            value = df['Date'])
    
    #to np conversion
    dates, X, y = ConvertDF(df_scaled,10,7)
    
    #We use the trained model
    reg_model_used = reg_model
    
    #We want to grab the last day to be predicted
    pred = reg_model_used.predict(X[-1:,:,:]).flatten()
    pred_inv_scaled = scaler_price.inverse_transform(pred.reshape(1,-1))
    
    return pred_inv_scaled[0][0]

import joblib



import xgboost as xgb

def FeatureEngineeringPipelineClassification(data, rows, n = 3):

  dataframe = data.copy()
  #Setting the index
  dataframe['Date'] = dataframe['Date'].apply(str_to_datetime)
  dataframe = dataframe.set_index('Date', inplace = False)

  #Grab the target column first (Class)
  df_close = df_to_windowed_df(dataframe,rows-n,'Class', n)
  cols1 = dataframe.columns.drop(['Close'])

  #Creating windowed freatures
  windowed_df_list = []
  for i in cols1:
    windowed_df_list.append(df_to_windowed_df(dataframe,rows-n,i, n))

  #We want to append the windowed features to our target column
  df2 = pd.DataFrame(df_close['Target'])

  #Grabbing the second-to-last column because the last column is the target
  idx_controller = -2
  iteration_stopper = False
  while True:
    if iteration_stopper:
      break

      #Adding features
    for j in range(0,len(cols1)):
      df2.insert(loc = 0,
            column = windowed_df_list[j].iloc[:,idx_controller].name,
            value = windowed_df_list[j].iloc[:,idx_controller])
    idx_controller -= 1

    if idx_controller == -(n+2):
      iteration_stopper = True
  return df2.reset_index()

def FeatureEngineeringPipelineClassification_GrabLast(data, rows, n = 3):

  dataframe = data.copy()
  #Setting the index
  dataframe['Date'] = dataframe['Date'].apply(str_to_datetime)
  dataframe = dataframe.set_index('Date', inplace = False)

  #Grab the target column first (Class)
  df_close = df_to_windowed_df_last(dataframe,rows,'Class', n)
  cols1 = dataframe.columns.drop(['Close'])

  #Creating windowed freatures
  windowed_df_list = []
  for i in cols1:
    windowed_df_list.append(df_to_windowed_df_last(dataframe,rows,i, n))

  #We want to append the windowed features to our target column
  df2 = pd.DataFrame(df_close['Target'])

  #Grabbing the second-to-last column because the last column is the target
  idx_controller = -2
  iteration_stopper = False
  while True:
    if iteration_stopper:
      break

      #Adding features
    for j in range(0,len(cols1)):
      df2.insert(loc = 0,
            column = windowed_df_list[j].iloc[:,idx_controller].name,
            value = windowed_df_list[j].iloc[:,idx_controller])
    idx_controller -= 1

    if idx_controller == -(n+2):
      iteration_stopper = True
  return df2.reset_index()
def GrabDataForNextDayClf(clf_model):
    #Preprocessing, no need to specify dates because the function always grabs the stock's price today until 600 days before.
    today = datetime.date.today()
    minus_600 = today - datetime.timedelta(days=600)
    df = QuickGrabAndPrepro('BBCA.JK',minus_600, today) 
    df = DataPrepPipeline(df, 'Close')
    
    dfc_w_last = FeatureEngineeringPipelineClassification_GrabLast(df,120,10)
    
    X = dfc_w_last.drop(['Date','Target'], axis=1)
    y = dfc_w_last['Target']
    
    X_last, y_last =  X[-1:], y[-1:]
    dlast_clf = xgb.DMatrix(X_last, y_last, enable_categorical = False)

    #We are taking the very last row, and we want today's class value as tomorrow's class-1
    pred_model = clf_model
    y_pred_prob = pred_model.predict(dlast_clf)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    if y_pred[0] == 0:
        return 'Down'
    else:
        return 'Up'
    
import joblib

# Save the model to a file
#joblib.dump(classifier, "(your directory, use backslash / instead of \)/xgbc_v1.gz")

#loading both the feature scaler and the price scaler

# my_scaler = joblib.load('D:/Unair Semester 6/Machine Learning/CuanCeria/feature_scaler.gz')
# my_price_scaler = joblib.load('D:/Unair Semester 6/Machine Learning/CuanCeria/price_scaler.gz')

#loading the lstm model
# new_model_lstm = tf.keras.models.load_model('D:/Unair Semester 6/Machine Learning/CuanCeria/lstm_regmodel_v2.keras', compile=False)
# new_model_lstm.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_error'])

#loading the gru model
# new_model_gru = tf.keras.models.load_model('D:/Unair Semester 6/Machine Learning/CuanCeria/gru_regmodel_v1.keras', compile=False)

# print(GrabDataForNextDayReg(my_scaler,my_price_scaler, new_model_gru))


# print(GrabDataForNextDayReg(my_scaler, my_price_scaler, new_model_lstm))

# my_classifier = joblib.load('D:/Unair Semester 6/Machine Learning/CuanCeria/xgbc_v1.gz')

# GrabDataForNextDayClf(my_classifier)