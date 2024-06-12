import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm


#setup for progress_apply()
tqdm.pandas()

# # Downloading the dat


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

today = datetime.date.today()

df = QuickGrabAndPrepro('BBCA.JK',datetime.datetime(2020,4,15), today)

def ClassificationLabeling(df):
    df['Target'] = (df['Close'] > df['Open']).astype(int)
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

def DataPrepPipeline(data):
    data = ClassificationLabeling(data)
    data = SwapRSI(data,'Close')
    return data

df = DataPrepPipeline(df)

def str_to_datetime(s):
  s = str(s)
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

df['Date'] = df['Date'].apply(str_to_datetime)

def df_to_windowed_df(dataframe, rows, namecol, n = 3):
    subset_df = pd.DataFrame({})
    subset_df['Target'] = dataframe[namecol]
    subset_df = subset_df.tail(rows + n)
    for i in range (1,n+1):
        subset_df.insert(loc = 0,
        column = f'{namecol}-{i}',
        value = subset_df['Target'].shift(i))
    subset_df = subset_df.dropna(how = 'any', axis = 0)
    return subset_df

def FeatureEngineeringPipeline(dataframe, rows, n = 3):

  #Grab the target column first (close price)
  df_close = df_to_windowed_df(dataframe,rows,'Close', n)
  cols1 = dataframe.columns.drop(['Date','Target'])

  #Creating windowed freatures
  windowed_df_list = []
  for i in cols1:
    windowed_df_list.append(df_to_windowed_df(dataframe,rows,i, n))

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

df1 = FeatureEngineeringPipeline(df,120,10)

df1['index'] = df['Date']
df1.rename(columns = {'index':'Date'}, inplace = True)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

qu_70 = int(len(df1) * .7)

subset_df1 = df1.drop('Date', axis = 1)
fitted_scaler=minmax_scaler.fit(subset_df1[:qu_70].to_numpy())
df2=fitted_scaler.transform(subset_df1.to_numpy())

df2 = pd.DataFrame(df2)
df2.insert(loc = 0,
            column = df1['Date'].name,
            value = df1['Date'])

def ConvertDF(df, n, feature_count):
  df_as_np = df.to_numpy()

  dates = df_as_np[:, 0]

  #Your features
  middle_matrix = df_as_np[:, 1:-1]

  #.reshape argument(row length, timestamp, feature count)
  X = middle_matrix.reshape((len(dates), n, feature_count))

  #Your target
  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = ConvertDF(df2,10,7)

q_70 = int(len(dates) * .7)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_70], X[:q_70], y[:q_70]

dates_val, X_val, y_val = dates[q_70:q_90], X[q_70:q_90], y[q_70:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

#pip install tensorflow_addons

#pip install keras.src.engine

from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience = 20,
                        restore_best_weights=True)

model_lstm = Sequential([layers.Input((10, 7)),
                    # layers.LSTM(128,return_sequences=True),
                    layers.LSTM(64,return_sequences=False),
                    layers.Dense(64,activation = 'relu'),
                    layers.Dense(32, activation = 'relu'),
                    layers.Dense(1)])

model_lstm.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'],
            run_eagerly = True)

history_lstm = model_lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)

model_gru = Sequential([layers.Input((10, 7)),
                    layers.GRU(128,return_sequences=True),
                    layers.GRU(64,return_sequences=False),
                    layers.Dense(64,activation = 'relu'),
                    layers.Dense(32, activation = 'relu'),
                    layers.Dense(1)])

model_gru.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'],
            run_eagerly = True)

history_gru = model_gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)

model_lstm.save('D:/Unair Semester 6/Machine Learning/CuanCeria/myproject/myapp/models/lstm_regmodel_v2.keras')
# model_gru.save('myapp/models/gru_regmodel_v1.keras')
