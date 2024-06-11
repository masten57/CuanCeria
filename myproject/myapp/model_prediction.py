import numpy as np
from keras.models import load_model

def predict_stock_price(model, data, scaler, look_back=1):
    X, _ = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    prediction = model.predict(X)
    return prediction

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)