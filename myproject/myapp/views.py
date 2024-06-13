from django.shortcuts import render
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
from .prednext import GrabDataForNextDayReg, GrabDataForNextDayClf
import requests
import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    data['Histogram_Positive'] = data['Histogram'].apply(lambda x: x if x >= 0 else 0)
    data['Histogram_Negative'] = data['Histogram'].apply(lambda x: x if x < 0 else 0)
    return data

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Std Dev'] = data['Close'].rolling(window=window).std()
    data['Upper Band'] = data['Middle Band'] + (data['Std Dev'] * num_std_dev)
    data['Lower Band'] = data['Middle Band'] - (data['Std Dev'] * num_std_dev)
    return data

def get_news(api_key, query="BBCA", page_size=8):
    url = f'https://newsapi.org/v2/everything?q={query}&pageSize={page_size}&apiKey={api_key}'
    response = requests.get(url)
    logging.debug(f"NewsAPI response status: {response.status_code}")
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        logging.debug(f"NewsAPI articles received: {len(articles)}")
        logging.debug(f"NewsAPI response content: {response.json()}")
        return articles
    else:
        logging.error(f"Failed to fetch news: {response.text}")
        return []

def home(request):
    ticker = "BBCA.JK"
    stock_data = get_stock_data(ticker)
    stock_data['RSI'] = calculate_rsi(stock_data)
    stock_data = calculate_macd(stock_data)
    stock_data = calculate_bollinger_bands(stock_data)

    # Filter data yang valid
    valid_data = stock_data.dropna(subset=['Middle Band', 'Upper Band', 'Lower Band'])

    last_price = valid_data['Close'][-1]
    prev_price = valid_data['Close'][-2]
    price_change = last_price - prev_price
    percent_change = (price_change / prev_price) * 100

    # Load models and make predictions
    my_scaler = joblib.load('myapp/models/feature_scalerv2.gz')
    my_price_scaler = joblib.load('myapp/models/price_scalerv2.gz')
    my_classifier = joblib.load('myapp/models/xgbc_v1.gz')
    new_model_lstm = tf.keras.models.load_model('myapp/models/lstm_regmodel_v2.keras', compile=False)
    new_model_gru = tf.keras.models.load_model('myapp/models/gru_regmodel_v2.keras', compile=False)

    lstm_prediction = GrabDataForNextDayReg(my_scaler, my_price_scaler, new_model_lstm)
    gru_prediction = GrabDataForNextDayReg(my_scaler, my_price_scaler, new_model_gru)
    xgb_prediction = GrabDataForNextDayClf(my_classifier)

    # Get news
    api_key = 'b612d39849564d4e8dd52b29f088985a'
    news_articles = get_news(api_key)
    logging.debug(f"News articles in context: {len(news_articles)}")

    context = {
        'ticker': ticker,
        'last_price': last_price,
        'price_change': price_change,
        'percent_change': percent_change,
        'dates': list(valid_data.index.strftime('%Y-%m-%d')),
        'open_prices': list(valid_data['Open']),
        'high_prices': list(valid_data['High']),
        'low_prices': list(valid_data['Low']),
        'close_prices': list(valid_data['Close']),
        'rsi_values': list(valid_data['RSI'].fillna(0)),  # fill NaN with 0 for simplicity
        'macd_values': list(valid_data['MACD'].fillna(0)),
        'signal_values': list(valid_data['Signal'].fillna(0)),
        'histogram_positive': list(valid_data['Histogram_Positive'].fillna(0)),
        'histogram_negative': list(valid_data['Histogram_Negative'].fillna(0)),
        'middle_band': list(valid_data['Middle Band'].fillna(0)),
        'upper_band': list(valid_data['Upper Band'].fillna(0)),
        'lower_band': list(valid_data['Lower Band'].fillna(0)),
        'lstm_prediction': lstm_prediction,
        'gru_prediction': gru_prediction,
        'xgb_prediction': xgb_prediction,
        'news_articles': news_articles
    }

    return render(request, 'home.html', context)
