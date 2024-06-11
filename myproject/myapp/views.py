from django.shortcuts import render
from myapp.data_ingestion import fetch_stock_data
from myapp.data_processing import preprocess_data
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np

def index(request):
    ticker = 'BBCA.JK'
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    raw_df = fetch_stock_data(ticker, start_date, end_date)
    processed_df = preprocess_data(raw_df)
    tables = processed_df.to_html(classes='table-auto border-collapse border border-gray-200', justify='center')
    return render(request, 'index.html', {'tables': tables})

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
        'lower_band': list(valid_data['Lower Band'].fillna(0))
    }

    return render(request, 'home.html', context)
