from django.shortcuts import render
from myapp.data_ingestion import fetch_stock_data
from myapp.data_processing import preprocess_data
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

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

def home(request):
    ticker = "BBCA.JK"
    stock_data = get_stock_data(ticker)
    stock_data['RSI'] = calculate_rsi(stock_data)

    last_price = stock_data['Close'][-1]
    prev_price = stock_data['Close'][-2]
    price_change = last_price - prev_price
    percent_change = (price_change / prev_price) * 100

    context = {
        'ticker': ticker,
        'last_price': last_price,
        'price_change': price_change,
        'percent_change': percent_change,
        'dates': list(stock_data.index.strftime('%Y-%m-%d')),
        'open_prices': list(stock_data['Open']),
        'high_prices': list(stock_data['High']),
        'low_prices': list(stock_data['Low']),
        'close_prices': list(stock_data['Close']),
        'rsi_values': list(stock_data['RSI'].fillna(0))  # fill NaN with 0 for simplicity
    }

    return render(request, 'home.html', context)