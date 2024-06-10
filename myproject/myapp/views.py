from django.shortcuts import render
from myapp.data_ingestion import fetch_stock_data
from myapp.data_processing import preprocess_data
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

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

def home(request):
    ticker = "BBCA.JK"
    stock_data = get_stock_data(ticker)

    # Plot the stock data
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()

    # Save the plot to a PNG in memory
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the PNG to base64 string
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    last_price = stock_data['Close'][-1]
    prev_price = stock_data['Close'][-2]
    price_change = last_price - prev_price
    percent_change = (price_change / prev_price) * 100

    context = {
        'ticker': ticker,
        'last_price': stock_data['Close'][-1],
        'price_change': stock_data['Close'][-1] - stock_data['Close'][-2],
        'percent_change': ((stock_data['Close'][-1] - stock_data['Close'][-2]) / stock_data['Close'][-2]) * 100,
        'dates': list(stock_data.index.strftime('%Y-%m-%d')),
        'open_prices': list(stock_data['Open']),
        'high_prices': list(stock_data['High']),
        'low_prices': list(stock_data['Low']),
        'close_prices': list(stock_data['Close']),
    }

    return render(request, 'home.html', context)