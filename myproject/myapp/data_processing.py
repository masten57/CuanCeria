import pandas as pd
from datetime import datetime, timedelta

def preprocess_data(df):
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    # Tambahkan langkah preprocessing lain jika diperlukan
    return df

if __name__ == "__main__":
    from data_ingestion import fetch_stock_data
    ticker = 'BBCKA.JK'
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    raw_df = fetch_stock_data(ticker, start_date, end_date)
    processed_df = preprocess_data(raw_df)
    print(processed_df.head())