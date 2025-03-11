![CuanceriaTeam](https://github.com/user-attachments/assets/0debb966-23c4-4aec-a606-861d2761efc1)

This project focuses on predicting the closing price of PT. Bank Central Asia Tbk. (BBCA) stock using historical stock data and technical indicators. The dataset, obtained from Yahoo Finance via the `yfinance` API, includes Open, High, Low, Close, and Volume data from April 15, 2020, to June 12, 2024. Additionally, technical indicators such as RSI, MACD, and Bollinger Bands are extracted for feature engineering. The project implements an MLOps architecture covering data ingestion, preprocessing, model training, deployment, and monitoring. LSTM, GRU, and XGBoost models are trained for stock price prediction and price movement classification. The LSTM model outperforms GRU in predicting stock prices, achieving a lower MAPE and MAE, while the XGBoost model for classification yields an accuracy of 33%. The trained models are stored for future use, and a web interface built with Django and Tailwind CSS provides real-time stock analysis, including technical indicators and price predictions.
