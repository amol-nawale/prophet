import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import date
import streamlit as st
from prophet import Prophet



st.title('Long Term Stock Price Forecasting')
ticker=st.text_input("Enter Stock Ticker Symbol")
days=st.number_input("Enter Approximate No of Days Forecast you want")

button=st.button('submit')

if button:
    tcs=yf.Ticker(ticker)
    tcs_history=tcs.history(start='2010-01-01',end=date.today())
    tcs_history.reset_index(inplace=True)
    data=tcs_history
    data.tail()

    # Select only the important features i.e. the date and price
    data = data[["Date","Close"]] # select Date and Price
    # Rename the features: These names are NEEDED for the model fitting
    data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
    data.head()

    m = Prophet(daily_seasonality = True) # the Prophet class (model)
    m.fit(data) # fit the model using all data

    future = m.make_future_dataframe(periods=730) #we need to specify the number of days in future
    prediction = m.predict(future)
    prediction

    m.plot(prediction)
    plt.title("Prediction of stock price using prophet")
    plt.xlabel("Date")
    plt.ylabel("Close Stock Price")
    plt.show()

    fig, ax = plt.subplots()
    ax=m.plot(prediction)
    st.pyplot(fig)
        
    st.pyplot(fig)

    f=prediction[-(days-1):]
    f.head(10)

    plt.plot(f['yhat'])

    plt.plot(prediction['yhat'])
    plt.plot(prediction['yhat'])

    import pandas_market_calendars as mcal

    # Create a calendar
    nse = mcal.get_calendar('NSE')
    date = nse.schedule(start_date='2023-04-12', end_date='2030-01-01')
    date.reset_index(inplace=True)
    date=date['index']
    date=pd.DataFrame(date)
    date

    f.reset_index(inplace=True)
    f.head()

    f.reset_index(inplace=True)
    f.head()

    date.dropna(inplace=True)
    date.rename(columns={'index':'date'},inplace=True)
    date

    date.set_index('date',inplace=True)
    date.head()


    plt.figure(figsize=(10,5))
    plt.plot(date,color='red')
