import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

from plotly import graph_objs as go 

start_date = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock predictor")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Please select dataset for prediction", stocks)

num_years = st.slider("Prediction Years", 1, 4) 
period = num_years * 365

### Loading stock data 
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, today)
    data.reset_index(inplace=True) ##sets the date to the very first coloumn
    return data

dataload_state = st.text("Loading data ...")

data = load_data(selected_stock)
dataload_state.text("....Done")

st.subheader("Raw data")
st.write(data.tail())

##plotting the data

def plot_raw_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_open'))
    figure.layout.update(title_text ="Time series data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_raw_data()

#Forcasting

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"}) #how fb prophet wants data from documentation

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

