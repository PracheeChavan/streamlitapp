import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from datetime import date
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

# Load data

start='2012-08-01'
end='2021-12-31'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock name ','TSLA')
data = web.DataReader(user_input, 'yahoo', start, end)



data.reset_index(inplace=True)

# Data visualization
from plotly import graph_objs as go
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()




st.subheader('Closing price vs Time chart with 100day MA and 200day MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'g',label='100day MA')
plt.plot(ma200,'r',label='200day MA')
plt.plot(data.Close,'b',label='closing price of the company')
plt.xlabel('time',fontsize=18)
plt.ylabel('share price of the company',fontsize=18)
plt.legend()
st.pyplot(fig)

# prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))


# Loading my model
model = load_model('keras_model.h5')

# load test data

test_start = '2022-01-01'
test_end = '2022-12-01'
test_data = web.DataReader(user_input, data_source='yahoo',start=test_start, end=test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

prediction_days = 60
model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
# st.table(model_inputs)
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

# make prediction on test data

x_test = []

for i in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[i - prediction_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# plot the test predictions

st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='b',label='Actual price of the company')
plt.plot(predicted_prices, color='black',label='Predicted price of the company')
plt.title('company share price chart')
plt.xlabel('time')
plt.ylabel('share price of the company')
plt.legend()
st.pyplot(fig2)

# predict next day price
empty_list = []
for date_delta in range(5):
    model_inputs = total_dataset[len(total_dataset)-prediction_days:].values
    model_inputs = np.append(model_inputs,empty_list)
    # st.table(model_inputs)
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1),0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = round(scaler.inverse_transform(prediction)[0][0],2)
    next_date = (dt.datetime.now() + dt.timedelta(days=date_delta)).date()
    st.subheader(f"{next_date} Prediction is: {prediction}")
    empty_list.append(prediction)
#print(f"Prediction: {prediction}")
    
