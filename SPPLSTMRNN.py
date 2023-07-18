import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, SimpleRNN , Dense
import streamlit as st

start = '2010-01-01'
end = '2023-07-02'

st.title('Stock Trend Prediction')

Ticker = st.text_input('Enter the ticker symbol: ')
start = st.text_input("Enter the start date (YYYY-MM-DD): ")
end = st.text_input("Enter the end date (YYYY-MM-DD): ")

df = yf.download(Ticker, start=start, end=end) 

if df.empty:
    st.write('No data available')
else:
    st.subheader('Data')
    st.write(df.describe())


    # Preprocess the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Split the data into training and testing sets
    train_data = scaled_data[:int(0.8 * len(scaled_data))]
    test_data = scaled_data[int(0.8 * len(scaled_data)):]

    # Prepare the training data
    X_train = []
    y_train = []
    for i in range(1, len(train_data)):
        X_train.append(train_data[i-1:i])
        y_train.append(train_data[i])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load the model
    lstm_model = load_model('keras_LSTM_model.h5')
    rnn_model = load_model('keras_RNN_model.h5')

    # Make predictions on the test data
    X_test = []
    y_test = []
    for i in range(1, len(test_data)):
        X_test.append(test_data[i-1:i])
        y_test.append(test_data[i])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    rnn_predictions = rnn_model.predict(X_test)
    rnn_predictions = scaler.inverse_transform(rnn_predictions)


    st.title('Accuracy and data loss of the model')
    
    
     # Calculate accuracy and loss
    actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_accuracy = 100 - np.mean(np.abs(actual_values - lstm_predictions) / actual_values) * 100
    rnn_accuracy = 100 - np.mean(np.abs(actual_values - rnn_predictions) / actual_values) * 100

    lstm_loss = np.mean(np.square(actual_values - lstm_predictions))
    rnn_loss = np.mean(np.square(actual_values - rnn_predictions))

    st.subheader('Accuracy and Loss')
    st.write('LSTM Accuracy: {:.2f}%'.format(lstm_accuracy))
    st.write('RNN Accuracy: {:.2f}%'.format(rnn_accuracy))
    st.write('LSTM Loss: {:.6f}'.format(lstm_loss))
    st.write('RNN Loss: {:.6f}'.format(rnn_loss))

    st.title('Predicted Graphs')

 # Plot the results
    fig1, ax = plt.subplots()
    ax.plot(lstm_predictions, color='red', label='LSTM Predicted value')
    ax.plot(rnn_predictions, color='green', label='RNN Predicted value')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Market Prediction (LSTM vs RNN)')
    ax.legend()
    st.pyplot(fig1)
   
    
    # Plot the results
    fig1, ax = plt.subplots()
    ax.plot(df['Close'].values[int(0.8 * len(scaled_data)) + 1:], color='blue', label='Actual Value')
    ax.plot(lstm_predictions, color='red', label='LSTM Predicted value')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Market Prediction (LSTM)')
    ax.legend()
    st.pyplot(fig1)

    # Plot the results
    fig2, ax = plt.subplots()
    ax.plot(df['Close'].values[int(0.8 * len(scaled_data)) + 1:], color='blue', label='Actual Value')
    ax.plot(rnn_predictions, color='green', label='RNN Predicted value')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock Price')
    ax.set_title('Stock Market Prediction (RNN)')
    ax.legend()
    st.pyplot(fig2)


# Compare and conclude
    if lstm_accuracy > rnn_accuracy:
        st.subheader('Result')
        st.write('LSTM model outperforms RNN model in terms of accuracy.')
        st.write('LSTM model is more accurate in predicting the stock trend.')
    elif lstm_accuracy < rnn_accuracy:
        st.subheader('Result')
        st.write('RNN model outperforms LSTM model in terms of accuracy.')
        st.write('RNN model is more accurate in predicting the stock trend.')
    else:
        st.subheader('Result')
        st.write('Both LSTM and RNN models have the same accuracy.')
        st.write('Both models perform similarly in predicting the stock trend.')