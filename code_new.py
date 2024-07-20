import streamlit as st
import pandas as pd
import numpy as np
# import requests
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import sklearn
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# import tensorflow as tf
from tensorflow import keras 
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Data Extraction
bats = pd.read_csv('bats_symbols.csv')
bats = bats['Name']

st.set_page_config(page_title='Stock Market Price Prediction')#, layout = 'wide')

st.title('Stock Market Price Prediction')
if not st.sidebar.checkbox('Get Started'):
    df = None
        
else:
    with st.expander('Choose the Stock you want to analyze'):
        user_input = st.selectbox('', bats)
        ticker = yf.Ticker(user_input)
        df = ticker.history(period="15y")
        



if df is None:
    if st.button('About the app'):
        st.info('The app provides: \n-A glimpse of the stock performance \n-Predicts future closing price values of the stock')
        
elif df is not None:
    st.sidebar.header('Choose your task')
    task1 = st.sidebar.selectbox('', ['Analyze the Stock performance', 'Predicting future stock price values', 'Performance Metrics of the model'])
    task_model = st.sidebar.checkbox('Train the model')
    if not task_model:
        model = None
    else:
        # Scaling:
        df1=df.reset_index()['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))


        ## Splitting dataset
        # splitting the dataset into train and test
        training_size=int(len(df1)*0.75)
        test_size = len(df1)-training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
        train_data = scaler.fit_transform(np.array(df1.reshape(-1, 1)))

        def create_dataset(data, time_step=1):
            X, Y = [], []
            for i in range(len(data) - time_step - 1):
                a = data[i:(i + time_step), 0]
                X.append(a)
                Y.append(data[i + time_step, 0])
                # X is an array with every row containing the first time_step values for every time period (t)
                # Y is an array with every row containing the value of t + time_step
            return np.array(X), np.array(Y)

        # Time steps = 100
        X_train, y_train= create_dataset(train_data, time_step= 100)
        X_test, y_test = create_dataset(test_data, time_step= 100)

        # Reshaping the data to (train rows, train cols, 1) or (test rows, test cols, 1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Model:
        model=Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64, verbose=1)
        

        # Prediction
        y_pred_train=model.predict(X_train)
        y_pred_test=model.predict(X_test)

        # Transforming to the original form
        y_pred_train = scaler.inverse_transform(y_pred_train)
        y_pred_test = scaler.inverse_transform(y_pred_test)

        # Reshaping the y test and y train to inverse scale
        y_test=y_test.reshape(len(y_test), 1)
        y_train=y_train.reshape(len(y_train), 1)
        y_test_1 = scaler.inverse_transform(y_test)
        y_train_1 = scaler.inverse_transform(y_train)
        
        task2 = st.sidebar.selectbox('', ['Predicting future stock price values', 'Performance Metrics of the model'])
        

    
    if task1 == 'Analyze the Stock performance':
        with st.expander('Show Data'):
            st.dataframe(df)
        task2 = st.sidebar.radio('',['Visualize a Specific Column', 'Visualize Closing price'])
        # Visualization
        if task2 == 'Visualize Closing price':
            st.header('Visualizing Closing price')
            st.subheader('Closing Price vs Time chart')
            fig = plt.figure(figsize = (12, 5))
            plt.plot(df.Close)
            plt.xlabel('Time')
            plt.ylabel('Closing Price values')
            st.pyplot(fig)

            st.subheader('Closing Price vs Time chart with 100 MA')
            fig = plt.figure(figsize = (12, 5))
            # ma100 = df.close.rolling(100).mean()
            plt.plot(df.Close.rolling(60).mean(),'r', label = 'MA 60')
            plt.plot(df.Close.rolling(100).mean(),'g', label = 'MA 100')
            plt.plot(df.Close.rolling(150).mean(),'b', label = 'MA 150')
            plt.plot(df.Close, label = 'Original data')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Closing Price values')
            st.pyplot(fig)

            st.subheader('Frequency of Closing price')
            fig = plt.figure(figsize = (12, 5))
            sns.violinplot(x = df.Close, data = df)
            plt.ylabel('Frequency')
            st.pyplot(fig)
        
        elif task2 == 'Visualize a Specific Column':
            cols = []
            cols.extend(df.columns)
            # cols.remove('date')
            cols.remove('Close')
            task2_choice = st.selectbox('Select a column', cols)
            if task2_choice != 'None':
                st.header(f'Visualizing ***{task2_choice}*** stock price')
                st.subheader(f'***{task2_choice}*** vs Time chart')
                fig = plt.figure(figsize = (12, 5))
                plt.plot(df[task2_choice])
                plt.xlabel('Time')
                plt.ylabel(f'{task2_choice}')
                st.pyplot(fig)


                st.subheader(f'Frequency of ***{task2_choice}*** stock price')
                fig = plt.figure(figsize = (12, 5))
                sns.violinplot(x = df[task2_choice], data = df)
                plt.ylabel('Frequency')
                st.pyplot(fig)
            else:
                st.markdown('**No column selected**')
                
    elif task1 == 'Predicting future stock price values' and model is not None:

        
        # Future Prediction:
        st.header('Future Prediction')
        # Input x contains the last 100 days data

        # Reshaping the data into (row, col, 1) (here it is (1, 100, 1) a list converted into the form [[[   ]]] )
        input_x = test_data[int(len(test_data)-100):].reshape(1, -1)

        temp_input = list(input_x)
        temp_input = temp_input[0].tolist()

        # Predicting for the next 30 days

        days_to_predict = 30

        output_lst = []
        output = {}
        n_steps = 100
        i = 0
        while(i < days_to_predict):
            if(len(temp_input) > 100):
                # Removing the first element from the input for the next day prediction
                input_x = np.array(temp_input[1:])

                # Reshaping again
                input_x=input_x.reshape(1,-1)
                input_x = input_x.reshape((1, n_steps, 1))
                
                # Predicting output of i th day
                yhat = model.predict(input_x, verbose=0)

                # Adding the output of i th day in the output list array
                output_lst.extend(yhat.tolist())

                # Adding the predicted output of i th day and removing the first element from temporary input for prediction of price on i+1 th day
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                
                i=i+1
            else:
                # For the first prediction
                input_x = input_x.reshape((1, n_steps, 1))

                # Predicting output of i th day
                yhat = model.predict(input_x, verbose=0)
                
                # Adding the output of i th day in the output list array
                output_lst.extend(yhat.tolist())

                # Adding the predicted output of i th day to input for prediction of price on i+1 th day
                temp_input.extend(yhat[0].tolist())
                i=i+1
                
        # st.header()

        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 131)

        plt.figure(figsize= (25, 5))
        df3=df1.tolist()
        df3.extend(output_lst)

        # plot 1:
        fig = plt.figure(figsize = (10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(day_new, scaler.inverse_transform(df1[len(df1)-100:]), label = 'Input used for predicting')
        plt.plot(day_pred, scaler.inverse_transform(output_lst), label = 'Next 30day Predicted Price')
        plt.title('Input and the predicted price values')
        plt.legend()


        #plot 3:
        plt.subplot(2, 1, 2)
        plt.plot(df3)
        plt.title('Final graph that involves the predicted price values')
        st.pyplot(fig)

        # Printing the output values
        out = []
        for i in range(len(output_lst)):
            out.append(output_lst[i][0])
            
            
        output_lst_1 = np.array(output_lst)
        out = output_lst_1.reshape(len(output_lst_1), 1)
        out = scaler.inverse_transform(out)
        st.write("Closing price value {} day after today  is ${} \n".format(1,out[1][0]))
        for i in range(1, len(output_lst)):
            st.write("Closing price value {} days after today  is ${} \n".format(i+1,out[i][0]))
       
        
    elif task1 == 'Performance Metrics of the model' and model is not None:

        # Performance Metrics

        ## Root Mean Squared Error

        RMSE_train = mean_squared_error(y_train_1, y_pred_train, squared = False)  
        RMSE_test = mean_squared_error(y_test_1, y_pred_test, squared = False)

        st.subheader('Performance Metrics')
        st.write('RMSE of training dataset:', RMSE_train)
        st.write('\nRMSE of testing dataset:', RMSE_test)

        ## PLOT1: Both Training and testing perfromance
        st.subheader('Visualizing the model')

        look_back=100
        # train
        y_pred_train_plot = np.empty_like(df1)
        y_pred_train_plot[:, :] = np.nan
        y_pred_train_plot[look_back:len(y_pred_train)+look_back, :] = y_pred_train
        #Test
        y_pred_test_plot = np.empty_like(df1)
        y_pred_test_plot[:, :] = np.nan
        y_pred_test_plot[len(y_pred_train)+(look_back*2)+1:len(df1)-1, :] = y_pred_test

        # plt.plot(df1)
        fig = plt.figure(figsize = (12, 5))
        plt.plot(scaler.inverse_transform(df1), 'b', label= 'Original dataset')
        plt.plot(y_pred_train_plot, 'r', label= 'Predicted training dataset')
        plt.plot(y_pred_test_plot, 'g', label= 'Predicted testing dataset')
        plt.legend()
        st.pyplot(fig)

        # Visualizing the testing dataset
        st.subheader('Visualizing the testing dataset')
        plt.figure()
        fig1 = plt.figure(figsize = (12, 5))
        plt.plot(y_test_1,'g', label = 'Original Testing data')
        plt.plot(y_pred_test, 'r', label = 'Predicted testing data')
        plt.legend()
        st.pyplot(fig1)

        # Visualizing the training dataset
        st.subheader('Visualizing the training dataset')
        plt.figure()
        fig2 = plt.figure(figsize = (12, 5))
        plt.plot(y_train_1,'g', label = 'Original Testing data')
        plt.plot(y_pred_train, 'r', label = 'Predicted training data')
        plt.legend()
        st.pyplot(fig2)
    elif task1 == 'Predicting future stock price values' and model is None:
        st.write('You have to train the model for predicting future stock values')
    elif task1 == 'Performance Metrics of the model' and model is None:
        st.write('You have to train the model for predicting performance metrics of the model')
        
        
        
# TO DO
# NASDAQ, Dow Jones
# BRICKS