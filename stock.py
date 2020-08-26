import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

dataset = pd.read_csv('F:/spyder_projects/Stock Prediction/train_set.csv',index_col='Date',parse_dates=True)

#print(dataset.isna().any())
#print(dataset.info())   
#dataset['Open'].plot(figsize=(16,6))

dataset['Volume']=dataset['Volume'].str.replace(',','').astype(float)

dataset.rolling(7).mean()
training_set=dataset['Open']
training_set=pd.DataFrame(training_set)

#print(dataset.isna().any())

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set_scaled=sc.fit_transform(training_set)

x_train=[]
y_train=[]
for i in range (60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential() 

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2))    

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2)) 

regressor.add(Dense(units=1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('F:/spyder_projects/Stock Prediction/test_set.csv',index_col="Date",parse_dates=True)
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_test["Volume"] = dataset_test["Volume"].str.replace(',', '').astype(float)

test_set=dataset_test['Open']
test_set=pd.DataFrame(test_set)

dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []

for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
    
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)    

predicted_stock_price=pd.DataFrame(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
