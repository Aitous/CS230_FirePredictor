#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def CreateDataset(file):
    df = pd.read_table(file)
    df.columns = ["x_coord", "y_coord", "month", "day",
                  "ffmc", "dmc", "dc", "isi", "temp",
                  "rh", "wind", "rain", "area"]
    np.random.seed(19)

    mms = MinMaxScaler()
    df["area"] = mms.fit_transform(df["area"].values.reshape(-1, 1))

    y = df.pop("area")
    X = df

    return X, y


def create_model():
    model = Sequential()

    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(10, activation='relu'))    
    model.add(Dense(1, activation='linear'))
    
    opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.15, random_state=19)

    model = create_model()
    
    history = model.fit(X_train, y_train, epochs=30, verbose=1, validation_data=(X_test, y_test))
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model's loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Training and test losses")
    plt.show()
    
    return model


def main():

    X, y = CreateDataset("data.txt")

    model = train_model(X, y)
    data1 = "6,5,9,6,92.5,121.1,674.4,8.6,25.1,27,4,0"
    data2 = "2,2,8,6,93.7,231.1,715.1,8.4,21.9,42,2.2,0"

    data1 = np.array(list(map(float, data1.split(",")))).reshape(-1,1).T
    data2 = np.array(list(map(float, data2.split(",")))).reshape(-1,1).T
    predicted1 = model.predict(data1)
    predicted2 = model.predict(data2)

    print(predicted1, predicted2)
    
    model.save_weights("model_weights.h5")
    print("\nModel weights saved to: 'model_weights.h5'")


if __name__ == "__main__":
    main()


# In[ ]:




