from sklearn.ensemble import RandomForestClassifier
from utils import load_data, get_data_day, get_common_sites
import numpy as np
from datetime import datetime
import sys
import pickle
from prediction import convert_data_day

from progressbar import ProgressBar

import logging
logging.basicConfig(level=logging.INFO)

def generate(year):

    all_data = load_data(dirname="../Data/training", year=year)

    day = 3

    print(all_data.keys())
    print(all_data['chimeres'].keys())
    print(all_data['chimeres']['NO2'].keys())
    # print(all_data['chimeres']['NO2']['date'])
    training_data = None
    # training_labels = None
    pbar = ProgressBar()
    sites = all_data['sites']
    idPolairs = get_common_sites(all_data)

    for day in pbar(range(day, 365)):
        data_day = get_data_day(day, all_data, max_days=3, year=year)
        chimeres_day = data_day[0]
        geops_day = data_day[1]
        meteo_day = data_day[2]
        concentrations_day = data_day[3]

        print("day: ", day)
        # print(data_day['NO2'])
        for pol in ["PM10","PM25","O3","NO2"]:
            # print('pollutant : ', pol)
            for idPolair in idPolairs:

                data = convert_data_day(day, pol, idPolair,
                                                  sites,
                                                  chimeres_day,
                                                  geops_day,
                                                  meteo_day,
                                                  concentrations_day)
                data = np.expand_dims(data, axis=0)
                # print(data.shape)
                if training_data is not None:
                    training_data = np.append(training_data, data, axis=0)
                    # training_labels = np.append(training_labels, labels, axis=0)
                else:
                    training_data = data
                    # training_labels = labels
        print(training_data.shape)
    np.savez('./tmp/training_data_'+str(year), a=training_data)




def origin_time(date_time):
    date = str(date_time).split(' ')[0]
    return np.datetime64(date + ' 00:00:00')


def convert_time(date_time, date_time_0):
    ts = (date_time - np.datetime64(date_time_0)) / np.timedelta64(1, 's')
    return int(ts / 3600)

def convert_pol(pol):
    if pol == "PM10":
        return 4
    elif pol == "PM25":
        return 3
    elif pol == "NO2":
        return 2
    elif pol == "O3":
        return 1



# generate(2012)

# generate(2014)
# generate(2015)
# generate(2016)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor


def transform(filename):

    data = np.load(filename)['a']
    x = None
    y = None
    print(data.shape)

    length = data.shape[0]
    for i in range(length - 3):
        temp_x = np.empty(shape=(1, 7, 26))
        for j in range(7):
            temp_x[0, j, 0:2] = data[i, 0:2]
            temp_x[0, j, 2:] = data[i, (2+24*j):(2+24*(j+1))]

        temp_y = np.empty(shape=(1, 72))
        temp_y[0, :] = data[i+3, 98:170]
        # print(temp_y.shape)
        if y is not None:
            y = np.append(y, temp_y, axis=0)
        else:
            y = temp_y


        if x is not None:
            x = np.append(x, temp_x, axis=0)
        else:
            x = temp_x

    print(x.shape)
    print(y.shape)
    return x, y
#X, Y = transform('./tmp/training_data_2012.npz')
#np.savez('./tmp/data_2012', a=X, b=Y)

def prepare_data(year):
    filename = './tmp/training_data_' + str(year) + '.npz'
    data = np.load(filename)['a']
    mean = np.mean(data, axis=0)
    np.savez('./tmp/mean_' + str(year), a=mean)
    labels = None
    print("data shape:", data.shape)
    pbar = ProgressBar()
    # labels = data[3:, 3:75]
    for i in pbar(range(data.shape[0]-3)):
        temp_y = np.empty(shape=[1, 72])
        temp_y[0, :] = data[i + 3, 0:72, 17]
        if labels is not None:
            labels = np.append(labels, temp_y, axis=0)
        else:
            labels = temp_y
    # features = data[:-3, :]
    features = data[:-3, :, :]
    print("features :", features.shape)
    print("labels :", labels.shape)
    np.savez('./tmp/full_data_' + str(year), a=features, b=labels)

# prepare_data(2012)
# prepare_data(2014)
# prepare_data(2015)
# prepare_data(2016)

def mini_generate(year):
    data = np.load('./tmp/full_data_'+str(year)+'.npz')
    X = data['a']
    Y = data['b']
    pbar = ProgressBar()
    X_ = None
    # labels = data[3:, 3:75]
    for i in pbar(range(X.shape[0])):
        temp_y = np.empty(shape=[1, 4, 24])
        tmp = X[i, :, 17]
        last_values = np.tile(tmp[72:], 3)[0:17]
        tmp = np.concatenate((tmp, last_values))
        temp_y[0, :, :] = tmp.reshape([1, 4, 24])
        if X_ is not None:
            X_ = np.append(X_, temp_y, axis=0)
        else:
            X_ = temp_y
    # Y = Y[:-3]
    print(X_.shape)
    print(Y.shape)
    np.savez('./tmp/test_data_' + str(year), a=X_, b=Y)

# mini_generate(2012)
# mini_generate(2014)
# mini_generate(2015)
# mini_generate(2016)

def mini_generate_2():
    data = np.load('./tmp/full_data_2012.npz')
    X = data['a']
    Y = data['b']
    pbar = ProgressBar()
    mean_x_y = np.load('./tmp/mean_coord.npz')['a']
    X_ = None
    for i in pbar(range(X.shape[0])):
        temp_y = np.empty(shape=[1, 82])
        temp_y[0, 0] = X[i, 0, 0]
        temp_y[0, 1:3] = X[i, 0, 1:3] - mean_x_y
        temp_y[0, 3:] = X[i, :, 17]
        if X_ is not None:
            X_ = np.append(X_, temp_y, axis=0)
        else:
            X_ = temp_y

    print(X_.shape)
    print(Y.shape)
    np.savez('./tmp/test_data_2012_2', a=X_, b=Y)

def final_data():
    data1 = np.load('./tmp/test_data_2012.npz')
    X1 = data1['a']
    X1 = X1.reshape(X1.shape[0], -1)[:, 0:79]
    Y1 = data1['b']
    data2 = np.load('./tmp/test_data_2014.npz')
    X2 = data2['a']
    X2 = X2.reshape(X2.shape[0], -1)[:, 0:79]
    Y2 = data2['b']
    data3 = np.load('./tmp/test_data_2015.npz')
    X3 = data3['a']
    X3 = X3.reshape(X3.shape[0], -1)[:, 0:79]
    Y3 = data3['b']
    data4 = np.load('./tmp/test_data_2016.npz')
    X4 = data4['a']
    X4 = X4.reshape(X4.shape[0], -1)[:, 0:79]
    Y4 = data4['b']
    X = np.concatenate((X1, X2, X3, X4))
    Y = np.concatenate((Y1, Y2, Y3, Y4))

    print(X.shape)
    print(Y.shape)

    np.savez('./tmp/final_data', a=X, b=Y)


# final_data()

def final_data2():
    data1 = np.load('./tmp/test_data_2012_2.npz')
    X1 = data1['a']
    Y1 = data1['b']
    data2 = np.load('./tmp/test_data_2014_2.npz')
    X2 = data2['a']
    Y2 = data2['b']
    data3 = np.load('./tmp/test_data_2015_2.npz')
    X3 = data3['a']
    Y3 = data3['b']

    X = np.concatenate((X1, X2, X3))
    Y = np.concatenate((Y1, Y2, Y3))

    print(X.shape)
    print(Y.shape)

    np.savez('./tmp/final_data_2', a=X, b=Y)


# final_data2()


# mini_generate_2()

import json

def train():


    # data_dim = 18
    # timesteps = 3*24+7
    data_dim = 24
    timesteps = 4

    # data = np.load('./tmp/all_data.npz')
    # X, Y = data['a'], data['b']



    # data = np.load('./tmp/full_data_2012.npz')
    # X = data['a']
    # Y = data['b']
    #
    # mean = np.load('./tmp/mean_2012.npz')['a']
    # X = X - np.repeat(np.array([mean]), X.shape[0], axis=0)

    data = np.load('./tmp/final_data.npz')
    X, Y = data['a'], data['b']
    Y = Y[:-3]

    # X = np.concatenate((x1, x2), axis=0)
    # Y = np.concatenate((y1, y2), axis=0)

    # X = normalize(X.reshape(X.shape[0], -1), axis=0, norm='l1').reshape(X.shape)
    # Y = normalize(Y, axis=0, norm='l1')
    print(X.shape)
    print(Y.shape)
    # X = X[0:200, :, :]
    # Y = Y[0:200, :]
    # #Place column means in the indices. Align the arrays using take

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(data_dim, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 24

    model.add(LSTM(48, return_sequences=True))  # returns a sequence of vectors of dimension 24
    # model.add(Dropout(0.4))
    # model.add(LSTM(48, return_sequences=True))  # returns a sequence of vectors of dimension 24
    # model.add(Dropout(0.4))
    model.add(LSTM(72, return_sequences=True))  # returns a sequence of vectors of dimension 24
    model.add(Dropout(0.2))

    model.add(LSTM(72))  # return a single vector of dimension 24
    model.add(Dense(72, activation='softmax'))

    sgd = optimizers.SGD(lr=0.9, clipnorm=1.)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd)

    model.fit(X, Y,
              batch_size=16, epochs=100,
              validation_split=0.3)

    # serialize model to JSON
    model_json = model.to_json()
    with open("./models/lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./models/lstm_model.h5")
    print("Saved model to disk")

from sklearn import metrics
from sklearn.model_selection import train_test_split

def train_randomforest():


    data = np.load('./tmp/final_data.npz')
    X, Y = data['a'], data['b']

    # X = normalize(X.reshape(X.shape[0], -1), axis=0, norm='l1').reshape(X.shape)
    # X = X[0:600, :]
    # Y = Y[0:600, :]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                Y,
                                test_size=0.2,
                                random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    model = RandomForestRegressor(max_depth=10,
                                  random_state=42,
                                  n_estimators=1300)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    filename = './models/randomforest_ffull.sav'

    pickle.dump(model, open(filename, 'wb'))

    print("Saved model to disk")

train_randomforest()
