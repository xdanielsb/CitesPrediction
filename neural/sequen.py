import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt

nameFile = "../data/cleaned.xlsx"

# read data
def readData():
    data = pd.read_excel(nameFile, index_col=0)
    info = data[data.columns[0:4]].values
    labels = np.array(data[data.columns[4]])
    labels = labels.reshape(labels.shape[0],1)

    columns_info = data.columns.values[0:4]
    columns_labels = data.columns.values[4]
    return info, labels, columns_info, [columns_labels]

def scalarMultipliers( s ):
    # print scaling adjustments
    print('Scalar multipliers')
    print(s.scale_)
    print('Scalar minimum')
    print(s.min_)

if __name__ == "__main__":
    config = 5
    res = open("results_conf_{}".format(config),"w+")
    res.write("run,mes,nameImg\n")

    for test in range( 10 ):
        data = open("data_figure_{}_{}".format(config, test+1), "w+")
        data.write("predicted,real\n")
        X, Y, LX, LY = readData()
        X, Y = shuffle( X, Y , random_state=random.randint(1, 1000))
        #Scaler
        scalerX = MinMaxScaler(feature_range=(0,1))
        scalerY = MinMaxScaler(feature_range=(0,1))

        #scale X
        scalerX.fit(X)
        ## scalarMultipliers( scalerX )
        SX = scalerX.transform(X)

        # Scale Y
        scalerY.fit(Y)
        ## scalarMultipliers(scalerY)
        SY = scalerY.transform(Y)

        # Conver scaled to data frames
        X = pd.DataFrame(SX, columns=LX)
        Y = pd.DataFrame(SY, columns=LY)


        trainX, testX, trainY,  testY = \
                train_test_split(X,Y, test_size= 0.2, random_state=432)


        print( "Train Dimesions ", trainX.shape )
        print( "Test Dimensions ", testX.shape )

        # create neural network model
        model = Sequential()
        model.add(Dense(1, input_dim=4, activation='linear'))
        model.add(Dense(50, activation='linear'))
        model.add(Dense(50, activation='tanh'))
        model.add(Dense(50, activation='linear'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mean_squared_error", optimizer="adam")


        # train the model
        model.fit(trainX, trainY,epochs=10,verbose=0,shuffle=True)

        mse = model.evaluate(testX,testY, verbose=1)
        print('Mean Squared Error: ', mse)

        predictions = model.predict(testX)
        sp = scalerY.inverse_transform( predictions )
        rv = scalerY.inverse_transform( testY )

        pairs = np.array(list(zip(sp, rv)))

        for pair in pairs:
            pre = pair[0][0]
            rea = pair[1][0]
            data.write("{},{}\n".format(pre, rea))

        plt.figure()

        plt.plot(range(rv.shape[0]),rv,'r-',label='actual')
        plt.plot(range(sp.shape[0]),sp,'k--',label='predict')
        plt.legend(loc='best')
        name = 'results{}_{}.eps'.format(config, test+1)
        plt.savefig(name , format='eps', dpi=1000)
        res.write("{},{},{}\n".format(test+1, mse,name))
        data.close()
    res.close()
