import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


nameFile = "../data/cleaned.xlsx"



#The model
def multilayer_percepetron(x, weights, biases):
    # Hidden layer with sigmoid activation

    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights["h3"]), biases["b3"])
    layer_3 = tf.nn.sigmoid(layer_3)

    #Output layer
    out_layer = tf.matmul(layer_3, weights["out"])+ biases["out"]

    return out_layer

def readData():
    data = pd.read_excel(nameFile, index_col=0)
    sns.pairplot(data[[ "Cuartil_2017", "Publisher", "DocumentType", "Citedby"]], diag_kind="kde")
    #plt.show()
    info = data[data.columns[0:4]].values
    labels = np.array(data[data.columns[4]])
    labels = labels.reshape(labels.shape[0],1)
    return info, labels

if __name__ == "__main__":
    X, Y = readData();
    #print(X,Y)


    X, Y = shuffle( X, Y , random_state=1)



    #print(X,Y)
    #divide the dataset
    trainX, testX, trainY,  testY = train_test_split(X,Y, test_size= 0.2, random_state=432)



    print( "Train Dimesions ", trainX.shape )
    print( "Test Dimensions ", testX.shape )

    # configure important parameters
    n_class = 1
    n_dim = X.shape[1]
    learning_rate = 0.0001
    training_epochs = 100

    #define umber of hiden layers and neurons
    n_hidden_1 = 30
    n_hidden_2 = 30
    n_hidden_3 = 30

    none = None

    x = tf.placeholder( tf.float32, [none, n_dim])
    W = tf.Variable(tf.zeros([n_dim, n_class]))
    b = tf.Variable(tf.zeros([n_class]))
    y_ = tf.placeholder(tf.float32, [none,n_class  ])

    weights = {
        "h1": tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
        "h2": tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        "h3": tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
        "out": tf.Variable(tf.truncated_normal([n_hidden_3, n_class]))
    }
    biases = {
        "b1": tf.Variable(tf.truncated_normal([n_hidden_1])),
        "b2": tf.Variable(tf.truncated_normal([n_hidden_2])),
        "b3": tf.Variable(tf.truncated_normal([n_hidden_3])),
        "out": tf.Variable(tf.truncated_normal([n_class]))
    }

    #Initialize all variables
    init = tf.global_variables_initializer()

    #Save the model
    saver = tf.train.Saver()

    #Call your model defined
    y = multilayer_percepetron(x, weights=weights, biases=biases)

    #Define the cost function and optimizer
    cost_function = tf.reduce_mean(y)
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    sess = tf.Session()

    sess.run(init)

    #Calculate the cost an accuray for each epoch

    mse_history = []
    accuracy_history = []
    cost_history = np.empty(shape=[1], dtype= float)

    print( "trainy.shape = ", trainY.shape)

    for epoch in range(training_epochs):
        sess.run(training_step, {x: trainX, y_: trainY})
        cost = sess.run(cost_function,  feed_dict={x:trainX, y_:trainY})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
        accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        pred_y = sess.run(y, feed_dict={x:testX})
        mse = tf.reduce_mean(tf.square(pred_y-testY))
        mse_ = sess.run(mse)
        mse_history.append(mse_)
        accuracy = sess.run(accuracy,  feed_dict={x:trainX, y_:trainY} )
        accuracy_history.append(accuracy)

        print("Epoch = {}, Cost = {}, MSE = {}, Train accuracy = {}".format(epoch, cost, mse_, accuracy))
    sess.close()
