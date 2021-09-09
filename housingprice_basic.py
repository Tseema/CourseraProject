import numpy as np
import tensorflow as tf
from tensorflow import keras
#from AppKit import NSSound


# coursera lab exercise 2
#1+x*50 50+50x

def house_model(y_new):
    # input
    xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0], dtype=float)
    #xs = [1.0,2.0,3.0,4.0,5.0,6.0]
    # data set for output
    ys = np.array([100.0,150.0,200.0,250.0,300.0,350.0,400.0,450.0,500,550.0], dtype=float)
    #ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5,4.0,4.5,5.0,5.5], dtype=float)
   # model = tf.keras.Sequential(
    #    [tf.keras.layers.Dense(input_shape(1,), activation='relu'),
     #   tf.keras.layers.Dense(1,activation='softmax')]
   # )

    # training the model for the above input and output
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    #model.compile(optimizer="adma",loss="mse")
    model.compile(optimizer='sgd', loss='mean_squared_error')
    #model.compile(optimizer='sgd', loss='mean_squared_error') 394.54
    #model.compile(optimizer='Adamax', loss='mean_squared_error') 43.13
    #model.compile(optimizer='Nadam', loss='mean_squared_error') 46.25
    #model.compile(optimizer='RMSprop', loss='mean_squared_error') 51.74
    #model.compile(optimizer='Adam', loss='mean_squared_error') 27.39
    model.fit(xs,ys, epochs=500)
    return model.predict(y_new)[0]

# test data to find if the training for the data was a success
prediction = house_model([7.0])

print("value = ")
print(prediction[0])