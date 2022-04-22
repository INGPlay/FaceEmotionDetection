import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow import optimizers

TRAINNUM = 28709
TESTNUM = 7178

NUMCLASSES = 7

def main():
    raw_data_file_name = './fer2013.csv'
    raw_data = pd.read_csv(raw_data_file_name)

    raw_data.info()
    print("-----------------------------")
    print(raw_data.head())
    print("-----------------------------")
    print(raw_data["Usage"].value_counts())
    print("-----------------------------")

    xTrain = np.empty([TRAINNUM, 48, 48, 3])
    for i in range(0, TRAINNUM):
        image = raw_data["pixels"][i]
        val = image.split(' ')

        pixels = np.array(val, np.float32)
        pixels /= 255
        pixels = pixels.reshape(48, 48)

        for j in range(48) :
            for k in range(48):
                xTrain[i][j][k][0] = pixels[j][k]
                xTrain[i][j][k][1] = pixels[j][k]
                xTrain[i][j][k][2] = pixels[j][k]

    yTrain = []
    for i in range(0, TRAINNUM):
        index = raw_data["emotion"][i]
        yTrain.append(index)
    yTrain = keras.utils.to_categorical(yTrain, NUMCLASSES)


    xTest = np.empty([TESTNUM, 48, 48, 3])
    for i in range(0, TESTNUM):
        image = raw_data["pixels"][i + TRAINNUM]
        val = image.split(' ')
        pixels = np.array(val, np.float32)
        pixels /= 255
        pixels = pixels.reshape(48, 48)

        for j in range(48) :
            for k in range(48):
                xTest[i][j][k][0] = pixels[j][k]
                xTest[i][j][k][1] = pixels[j][k]
                xTest[i][j][k][2] = pixels[j][k]

    yTest = []
    for i in range(TRAINNUM, TRAINNUM+TESTNUM):
        index = raw_data["emotion"][i]
        yTest.append(index)
    yTest = keras.utils.to_categorical(yTest, NUMCLASSES)

    vgg16 = keras.applications.VGG16(include_top=False, input_shape=(48, 48, 3), pooling='avg', weights='imagenet')

    model = keras.Sequential([
        keras.layers.Dense(units=256, input_shape=(512,), activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units = NUMCLASSES, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])

    xTrainPredicted = vgg16.predict(xTrain)
    xTestPredicted = vgg16.predict(xTest)

    model.fit(xTrainPredicted, yTrain, validation_data=(xTrainPredicted, yTrain), epochs=100, batch_size=32)
    model.evaluate(xTestPredicted, yTest, batch_size=32)

    inputs = keras.layers.Input(shape=(48, 48, 3))
    vggOutput = vgg16(inputs)
    modelOutput = model(vggOutput)

    finalModel = keras.Model(inputs=inputs, outputs=modelOutput)
    finalModel.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])

    finalModel.evaluate(xTrain, yTrain, batch_size=32)
    finalModel.evaluate(xTest, yTest, batch_size=32)

main()