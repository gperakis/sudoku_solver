from typing import Optional

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Activation, Flatten,
                                     Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sudoku_solver.config import ModelConf


class SudokuNet:
    def __init__(self, width=28, height=28, depth=1, classes=10):
        """

        :param width:
        :param height:
        :param depth:
        :param classes:
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.classes = classes

        self._model: Optional[Sequential] = None
        self.le: Optional[LabelBinarizer] = None

        # initialize the initial learning rate, number of epochs to train
        # for, and batch size
        self.init_lr = ModelConf.INIT_LR
        self.epochs = ModelConf.EPOCHS
        self.batch_size = ModelConf.BS

        self.trainData = None
        self.testData = None
        self.trainLabels = None
        self.testLabels = None

    @property
    def model(self):
        if self._model is None:
            self._model = self.build()
        return self._model

    def build(self):
        # initialize the model
        model = Sequential()
        inputShape = (self.height, self.width, self.depth)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # second set of FC => RELU layers
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        # initialize the optimizer and model
        print("[INFO] compiling model...")
        opt = Adam(lr=self.init_lr)

        model.compile(loss="categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

        # return the constructed network architecture
        return model

    def load(self):
        # grab the MNIST dataset
        print("[INFO] accessing MNIST...")
        ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

        # add a channel (i.e., grayscale) dimension to the digits
        trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
        testData = testData.reshape((testData.shape[0], 28, 28, 1))

        # scale data to the range of [0, 1]
        self.trainData = trainData.astype("float32") / 255.0
        self.testData = testData.astype("float32") / 255.0

        # convert the labels from integers to vectors
        self.le = LabelBinarizer()
        self.trainLabels = self.le.fit_transform(trainLabels)
        self.testLabels = self.le.transform(testLabels)

    def fit(self):
        # train the network
        print("[INFO] training network...")
        H = self.model.fit(self.trainData, self.trainLabels,
                           validation_data=(self.testData, self.testLabels),
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           verbose=1)
        return H

    def evaluate(self):
        # evaluate the network
        print("[INFO] evaluating network...")
        predictions = self.model.predict(self.testData)
        print(classification_report(
            self.testLabels.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[str(x) for x in self.le.classes_]))

    def save(self):
        # serialize the model to disk
        print("[INFO] serializing digit model...")
        self.model.save('digit_classifier.h5', save_format="h5")

    def run(self):
        self.load()
        history = self.fit()
        self.evaluate()
        return history


if __name__ == "__main__":
    sudoku_net = SudokuNet()
    sudoku_net.run()

    # import argparse

    # # USAGE
    # # python train_digit_classifier.py --model output/digit_classifier.h5
    #
    # # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-m", "--model", required=True,
    #                 help="path to output model after training")
    # args = vars(ap.parse_args())
