#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
from time import sleep
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pandas import Series, DataFrame
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QTimer
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# set seed
np.random.seed(7)

# import data set
df = pd.read_csv('./passengers.csv', sep=';', parse_dates=True, index_col=0)
data = df.values

# using keras often requires the data type float32
data = data.astype('float32')

lags = 1
dataParserFlag = 0

X_split=0
y_split=0

class Features():
    def neurelNetwork(self, numLayer=4, inputDim=1, outputDim=1):
        print("\n*** Creating a new model with #{} hidden layers. ***\n".format(numLayer))
        mdl0=Sequential()
        mdl0.add(Dense(numLayer, input_dim=inputDim, activation='relu'))
        mdl0.add(Dense(outputDim))
        mdl0.compile(loss='mean_squared_error', optimizer='adam')
        return mdl0

    def dataParser(self, trainRatio=85, n_steps_in=4, n_steps_out=3):
        print("Parsing data with the percent of {}".format(trainRatio))
        window.x = int((144/100)*trainRatio);
        train = data[0:window.x, :]
        test = data[window.x:, :]

        global X_split, y_split
        X_split, y_split = self.splitSequence(train, n_steps_in, n_steps_out)

    def splitSequence(self, data, n_steps_in=4, n_steps_out=3):
    	X, y = list(), list()
    	for i in range(len(data)):
    		end_ix = i + n_steps_in
    		out_end_ix = end_ix + n_steps_out
    		if out_end_ix > len(data):
    			break
    		seq_x, seq_y = data[i:end_ix, 0], data[end_ix:out_end_ix, 0]
    		X.append(seq_x)
    		y.append(seq_y)
    	return np.array(X), np.array(y)

feat = Features()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Time Series Forecasting with MLP Tool")

        self.lineEdit = QLineEdit()

        self.label = QLabel()

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setFocusPolicy(Qt.StrongFocus)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)

        self.pushButton = QPushButton("Predict")
        self.pushButton.clicked.connect(self.predictButton)

        self.x = 0
        self.ratio = 85
        self.layerNumber = 120
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.label)
        rightLayout.addWidget(self.lineEdit)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.label.setText("N-step Prediction")

        self.setLayout(layout)

    def predictButton(self):
        global data, X_train, y_train, X_test, y_test, X_split, y_split

        nStep = str(self.lineEdit.text())
        X_input = np.array([], dtype='float32')
        prediction_expected = np.array([], dtype='float32')

        try:
            nStep = int(nStep)

        except ValueError:
            nStep = 3

        feat.dataParser(self.ratio, nStep+1, nStep)

        mdl = feat.neurelNetwork(self.layerNumber, nStep+1, nStep)
        mdl.fit(X_split, y_split, epochs=200, batch_size=2, verbose=0)

        X_input = np.append(X_input, data[window.x-(nStep+1):window.x, 0])
        prediction_expected = np.append(prediction_expected, data[window.x:window.x+nStep, 0])
        X_input = X_input.reshape((1, nStep+1))
        prediction_result = mdl.predict(X_input)

        # plot baseline and predictions
        ax = self.fig.add_subplot(111)
        ax.clear()
        mse = ((prediction_expected.reshape(-1, 1) - prediction_result.reshape(-1, 1)) ** 2).mean()
        ax.set_title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
        ax.plot(prediction_expected.reshape(-1, 1), label='Observed', color='#006699')
        ax.plot(prediction_result.reshape(-1, 1), label='Prediction', color='#ff0066')
        ax.legend(loc='best');
        ax.grid()

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
