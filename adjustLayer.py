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

# slice the data
train = data[0:120, :]   # length 120
test = data[120:, :]     # length 24

def prepare_data(data, lags=1):
    #Create lagged data from an input time series
    X, y = [], []
    for row in range(len(data) - lags - 1):
        a = data[row:(row + lags), 0]
        X.append(a)
        y.append(data[row + lags, 0])
    return np.array(X), np.array(y)

# prepare the data
lags = 1
X_train, y_train = prepare_data(train, lags)
X_test, y_test = prepare_data(test, lags)
y_true = y_test

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setGeometry(600, 200, 1200, 600)
        self.setWindowTitle("Hidden Layer Number Adjuster")

        self.lineEdit_0 = QLineEdit()
        self.lineEdit_1 = QLineEdit()
        self.label_0 = QLabel()
        self.label_1 = QLabel()
        self.label_2 = QLabel()
        self.label_3 = QLabel()

        self.pushButton = QPushButton("Calculate")
        self.pushButton.clicked.connect(self.calculateButton)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(100)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.canvas)
        leftLayout.addWidget(self.progress)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.label_0)
        rightLayout.addWidget(self.label_1)
        rightLayout.addWidget(self.lineEdit_0)
        rightLayout.addWidget(self.label_2)
        rightLayout.addWidget(self.lineEdit_1)
        rightLayout.addWidget(self.pushButton)
        rightLayout.addWidget(self.label_3)
        rightLayout.addStretch(1)

        layout = QHBoxLayout()
        layout.addLayout(leftLayout)
        layout.addLayout(rightLayout)
        layout.setStretchFactor(leftLayout, 1)
        layout.setStretchFactor(rightLayout, 0)
        self.label_0.setText("Please enter the range of the\nhidden layers will tested:")
        self.label_1.setText("From")
        self.label_2.setText("to")
        self.label_3.setText("\n")
        self.progress.setValue(0)

        self.setLayout(layout)

    def onButtonClick(self):
        self.calc = External()
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.start()

    def onCountChanged(self, value):
        self.progress.setValue(value)

    def neurelNetwork(self, numLayer=1):
        mdl0=Sequential()
        mdl0.add(Dense(numLayer, input_dim=lags, activation='relu'))
        mdl0.add(Dense(1))
        mdl0.compile(loss='mean_squared_error', optimizer='adam')
        return mdl0

    def calculateButton(self):
        self.progress.setValue(0)
        x = str(self.lineEdit_0.text())
        y = str(self.lineEdit_1.text())

        try:
            x = int(x)
            y = int(y)

        except ValueError:
            x = 1
            y = 3

        trainScores = [];

        mdl1 = []

        for i in range(0,(y-x+1)):

            print("Testing with #",i+x)
            mdl1.append(self.neurelNetwork(i+x))
            mdl1[i].fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

            train_score = mdl1[i].evaluate(X_train, y_train, verbose=0)
            if(train_score > 2500):
                trainScores.append(0)
            else:
                trainScores.append(train_score)
            print(train_score)
            print("\n")
            self.progress.setValue((100/(y-x+1))*(i+1)-10)

        bestLayerNumber = trainScores.index(max(trainScores))+x
        self.label_3.setText("\nBest layer number is %d\nIt's train score is %.2f"%(bestLayerNumber, max(trainScores)))

        mdl = self.neurelNetwork(bestLayerNumber)
        mdl.fit(X_train, y_train, epochs=200, batch_size=2, verbose=0)

        # generate predictions for training
        train_predict = mdl.predict(X_train)
        test_predict = mdl.predict(X_test)

        self.progress.setValue(100)

        # shift train predictions for plotting
        train_predict_plot = np.empty_like(data)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[lags: len(train_predict) + lags, :] = train_predict

        # shift test predictions for plotting
        test_predict_plot = np.empty_like(data)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(train_predict)+(lags*2)+1:len(data)-1, :] = test_predict

        # plot baseline and predictions
        ax = self.fig.add_subplot(111)
        ax.clear()
        mse = ((y_test.reshape(-1, 1) - test_predict.reshape(-1, 1)) ** 2).mean()
        ax.set_title('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, math.sqrt(mse)))
        ax.plot(y_test.reshape(-1, 1), label='Observed', color='#006699')
        ax.plot(test_predict.reshape(-1, 1), label='Prediction', color='#ff0066')
        ax.legend(loc='best');
        ax.grid()

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
